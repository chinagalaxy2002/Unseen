"""DQ-CGP-v3: sparse hierarchical prompting for FlashVTG's point pyramid.

FlashVTG points are dense samples from a temporal feature pyramid, not
independent DETR instance slots. This module keeps temporal binding and
feature reconstruction point-wise, while assigning the primary prompt
composition once per pyramid level.  V3 uses a sparse, piecewise-
differentiable top-k router and makes the point contribution explicit in the
probability simplex, rather than relying on a vanishing logit perturbation.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FlashPointHSDQCGPOutput(NamedTuple):
    """Diagnostics retained from the most recent active forward pass."""

    adapted_state: Tensor
    temporal_logits: Tensor
    temporal_attention: Tensor
    temporal_context: Tensor
    level_pool_logits: Tensor
    level_pool_attention: Tensor
    level_prototypes: Tensor
    local_prototypes: Tensor
    local_pool_attention: Tensor
    level_router_logits: Tensor
    level_basis_weights: Tensor
    point_context_residual: Tensor
    point_router_logits: Tensor
    point_basis_weights: Tensor
    point_correction_weights: Tensor
    router_logits: Tensor
    basis_weights: Tensor
    prompt_sequence: Tensor
    prompt_attention: Tensor
    pooled_prompt: Tensor
    frf_feature: Tensor
    residual_update: Tensor


class FlashPointHSDQCGP(nn.Module):
    """HS-DQ-CGP refinement for FlashVTG multi-scale dense candidates.

    Level router (default): ``[LN(c_bar_l); LN(e); LN(c_bar_l)*LN(e)] -> N``.
    The explicit level embedding remains in temporal binding, but is excluded
    from the default router to prevent a fixed ``level -> basis`` shortcut.
    Point router: ``[c_local_lp - c_bar_l; h_lp] -> N``.  Its sparse
    distribution is mixed with the level distribution as
    ``(1-lambda) * w_level + lambda * w_point``.  Thus ``lambda`` is an
    actual probability-space contribution, not an opaque logit scale.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        point_router_hidden_dim: int = 128,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        point_mixture_ratio: float = 0.10,
        router_logit_scale: float = 2.0,
        router_output_init_std: float = 1e-3,
        use_level_embedding_in_router: bool = False,
        beta: float = 0.05,
        num_levels: int = 4,
        locality_strength: float = 0.0,
        routing_topk: int = 4,
        local_prototype_radius: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_basis <= 0 or prompt_length <= 0:
            raise ValueError("num_basis and prompt_length must be positive")
        if router_hidden_dim <= 0 or point_router_hidden_dim <= 0:
            raise ValueError("router hidden dimensions must be positive")
        if frf_hidden_dim <= 0:
            raise ValueError("frf_hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= point_mixture_ratio <= 1:
            raise ValueError("point_mixture_ratio must lie in [0, 1]")
        if router_logit_scale <= 0:
            raise ValueError("router_logit_scale must be positive")
        if router_output_init_std < 0:
            raise ValueError("router_output_init_std must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if locality_strength < 0:
            raise ValueError("locality_strength must be non-negative")
        if routing_topk <= 0:
            raise ValueError("routing_topk must be positive")
        if local_prototype_radius < 0:
            raise ValueError("local_prototype_radius must be non-negative")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.num_levels = int(num_levels)
        self.temperature = float(temperature)
        self.locality_strength = float(locality_strength)
        self.point_mixture_ratio = float(point_mixture_ratio)
        self.router_logit_scale = float(router_logit_scale)
        self.router_output_init_std = float(router_output_init_std)
        self.use_level_embedding_in_router = bool(use_level_embedding_in_router)
        self.routing_topk = min(int(routing_topk), self.num_basis)
        self.local_prototype_radius = int(local_prototype_radius)

        self.register_buffer("beta", torch.tensor(float(beta)))
        self._beta_is_zero = float(beta) == 0.0

        # Stage 1: point-wise temporal binding.
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.point_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.level_embedding = nn.Embedding(num_levels, hidden_dim)

        # Stage 2: query-conditioned attention pooling inside each level.
        self.prototype_context_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prototype_semantic_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prototype_candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prototype_score = nn.Linear(hidden_dim, 1, bias=False)

        # Stages 3--4: sparse level routing plus probability-space point mix.
        self.router_prototype_norm = nn.LayerNorm(hidden_dim)
        self.router_semantic_norm = nn.LayerNorm(hidden_dim)
        self.level_router = nn.Sequential(
            nn.Linear(3 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )
        self.point_router = nn.Sequential(
            nn.Linear(2 * hidden_dim, point_router_hidden_dim),
            nn.ReLU(),
            nn.Linear(point_router_hidden_dim, num_basis),
        )
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )

        # Stages 5--6: prompt-token attention and point-wise FRF [p,e,c,h].
        self.frf_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(4 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_norm = nn.LayerNorm(hidden_dim)

        self.last_output: Optional[FlashPointHSDQCGPOutput] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        # Near-uniform logits prevent arbitrary large early routing scores. The
        # top-k support is still data-dependent and learns through its selected
        # logits.
        nn.init.normal_(
            self.level_router[-1].weight,
            mean=0.0,
            std=self.router_output_init_std,
        )
        nn.init.zeros_(self.level_router[-1].bias)
        nn.init.normal_(
            self.point_router[-1].weight,
            mean=0.0,
            std=self.router_output_init_std,
        )
        nn.init.zeros_(self.point_router[-1].bias)

    def _bound_router_logits(self, logits: Tensor) -> Tensor:
        """Bound logit magnitude so softmax gradients cannot saturate early."""

        scale = self.router_logit_scale
        return scale * torch.tanh(logits / scale)

    @staticmethod
    def _topk_softmax(logits: Tensor, topk: int) -> Tensor:
        """Sparse top-k softmax with gradients through selected logits.

        ``torch.topk`` makes the support piecewise constant, while the values
        inside that support retain ordinary softmax gradients.  This is the
        standard sparse-MoE interpretation of differentiable top-k routing.
        """

        if logits.shape[-1] <= topk:
            return torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, k=topk, dim=-1)
        masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        masked_logits = masked_logits.scatter(-1, top_indices, top_values)
        return torch.softmax(masked_logits, dim=-1)

    def set_beta(self, beta: float) -> None:
        """Set fixed residual strength, including the exact-identity ablation."""

        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta.fill_(float(beta))
        self._beta_is_zero = float(beta) == 0.0

    def clear_diagnostics(self) -> None:
        self.last_output = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._beta_is_zero = float(self.beta.detach().cpu()) == 0.0

    def _check_inputs(
        self,
        candidate_state: Tensor,
        video_memory: Tensor,
        video_valid_mask: Tensor,
        candidate_valid_mask: Tensor,
        query_semantic: Tensor,
        point_metadata: Tensor,
        level_ids: Tensor,
    ) -> None:
        if candidate_state.ndim != 3:
            raise ValueError("candidate_state must have shape [batch, points, hidden_dim]")
        if video_memory.ndim != 3:
            raise ValueError("video_memory must have shape [batch, time, hidden_dim]")
        if query_semantic.ndim != 2:
            raise ValueError("query_semantic must have shape [batch, hidden_dim]")

        batch_size, num_points, candidate_dim = candidate_state.shape
        memory_batch, video_length, memory_dim = video_memory.shape
        if memory_batch != batch_size or query_semantic.shape[0] != batch_size:
            raise ValueError("candidate, memory, and semantic batch sizes must match")
        if (
            candidate_dim != self.hidden_dim
            or memory_dim != self.hidden_dim
            or query_semantic.shape[1] != self.hidden_dim
        ):
            raise ValueError(f"all feature dimensions must equal hidden_dim={self.hidden_dim}")
        if video_valid_mask.shape != (batch_size, video_length):
            raise ValueError("video_valid_mask must have shape [batch, time]")
        if candidate_valid_mask.shape != (batch_size, num_points):
            raise ValueError("candidate_valid_mask must have shape [batch, points]")
        if point_metadata.shape != (num_points, 4):
            raise ValueError("point_metadata must have shape [points, 4]")
        if level_ids.shape != (num_points,):
            raise ValueError("level_ids must have shape [points]")
        if level_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("level_ids must be an integer tensor")
        if bool((level_ids < 0).any()) or bool((level_ids >= self.num_levels).any()):
            raise ValueError("level_ids contain an out-of-range pyramid level")

    @staticmethod
    def _masked_softmax(logits: Tensor, valid_mask: Tensor, dim: int = -1) -> Tensor:
        """Softmax with an all-zero output for an entirely invalid row."""

        valid = valid_mask.bool()
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(masked_logits, dim=dim)
        attention = attention * valid.to(attention.dtype)
        denominator = attention.sum(dim=dim, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )
        return attention / denominator

    def _point_features(
        self,
        point_metadata: Tensor,
        level_ids: Tensor,
        valid_lengths: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        centers = point_metadata[:, 0].to(dtype)
        strides = point_metadata[:, 3].to(dtype).clamp_min(1.0)
        denominator = valid_lengths.to(dtype).clamp_min(1.0).unsqueeze(1)
        coordinates = torch.stack(
            (centers.unsqueeze(0) / denominator, strides.unsqueeze(0) / denominator),
            dim=-1,
        )
        point_feature = self.point_projection(coordinates)
        return point_feature + self.level_embedding(level_ids).to(dtype).unsqueeze(0)

    def _level_prototypes(
        self,
        temporal_context: Tensor,
        candidate_state: Tensor,
        query_semantic: Tensor,
        candidate_valid_mask: Tensor,
        level_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute query-conditioned semantic prototype for every level."""

        point_scores = self.prototype_score(
            torch.tanh(
                self.prototype_context_projection(temporal_context)
                + self.prototype_semantic_projection(query_semantic).unsqueeze(1)
                + self.prototype_candidate_projection(candidate_state)
            )
        ).squeeze(-1)
        level_membership = F.one_hot(
            level_ids, num_classes=self.num_levels
        ).transpose(0, 1).bool()
        level_valid_mask = candidate_valid_mask.unsqueeze(1) & level_membership.unsqueeze(0)
        level_pool_logits = point_scores.unsqueeze(1).expand(-1, self.num_levels, -1)
        level_pool_attention = self._masked_softmax(
            level_pool_logits, level_valid_mask, dim=-1
        )
        level_prototypes = torch.einsum(
            "blp,bpd->bld", level_pool_attention, temporal_context
        )
        return level_prototypes, level_pool_logits, level_pool_attention, point_scores

    def _local_prototypes(
        self,
        temporal_context: Tensor,
        point_scores: Tensor,
        candidate_valid_mask: Tensor,
        point_metadata: Tensor,
        level_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Query-conditioned local occurrence prototype for every point.

        A global level prototype can merge distinct occurrences.  Each point
        therefore pools only same-level neighbours within a fixed number of
        stride units.  The pool uses the Stage-2 query-conditioned scores, so
        it adds locality without another free routing network.
        """

        centers = point_metadata[:, 0].to(temporal_context.dtype)
        strides = point_metadata[:, 3].to(temporal_context.dtype).clamp_min(1.0)
        distance = (centers[:, None] - centers[None, :]).abs()
        radius = self.local_prototype_radius * strides[:, None]
        same_level = level_ids[:, None].eq(level_ids[None, :])
        local_neighbourhood = same_level & (distance <= radius)
        valid_pairs = (
            candidate_valid_mask.unsqueeze(2)
            & candidate_valid_mask.unsqueeze(1)
            & local_neighbourhood.unsqueeze(0)
        )
        local_logits = point_scores.unsqueeze(1).expand(
            -1, temporal_context.shape[1], -1
        )
        local_attention = self._masked_softmax(local_logits, valid_pairs, dim=-1)
        local_prototypes = torch.einsum(
            "bpq,bqd->bpd", local_attention, temporal_context
        )
        return local_prototypes, local_attention

    def forward(
        self,
        candidate_state: Tensor,
        video_memory: Tensor,
        video_valid_mask: Tensor,
        candidate_valid_mask: Tensor,
        query_semantic: Tensor,
        point_metadata: Tensor,
        level_ids: Tensor,
    ) -> Tensor:
        """Return shape-preserving HS-DQ-CGP-refined pyramid candidates."""

        # This exact early return makes beta=0 a strict FlashVTG baseline.
        if self._beta_is_zero:
            self.last_output = None
            return candidate_state

        self._check_inputs(
            candidate_state, video_memory, video_valid_mask, candidate_valid_mask,
            query_semantic, point_metadata, level_ids,
        )
        video_valid_mask = video_valid_mask.bool()
        candidate_valid_mask = candidate_valid_mask.bool()
        valid_lengths = video_valid_mask.sum(dim=1)

        # Stage 1: each temporal candidate keeps its own binding distribution.
        point_feature = self._point_features(
            point_metadata, level_ids, valid_lengths, candidate_state.dtype
        )
        candidate_key = self.candidate_projection(self.candidate_norm(candidate_state))
        semantic_key = self.semantic_projection(query_semantic).unsqueeze(1)
        temporal_query = candidate_key + semantic_key + point_feature
        temporal_key = self.memory_key_projection(self.memory_norm(video_memory))
        temporal_logits = torch.einsum(
            "bpd,btd->bpt", temporal_query, temporal_key
        ) / math.sqrt(self.hidden_dim)
        if self.locality_strength > 0:
            frame_positions = torch.arange(
                video_memory.shape[1], device=video_memory.device, dtype=temporal_logits.dtype
            )
            centers = point_metadata[:, 0].to(temporal_logits.dtype)
            strides = point_metadata[:, 3].to(temporal_logits.dtype).clamp_min(1.0)
            relative_distance = (
                frame_positions.view(1, 1, -1) - centers.view(1, -1, 1)
            ).abs() / strides.view(1, -1, 1)
            temporal_logits = temporal_logits - self.locality_strength * relative_distance
        temporal_attention = self._masked_softmax(
            temporal_logits, video_valid_mask.unsqueeze(1), dim=-1
        )
        temporal_context = torch.einsum(
            "bpt,btd->bpd", temporal_attention, self.memory_value_projection(video_memory)
        )

        # Stage 2: level semantic prototypes, pooled over valid level points.
        level_prototypes, level_pool_logits, level_pool_attention, point_scores = self._level_prototypes(
            temporal_context, candidate_state, query_semantic, candidate_valid_mask, level_ids
        )
        local_prototypes, local_pool_attention = self._local_prototypes(
            temporal_context, point_scores, candidate_valid_mask, point_metadata, level_ids
        )

        # Stage 3: main routing is only [B, L, N], using a global basis bank.
        semantic_by_level = query_semantic.unsqueeze(1).expand(-1, self.num_levels, -1)
        normalized_prototype = self.router_prototype_norm(level_prototypes)
        normalized_semantic = self.router_semantic_norm(semantic_by_level)
        if self.use_level_embedding_in_router:
            level_index = torch.arange(self.num_levels, device=candidate_state.device)
            router_third_feature = self.level_embedding(level_index).to(candidate_state.dtype)
            router_third_feature = router_third_feature.unsqueeze(0).expand(
                candidate_state.shape[0], -1, -1
            )
        else:
            # Multiplicative interaction makes the routing decision explicitly
            # query-conditioned instead of supplying a scale-only shortcut.
            router_third_feature = normalized_prototype * normalized_semantic
        raw_level_router_logits = self.level_router(
            torch.cat(
                (normalized_prototype, normalized_semantic, router_third_feature),
                dim=-1,
            )
        )
        level_router_logits = self._bound_router_logits(raw_level_router_logits)
        level_basis_weights = self._topk_softmax(
            level_router_logits / self.temperature, self.routing_topk
        )

        # Stage 4: local occurrence routing is mixed in probability space.
        point_context_residual = local_prototypes - level_prototypes[:, level_ids]
        raw_point_router_logits = self.point_router(
            torch.cat((point_context_residual, candidate_state), dim=-1)
        )
        point_router_logits = self._bound_router_logits(raw_point_router_logits)
        point_basis_weights = self._topk_softmax(
            point_router_logits / self.temperature, self.routing_topk
        )
        level_weights_by_point = level_basis_weights[:, level_ids]
        basis_weights = (
            (1.0 - self.point_mixture_ratio) * level_weights_by_point
            + self.point_mixture_ratio * point_basis_weights
        )
        point_correction_weights = basis_weights - level_weights_by_point
        # Retained as a diagnostic tensor for external consumers.  The final
        # routing is no longer represented by one shared logit tensor.
        router_logits = level_router_logits[:, level_ids]

        # Stage 5: shared-basis composition followed by prompt-token attention.
        prompt_sequence = torch.einsum("bpn,nld->bpld", basis_weights, self.basis_prompts)
        prompt_logits = torch.einsum(
            "bpd,bpld->bpl", candidate_key + semantic_key, prompt_sequence
        ) / math.sqrt(self.hidden_dim)
        prompt_attention = torch.softmax(prompt_logits, dim=-1)
        pooled_prompt = torch.einsum("bpl,bpld->bpd", prompt_attention, prompt_sequence)

        # Stage 6: candidate-specific FRF and fixed-beta residual, without a gate.
        semantic_by_point = query_semantic.unsqueeze(1).expand(-1, candidate_state.shape[1], -1)
        frf_feature = self.frf(
            torch.cat(
                (
                    pooled_prompt,
                    semantic_by_point,
                    self.frf_context_projection(temporal_context),
                    candidate_state,
                ),
                dim=-1,
            )
        )
        residual_update = self.residual_norm(self.residual_projection(frf_feature))
        residual_update = residual_update * candidate_valid_mask.unsqueeze(-1).to(residual_update.dtype)
        adapted_state = candidate_state + self.beta.to(candidate_state.dtype) * residual_update

        self.last_output = FlashPointHSDQCGPOutput(
            adapted_state, temporal_logits, temporal_attention, temporal_context,
            level_pool_logits, level_pool_attention, level_prototypes,
            local_prototypes, local_pool_attention, level_router_logits,
            level_basis_weights, point_context_residual, point_router_logits,
            point_basis_weights, point_correction_weights, router_logits, basis_weights,
            prompt_sequence, prompt_attention, pooled_prompt, frf_feature, residual_update,
        )
        return adapted_state


# Compatibility aliases only for import users within the HS experiment tree.
FlashPointDQCGP = FlashPointHSDQCGP
FlashPointDQCGPOutput = FlashPointHSDQCGPOutput

__all__ = [
    "FlashPointHSDQCGP",
    "FlashPointHSDQCGPOutput",
    "FlashPointDQCGP",
    "FlashPointDQCGPOutput",
]
