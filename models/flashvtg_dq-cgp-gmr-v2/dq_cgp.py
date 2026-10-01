"""Candidate-conditioned DQ-CGP for FlashVTG's temporal point pyramid.

The original DQ-CGP uses DETR decoder queries as its candidate-instance axis.
FlashVTG has no decoder queries, so this variant uses the locations from its
multi-scale point pyramid instead.  Every point independently binds to the
encoded video, composes a prompt from a shared basis bank, and receives a
small residual update before the unchanged prediction heads.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn


class FlashPointDQCGPOutput(NamedTuple):
    """Diagnostics retained from the most recent active forward pass."""

    adapted_state: Tensor
    temporal_logits: Tensor
    temporal_attention: Tensor
    temporal_context: Tensor
    basis_weights: Tensor
    prompt_sequence: Tensor
    prompt_attention: Tensor
    pooled_prompt: Tensor
    frf_feature: Tensor
    update_gate: Tensor
    residual_update: Tensor


class FlashPointDQCGP(nn.Module):
    """Apply DQ-CGP independently to FlashVTG multi-scale point candidates.

    Args:
        hidden_dim: Common FlashVTG feature dimension.
        num_basis: Number of shared prompt bases.
        prompt_length: Tokens in each prompt basis.
        router_hidden_dim: Hidden dimension of the RCG router.
        frf_hidden_dim: Hidden dimension of the feature reconstruction MLP.
        temperature: Softmax temperature used by basis routing.
        beta: Fixed residual injection strength.  ``beta=0`` is an exact,
            computation-free identity path.
        num_levels: Number of FlashVTG pyramid levels.
        locality_strength: Optional soft anchor-distance bias.  Zero disables
            it and recovers the global temporal binding used by DQ-CGP.

    Tensor contract:
        candidate_state: ``[batch, num_points, hidden_dim]``.
        video_memory: ``[batch, video_length, hidden_dim]``.
        video_valid_mask: ``[batch, video_length]``; True means valid.
        candidate_valid_mask: ``[batch, num_points]``; True means valid.
        query_semantic: ``[batch, hidden_dim]``.
        point_metadata: ``[num_points, 4]`` with
            ``(center, reg_min, reg_max, stride)``.
        level_ids: ``[num_points]``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        beta: float = 0.05,
        num_levels: int = 4,
        locality_strength: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_basis <= 0 or prompt_length <= 0:
            raise ValueError("num_basis and prompt_length must be positive")
        if router_hidden_dim <= 0 or frf_hidden_dim <= 0:
            raise ValueError("router_hidden_dim and frf_hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if locality_strength < 0:
            raise ValueError("locality_strength must be non-negative")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.num_levels = int(num_levels)
        self.temperature = float(temperature)
        self.locality_strength = float(locality_strength)

        self.register_buffer("beta", torch.tensor(float(beta)))
        self._beta_is_zero = float(beta) == 0.0

        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_value_projection = nn.Linear(hidden_dim, hidden_dim)

        # Center/stride metadata and pyramid level make the dense candidates
        # distinguishable even when their visual contents are similar.
        self.point_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.level_embedding = nn.Embedding(num_levels, hidden_dim)

        self.router = nn.Sequential(
            nn.Linear(3 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )

        self.frf_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.update_gate = nn.Linear(hidden_dim, 1)
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_norm = nn.LayerNorm(hidden_dim)

        # Transient diagnostics: never saved in checkpoints.
        self.last_output: Optional[FlashPointDQCGPOutput] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        # Start close to the warm-started FlashVTG identity path.  Localization
        # gradients and the candidate gate loss can then open only useful
        # points instead of perturbing every dense pyramid point immediately.
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, -2.0)

    def set_beta(self, beta: float) -> None:
        """Set residual strength, including the exact-identity ablation."""

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
    def _masked_temporal_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
        valid = valid_mask.bool()
        masked_logits = logits.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * valid.unsqueeze(1).to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(
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
        center_normalized = centers.unsqueeze(0) / denominator
        stride_normalized = strides.unsqueeze(0) / denominator
        coordinates = torch.stack((center_normalized, stride_normalized), dim=-1)
        point_feature = self.point_projection(coordinates)
        return point_feature + self.level_embedding(level_ids).to(dtype).unsqueeze(0)

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
        """Return shape-preserving, DQ-CGP-refined pyramid candidates."""

        # Preserve DQ-CGP's strong identity contract.  This branch deliberately
        # precedes validation and every projection/stochastic computation.
        if self._beta_is_zero:
            self.last_output = None
            return candidate_state

        self._check_inputs(
            candidate_state,
            video_memory,
            video_valid_mask,
            candidate_valid_mask,
            query_semantic,
            point_metadata,
            level_ids,
        )

        video_valid_mask = video_valid_mask.bool()
        candidate_valid_mask = candidate_valid_mask.bool()
        valid_lengths = video_valid_mask.sum(dim=1)
        point_feature = self._point_features(
            point_metadata, level_ids, valid_lengths, candidate_state.dtype
        )

        candidate_key = self.candidate_projection(
            self.candidate_norm(candidate_state)
        )
        semantic_key = self.semantic_projection(query_semantic).unsqueeze(1)
        temporal_query = candidate_key + semantic_key + point_feature
        temporal_key = self.memory_key_projection(self.memory_norm(video_memory))
        temporal_logits = torch.einsum(
            "bpd,btd->bpt", temporal_query, temporal_key
        ) / math.sqrt(self.hidden_dim)

        if self.locality_strength > 0:
            frame_positions = torch.arange(
                video_memory.shape[1],
                device=video_memory.device,
                dtype=temporal_logits.dtype,
            )
            centers = point_metadata[:, 0].to(temporal_logits.dtype)
            strides = point_metadata[:, 3].to(temporal_logits.dtype).clamp_min(1.0)
            relative_distance = (
                frame_positions.view(1, 1, -1) - centers.view(1, -1, 1)
            ).abs() / strides.view(1, -1, 1)
            temporal_logits = temporal_logits - self.locality_strength * relative_distance

        temporal_attention = self._masked_temporal_softmax(
            temporal_logits, video_valid_mask
        )
        video_value = self.memory_value_projection(video_memory)
        temporal_context = torch.einsum(
            "bpt,btd->bpd", temporal_attention, video_value
        )

        semantic = query_semantic.unsqueeze(1).expand(
            -1, candidate_state.shape[1], -1
        )
        router_input = torch.cat(
            (temporal_context, semantic, point_feature), dim=-1
        )
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(
            router_logits / self.temperature, dim=-1
        )
        prompt_sequence = torch.einsum(
            "bpn,nld->bpld", basis_weights, self.basis_prompts
        )
        # Retain prompt-token structure.  A plain mean makes prompt_length
        # mathematically equivalent to a single basis vector.
        prompt_query = candidate_key + semantic_key
        prompt_logits = torch.einsum(
            "bpd,bpld->bpl", prompt_query, prompt_sequence
        ) / math.sqrt(self.hidden_dim)
        prompt_attention = torch.softmax(prompt_logits, dim=-1)
        pooled_prompt = torch.einsum(
            "bpl,bpld->bpd", prompt_attention, prompt_sequence
        )

        projected_context = self.frf_context_projection(temporal_context)
        frf_feature = self.frf(
            torch.cat((pooled_prompt, semantic, projected_context), dim=-1)
        )
        update_gate = torch.sigmoid(self.update_gate(frf_feature)).squeeze(-1)
        update_gate = update_gate * candidate_valid_mask.to(update_gate.dtype)
        residual_update = self.residual_norm(
            self.residual_projection(frf_feature)
        )
        residual_update = residual_update * update_gate.unsqueeze(-1)
        adapted_state = candidate_state + self.beta.to(
            candidate_state.dtype
        ) * residual_update

        self.last_output = FlashPointDQCGPOutput(
            adapted_state=adapted_state,
            temporal_logits=temporal_logits,
            temporal_attention=temporal_attention,
            temporal_context=temporal_context,
            basis_weights=basis_weights,
            prompt_sequence=prompt_sequence,
            prompt_attention=prompt_attention,
            pooled_prompt=pooled_prompt,
            frf_feature=frf_feature,
            update_gate=update_gate,
            residual_update=residual_update,
        )
        return adapted_state


__all__ = ["FlashPointDQCGP", "FlashPointDQCGPOutput"]
