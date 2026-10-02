"""DETR Query CGP (v3) Model Wrapper and Loss Integration.

Replicated from upstream DQ-CGP repository:
- Source: /home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main/models/moment_detr_gmr/moment_detr.py
- Commit SHA: 1610bf00787e33f1b4de860ad4be18b331bdff8f
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from experiments.dq_cgp_semantic_generalization.dq_cgp_v3.query_cgp import DETRQueryCGP, DETRQueryCGPOutput


class DQCGPv3Model(nn.Module):
    """Wrapper that equips Moment-DETR with candidate-specific DETR-Query CGP (v3)."""

    def __init__(
        self,
        base_model: nn.Module,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_dim = base_model.transformer.d_model
        self.query_cgp = DETRQueryCGP(
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            prompt_length=prompt_length,
            router_hidden_dim=router_hidden_dim,
            frf_hidden_dim=frf_hidden_dim,
            temperature=temperature,
            beta=beta,
        )

    def forward(
        self,
        src_txt: Tensor,
        src_txt_mask: Tensor,
        src_vid: Tensor,
        src_vid_mask: Tensor,
        src_aud: Optional[Tensor] = None,
        src_aud_mask: Optional[Tensor] = None,
        src_txt_semantic_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        src_vid = self.base_model.input_vid_proj(src_vid)
        src_txt = self.base_model.input_txt_proj(src_txt)

        semantic_mask = (
            src_txt_mask.bool()
            if src_txt_semantic_mask is None
            else src_txt_semantic_mask.bool() & src_txt_mask.bool()
        )
        semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
        semantic_weights = semantic_mask.to(src_txt.dtype).unsqueeze(-1)
        query_semantic = (src_txt * semantic_weights).sum(dim=1) / semantic_count.to(src_txt.dtype)

        src = torch.cat([src_vid, src_txt], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos_vid = self.base_model.position_embed(src_vid, src_vid_mask)
        pos_txt = (
            self.base_model.txt_position_embed(src_txt)
            if self.base_model.use_txt_pos
            else torch.zeros_like(src_txt)
        )
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        self.query_cgp.clear_diagnostics()
        hs, memory = self.base_model.transformer(
            src,
            ~mask,
            self.base_model.query_embed.weight,
            pos,
            decoder_interlayer_adapter=self.query_cgp,
            decoder_adapter_after_layer=0,
            decoder_adapter_kwargs={
                "query_semantic": query_semantic,
                "video_length": src_vid.shape[1],
            },
        )

        outputs_class = self.base_model.class_embed(hs)
        outputs_coord = self.base_model.span_embed(hs)
        if self.base_model.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
        }

        if self.query_cgp.last_output is not None:
            cgp_out = self.query_cgp.last_output
            out["query_cgp_temporal_attention"] = cgp_out.temporal_attention
            out["query_cgp_basis_weights"] = cgp_out.basis_weights
            out["query_cgp_video_mask"] = src_vid_mask.bool()

        if self.base_model.exist_head is not None:
            out["pred_exist_logits"] = self.base_model.exist_head(hs[-1])

        vid_mem = memory[:, :src_vid.shape[1]]
        out["saliency_scores"] = self.base_model.saliency_proj(vid_mem).squeeze(-1)

        if self.base_model.aux_loss:
            aux_logits = self.base_model.class_embed(hs[:-1])
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(aux_logits, outputs_coord[:-1])
            ]

        return out


def compute_query_cgp_loss(
    outputs: Dict[str, Tensor],
    targets: Dict,
    indices: List,
    span_loss_type: str = "l1",
) -> Dict[str, Tensor]:
    """Compute candidate-specific temporal binding loss and route entropy loss."""
    required = {
        "query_cgp_temporal_attention",
        "query_cgp_basis_weights",
        "query_cgp_video_mask",
    }
    if not required.issubset(outputs) or targets is None or "span_labels" not in targets or indices is None:
        zero = outputs["pred_logits"].sum() * 0.0
        return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": zero}

    attention = outputs["query_cgp_temporal_attention"]
    basis_weights = outputs["query_cgp_basis_weights"]
    video_mask = outputs["query_cgp_video_mask"].bool()
    binding_terms = []
    matched_routes = []
    eps = torch.finfo(attention.dtype).eps

    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].sum().item())
        if valid_length <= 0:
            continue

        device = attention.device
        src_indices = src_indices.to(device)
        target_indices = target_indices.to(device)
        matched_attention = attention[batch_index, src_indices, :valid_length]
        target_spans = targets["span_labels"][batch_index]["spans"][target_indices].to(device)

        if span_loss_type == "l1":
            target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
            clip_starts = torch.arange(
                valid_length, device=device, dtype=attention.dtype
            ) / float(valid_length)
            clip_ends = clip_starts + 1.0 / float(valid_length)
            overlap = (
                (clip_starts.unsqueeze(0) < target_xx[:, 1:])
                & (clip_ends.unsqueeze(0) > target_xx[:, :1])
            )
            empty_overlap = ~overlap.any(dim=1)
            if bool(empty_overlap.any()):
                clip_centers = 0.5 * (clip_starts + clip_ends)
                nearest = (
                    clip_centers.unsqueeze(0) - target_xx[:, :1]
                ).abs().argmin(dim=1)
                overlap[empty_overlap] = False
                overlap[empty_overlap, nearest[empty_overlap]] = True
        else:
            clip_indices = torch.arange(valid_length, device=device).unsqueeze(0)
            overlap = (
                (clip_indices >= target_spans[:, :1])
                & (clip_indices <= target_spans[:, 1:])
            )

        target_mass = (matched_attention * overlap.to(attention.dtype)).sum(dim=1)
        binding_terms.append(-target_mass.clamp_min(eps).log())
        matched_routes.append(basis_weights[batch_index, src_indices])

    if binding_terms:
        binding_loss = torch.cat(binding_terms).mean()
        routes = torch.cat(matched_routes, dim=0)
        route_eps = torch.finfo(routes.dtype).eps
        conditional_entropy = -(
            routes * routes.clamp_min(route_eps).log()
        ).sum(dim=-1).mean()
        marginal = routes.mean(dim=0)
        marginal_entropy = -(
            marginal * marginal.clamp_min(route_eps).log()
        ).sum()
        route_loss = conditional_entropy - marginal_entropy
    else:
        binding_loss = attention.sum() * 0.0
        route_loss = basis_weights.sum() * 0.0

    return {
        "loss_query_cgp_bind": binding_loss,
        "loss_query_cgp_route": route_loss,
    }


def install_dq_cgp_v3_loss(
    criterion: nn.Module,
    binding_coef: float = 0.2,
    route_coef: float = 0.01,
) -> None:
    """Install DQ-CGPv3 binding and route loss on SetCriterion."""
    original_forward = criterion.forward

    def controlled_forward(this, outputs, targets):
        losses = original_forward(outputs, targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = this.matcher(outputs_without_aux, targets)
        cgp_losses = compute_query_cgp_loss(
            outputs, targets, indices, span_loss_type=getattr(this, "span_loss_type", "l1")
        )
        losses.update(cgp_losses)
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict["loss_query_cgp_bind"] = float(binding_coef)
    criterion.weight_dict["loss_query_cgp_route"] = float(route_coef)
    criterion._dq_cgp_v3_original_forward = original_forward
