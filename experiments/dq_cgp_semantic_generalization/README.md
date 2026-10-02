# DQ-CGP Semantic Generalization on Moment-DETR-GMR

This directory contains the complete codebase, configurations, experiments, and evaluation results for evaluating **DQ-CGP** transfer under the **Semantic Novelty × Event Existence Benchmark** on Charades-STA.

## 1. Research Question

> **In the exact semantic novelty × event existence protocol on Charades-STA, does equipping Moment-DETR-GMR with DQ-CGP (LS-DQCGP or DQ-CGPv3) mitigate or reverse the observed seen-to-unseen degradation?**

## 2. Evaluated Methods

1. **LS-DQCGP** (Late-Semantic DQ-CGP + Existence Head):
   - Hook native D1 cross-attention from decoder layer 0.
   - Per-candidate local visual context extraction: $V_q = \text{bmm}(A_{D1}, M_{\text{vid}})$.
   - RCG, BPS (16 basis prompts), and FRF semantic modulation.
   - Late cosine semantic matcher with learnable logit scale and bias.
   - Matched protocol: `use_exist_head=True`, `mr_only=True`, `lw_saliency=0`, `native_bind_coef=0.2`.

2. **DQ-CGPv3** (DETR-Query CGP v3):
   - Candidate-specific DETR query adaptation between decoder layers 0 and 1.
   - Candidate-specific temporal cross-attention and basis routing (16 basis prompts).
   - FRF feature refinement and fixed-beta residual injection ($\beta = 0.05$).
   - Matched protocol: `use_exist_head=True`, `mr_only=True`, `lw_saliency=0`, `query_cgp_binding_loss_coef=0.2`, `query_cgp_route_loss_coef=0.01`.

## 3. Directory Structure

- `SOURCE_AUDIT.md`: Upstream repository URLs, commit SHAs, file provenance, and checkpoint notices.
- `COMPATIBILITY.md`: Detailed audit of protocol differences and interface compatibility.
- `EXPERIMENT_PLAN.md`: Full 10-run experiment matrix, protocol specifications, and evaluation workflow.
- `ls_dqcgp/`: LS-DQCGP module, model wrapper, and training script.
- `dq_cgp_v3/`: DQ-CGPv3 module, model wrapper, and training script.
- `scripts/`: Test evaluation, diagnostics automation, and parallel multi-split runners.
- `runs/`: Checkpoints, predictions, and logs for all 10 experimental runs.
- `reports/`: Per-split comparative diagnostics and equal-weighted aggregate summaries.
