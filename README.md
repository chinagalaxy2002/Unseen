# Unseen2: DQ-CGP Transfer on Semantic Novelty × Event Existence Benchmark

This branch (`Unseen2`) implements the evaluation of **DQ-CGP** (incorporating both **LS-DQCGP** and **DQ-CGPv3**) transfer on the **Charades-STA Semantic Novelty × Event Existence Benchmark** based on Moment-DETR-GMR.

## 1. Core Research Question

> **In the exact semantic novelty × event existence protocol on Charades-STA, does equipping Moment-DETR-GMR with DQ-CGP (LS-DQCGP or DQ-CGPv3) mitigate or reverse the observed seen-to-unseen degradation?**

## 2. Implemented Methods

1. **LS-DQCGP** (`experiments/dq_cgp_semantic_generalization/ls_dqcgp`):
   - Hook native D1 cross-attention from decoder layer 0.
   - Candidate-specific local visual context extraction: $V_q = \text{bmm}(A_{D1}, M_{\text{vid}})$.
   - RCG (Relational Context Gating), BPS (16 basis prompts), and FRF (Feature Refinement & Fusion).
   - Late cosine semantic matcher with learnable logit scale and bias.
   - Hungarian-matched D1 binding loss ($\text{coef} = 0.2$).
   - Matched protocol: `use_exist_head=True`, `mr_only=True`, `lw_saliency=0`.

2. **DQ-CGPv3** (`experiments/dq_cgp_semantic_generalization/dq_cgp_v3`):
   - Candidate-specific DETR query adaptation between decoder layers 0 and 1.
   - Candidate-specific temporal cross-attention and basis routing over 16 prompt bases.
   - FRF feature refinement and fixed-beta residual injection ($\beta = 0.05$).
   - Interlayer binding loss ($\text{coef} = 0.2$) and route entropy loss ($\text{coef} = 0.01$).
   - Matched protocol: `use_exist_head=True`, `mr_only=True`, `lw_saliency=0`.

## 3. Experiment Protocol & Setup

- **Five Frozen Splits**:
  - Action splits: `A1` (put/take), `A2_alt` (drink/pour), `A3` (run/walk)
  - Composition splits: `C1` (sit | bed/chair/couch), `C2_alt` (open/close | box/cabinet)
- **Protocol Parity**:
  - Seed: `3407`
  - Training: 100 epochs, no early stopping (`max_es_cnt = -1`)
  - Checkpoint selection: Strictly based on seen validation MR-full-mAP (`val_seen.jsonl`)
  - Existence threshold: Calibrated on `val_seen.jsonl` balanced accuracy
  - Evaluation: Full four-quadrant ($S^+, S^-, U^+, U^-$) and matched-pair testing via `scripts/analyze_semantic_existence.py`

## 4. Documentation & Analysis

- Detailed upstream source audit: [SOURCE_AUDIT.md](experiments/dq_cgp_semantic_generalization/SOURCE_AUDIT.md)
- Protocol compatibility analysis: [COMPATIBILITY.md](experiments/dq_cgp_semantic_generalization/COMPATIBILITY.md)
- Formal experiment plan: [EXPERIMENT_PLAN.md](experiments/dq_cgp_semantic_generalization/EXPERIMENT_PLAN.md)
- Experiment tracking & results report: [FINAL_EVALUATION_REPORT.md](experiments/dq_cgp_semantic_generalization/reports/FINAL_EVALUATION_REPORT.md)
