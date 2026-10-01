# DQ-CGP on Semantic Novelty × Event Existence (Unseen)

This repository evaluates whether applying **DQ-CGP** candidate enhancement alleviates the seen-to-unseen generalization degradation of **FlashVTG-GMR** on the Charades-STA derived semantic novelty × event existence benchmark.

---

## 1. Research Question

In generalized moment retrieval, models often display severe performance drops when presented with semantically novel queries (unseen actions/compositions), failing to distinguish whether an unseen event actually exists in the video:

$$\text{Baseline Gap} = \text{Seen AUROC} - \text{Unseen AUROC}$$

On the frozen Charades-STA benchmark (split **A1: put/take holdout**), baseline FlashVTG-GMR exhibits:
- **Seen AUROC**: 0.8158
- **Unseen AUROC**: 0.5065
- **Gap**: 0.3093

**Core Hypothesis**: Does integrating DQ-CGP candidate representation refinement mitigate this degradation and achieve positive gap recovery ($\Delta \text{Gap} > 0$) without collapsing seen-split discrimination?

---

## 2. Experimental Protocol

- **Partitions**:
  - $S^+$: Seen semantics, event present
  - $S^-$: Seen semantics, event absent
  - $U^+$: Unseen semantics, event present
  - $U^-$: Unseen semantics, event absent
- **Training**: Strictly $S^+ / S^-$ only.
- **Model Selection / Validation**: Strictly seen validation ($S^+ / S^-$) using $(R1@0.7 + R1@0.5)/2$. Unseen splits ($U^+, U^-$) are never observed during training, hyperparameter tuning, or checkpoint selection.
- **Evaluation**: 4-quadrant diagnostic analysis (AUROC, FRR, RR, raw/gated R1@0.5, matched-pair accuracy) and official GMR evaluation.

---

## 3. Repository Structure

```text
.
├── configs/            # Configuration files
├── eval/               # Official GMR evaluation protocol (eval_main.py)
├── models/             # Model architectures
│   ├── flash_vtg_gmr/            # Baseline FlashVTG-GMR backbone
│   ├── flashvtg_dq_cgp_v3_gmr/   # DQ-CGP v3 (sparse top-4 routing + relation loss)
│   └── flashvtg_dq-cgp-gmr-v2/   # DQ-CGP v2
├── scripts/            # Automation & diagnostic scripts
│   ├── sanity_check.py                    # 10-step protocol verification
│   ├── train_dq_cgp_semantic_existence.sh # Matched 100-epoch training
│   ├── infer_dq_cgp_semantic_existence.sh # Full test inference & 4-quadrant analysis
│   ├── analyze_semantic_existence.py      # Diagnostic metric computation
│   └── validate_release.py                # Release dataset checksum validation
├── training/           # FlashVTG training, inference, and dataset loaders
├── plan.md             # Detailed benchmark specification and protocol rules
└── README.md
```

---

## 4. Quick Start

### Environment Setup
Requires PyTorch 2.0+ with CUDA support and `nncore`:
```bash
conda activate univtg
```

### Pre-Training Sanity Check
Verify dataset completeness, feature coverage, loss finiteness, and candidate identity degradation:
```bash
python scripts/sanity_check.py
```

### Training
Train FlashVTG DQ-CGP from scratch for 100 epochs on split `A1` (or `A2_alt`, `A3`, `C1`, `C2_alt`):
```bash
bash scripts/train_dq_cgp_semantic_existence.sh A1
```

### Inference & Evaluation
Run full inference with the best validation checkpoint, compute 4-quadrant diagnostics, and run official evaluation:
```bash
bash scripts/infer_dq_cgp_semantic_existence.sh A1
```
