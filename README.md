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


---

## 5. Recorded Experiments (2026-10-01)

Two DQ-CGP v3 runs are available: **A1** (put/take holdout) and **C1** (sit with bed/chair/couch composition holdout). Both use seed **3407**, a configured **100 epochs**, and batch size **8**. Exact arguments are recorded in each run's `opt.json`; training and inference commands are implemented in [the training script](scripts/train_dq_cgp_semantic_existence.sh) and [the inference script](scripts/infer_dq_cgp_semantic_existence.sh).

### Semantic existence diagnostics

AUROC and its gap are on a 0–1 scale; matched-pair accuracy is a percentage. The refusal threshold is **0.995** for both runs, selected using seen validation balanced accuracy.

| Split | Seen AUROC | Unseen AUROC | Seen − unseen gap | Matched-pair accuracy (%) | Pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 0.8151 | 0.4946 | 0.3205 | 56.25 | 312 |
| C1 | 0.7443 | 0.6206 | 0.1237 | 65.28 | 144 |

On A1, the observed gap (0.3205) exceeds the baseline gap reported above (0.3093), giving an approximate gap recovery of **−0.0112**. This run does not support the proposed positive gap recovery on A1. C1 is a different partition and must be compared with its own matched baseline; these single-seed results do not establish statistical significance.

### Four-quadrant diagnostics

FRR is false refusal on present events; RR is rejection on absent events. R1@0.5 and rates are percentages. Gated retrieval uses the diagnostic threshold of 0.995.

| Split | Quadrant | Samples | FRR (%) | RR (%) | Raw R1@0.5 (%) | Gated R1@0.5 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | S+ | 2218 | 18.94 | — | 59.65 | 49.01 |
| A1 | S- | 1368 | — | 69.08 | — | — |
| A1 | U+ | 465 | 17.42 | — | 25.38 | 20.22 |
| A1 | U- | 1119 | — | 17.07 | — | — |
| C1 | S+ | 2799 | 44.27 | — | 54.16 | 29.72 |
| C1 | S- | 1474 | — | 76.32 | — | — |
| C1 | U+ | 162 | 8.02 | — | 51.23 | 46.91 |
| C1 | U- | 270 | — | 21.11 | — | — |

### Official GMR evaluation

The following values use the official evaluator's 0–100 scale. Rej-F1, accuracy, and G-mIoU@1 use the official threshold **0.4**, which differs from the diagnostic threshold.

| Split | Overall AUROC | Rej-F1@0.4 | Acc@0.4 | G-mIoU@1 | mAP | mR@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 66.83 | 42.59 | 62.51 | 36.72 | 38.34 | 29.80 |
| C1 | 69.57 | 38.52 | 69.33 | 37.49 | 38.26 | 29.35 |

### Published artifacts

- [A1 experiment logs and metrics](results/dq_cgp_semantic_existence/A1/)
- [C1 experiment logs and metrics](results/dq_cgp_semantic_existence/C1/)
- Each experiment contains training/inference console logs, per-epoch training and validation logs, run metadata, `opt.json`, best/latest validation metrics, `diagnostics.json`, and `official_test_metrics.json`.
- Checkpoints, TensorBoard event files, code archives, and per-query prediction dumps are kept locally and excluded from Git. Running inference again requires the local best checkpoint, datasets, and features.

To repeat the two runs with the required data and features installed:

```bash
bash scripts/train_dq_cgp_semantic_existence.sh A1
bash scripts/infer_dq_cgp_semantic_existence.sh A1
bash scripts/train_dq_cgp_semantic_existence.sh C1
bash scripts/infer_dq_cgp_semantic_existence.sh C1
```
