# Comparative Report: Moment-DETR-GMR vs. LS-DQCGP vs. DQ-CGPv3

## 1. Per-Split Diagnostics Table

| Split | Method | Seen AUROC | Unseen AUROC | Δ Unseen | Gap (Seen - Unseen) | Gap Reduction | PairAcc | U+ raw R1@0.5 | U+ gated R1@0.5 | U+ FRR | U− RR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Moment Baseline | 0.8044 | 0.4973 | - | 0.3071 | - | 0.5593 | 23.01% | 17.85% | 23.23% | 25.47% |
| **A1** | LS-DQCGP | *Pending* | - | - | - | - | - | - | - | - | - |
| **A1** | DQ-CGPv3 | *Pending* | - | - | - | - | - | - | - | - | - |
| **A2_alt** | Moment Baseline | 0.7690 | 0.5511 | - | 0.2180 | - | 0.6203 | 26.79% | 26.79% | 0.00% | 0.64% |
| **A2_alt** | LS-DQCGP | *Pending* | - | - | - | - | - | - | - | - | - |
| **A2_alt** | DQ-CGPv3 | *Pending* | - | - | - | - | - | - | - | - | - |
| **A3** | Moment Baseline | 0.7488 | 0.5643 | - | 0.1845 | - | 0.4690 | 39.58% | 28.12% | 29.17% | 33.00% |
| **A3** | LS-DQCGP | *Pending* | - | - | - | - | - | - | - | - | - |
| **A3** | DQ-CGPv3 | *Pending* | - | - | - | - | - | - | - | - | - |
| **C1** | Moment Baseline | 0.7610 | 0.5621 | - | 0.1989 | - | 0.6285 | 48.77% | 47.53% | 3.70% | 7.78% |
| **C1** | LS-DQCGP | *Pending* | - | - | - | - | - | - | - | - | - |
| **C1** | DQ-CGPv3 | *Pending* | - | - | - | - | - | - | - | - | - |
| **C2_alt** | Moment Baseline | 0.6759 | 0.4687 | - | 0.2073 | - | 0.7273 | 35.65% | 0.00% | 98.26% | 96.85% |
| **C2_alt** | LS-DQCGP | *Pending* | - | - | - | - | - | - | - | - | - |
| **C2_alt** | DQ-CGPv3 | *Pending* | - | - | - | - | - | - | - | - | - |

## 2. Axis-Level Equal-Weighted Means

| Axis | Method | Mean Seen AUROC | Mean Unseen AUROC | Δ Unseen | Mean Gap | Mean Gap Reduction | Mean PairAcc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Action (A1, A2_alt, A3)** | Moment Baseline | 0.7741 | 0.5376 | - | 0.2365 | - | 0.5495 |
| Action (A1, A2_alt, A3) | LS-DQCGP | *Incomplete (0/3)* | - | - | - | - | - |
| Action (A1, A2_alt, A3) | DQ-CGPv3 | *Incomplete (0/3)* | - | - | - | - | - |
| **Composition (C1, C2_alt)** | Moment Baseline | 0.7185 | 0.5154 | - | 0.2031 | - | 0.6779 |
| Composition (C1, C2_alt) | LS-DQCGP | *Incomplete (0/2)* | - | - | - | - | - |
| Composition (C1, C2_alt) | DQ-CGPv3 | *Incomplete (0/2)* | - | - | - | - | - |
| **Overall (5 splits)** | Moment Baseline | 0.7518 | 0.5287 | - | 0.2231 | - | 0.6009 |
| Overall (5 splits) | LS-DQCGP | *Incomplete (0/5)* | - | - | - | - | - |
| Overall (5 splits) | DQ-CGPv3 | *Incomplete (0/5)* | - | - | - | - | - |

## 3. Scientific Question Evaluation

> **Question: Does DQ-CGP reduce the seen-to-unseen degradation of Moment-DETR-GMR under the semantic novelty × event existence protocol?**

