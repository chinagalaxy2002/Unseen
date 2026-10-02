# Source Audit: DQ-CGP Methods Migration to Semantic Novelty × Existence Benchmark

This document records the provenance, commit hashes, and replicated files for migrating **LS-DQCGP** and **DQ-CGPv3** into the **Moment-DETR-GMR** semantic generalization benchmark.

---

## 1. Upstream Repositories and Exact Commit Hashes

| Project | Upstream Remote URL | Local Workspace / Source Path | Commit SHA / State |
| :--- | :--- | :--- | :--- |
| **GMR_Unseen** (Base Benchmark) | `https://github.com/chinagalaxy2002/GMR_Unseen.git` | `/home/guoxiangyu/paper/Openword/generalized-moment-retrieval` | `bc88348e0e6b7b629a77d884fe66efe8c87b2774` |
| **DQ-CGP Published** | `https://github.com/chinagalaxy2002/DQ-CGP.git` | `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-github-publish` | `1610bf00787e33f1b4de860ad4be18b331bdff8f` |
| **DQ-CGP Main** | `https://github.com/chinagalaxy2002/DQ-CGP` | `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main` | Extracted from `DQ-CGP-main.zip` |
| **Unseen2 (Active Experiment)** | N/A (Isolated experiment workspace) | `/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen2` | Current local execution workspace |

> [!IMPORTANT]
> **Pretrained Checkpoint Audit**:
> The checkpoints present in `DQ-CGP-main` and `DQ-CGP-github-publish` (e.g., `outputs/ls_dq_cgp_exist_seed2023`, `outputs/dq_cgp_v3_seed2023`) were trained exclusively on **Soccer-GMR**. They are **not** applicable to Charades-STA or the semantic novelty × event existence splits. In accordance with the experiment plan, all models will be trained from scratch on each split using seed `3407` for 100 epochs under the matched GMR_Unseen training protocol.

---

## 2. Method 1: LS-DQCGP (Late-Semantic DQ-CGP + Existence Head)

### Source File Provenance
- Source directory: `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-github-publish/ls_dq_cgp_lab`
- Copied/adapted to: `experiments/dq_cgp_semantic_generalization/ls_dqcgp/`

| Replicated Component | Source File | Destination File | Description |
| :--- | :--- | :--- | :--- |
| **LateSemanticCGP** | `cgp_module.py` | `ls_dqcgp/cgp_module.py` | RCG, BPS, FRF, and Late Semantic Matcher |
| **LSDQCGPModel** | `ls_dq_cgp_model.py` | `ls_dqcgp/model.py` | Model wrapper with `NativeD1AttentionCapture`, `native_matched_binding_loss`, and criterion wrapper |
| **Training Routine** | `train_ls_dq_cgp.py` | `ls_dqcgp/train.py` | Training script conforming to GMR_Unseen matched 100-epoch protocol |

### Method Invariants Preserved
- Native D1 temporal cross-attention capture from decoder layer 0
- Candidate-specific local visual context extraction: $V_q = \text{bmm}(A_{D1}, M_{\text{vid}})$
- Relational Context Gating (RCG): MLP over $[V_q; E_{\text{static}}]$
- Basis Prompt Synthesis (BPS): 16 learnable prompt bases of length 6
- Feature Refinement & Fusion (FRF): Fusion of pooled prompt, static semantic, and visual context
- Late semantic cosine matcher with learnable logit scale and bias
- Independent GMR existence head on $h_q$ ($hs[-1]$)
- Hungarian-matched D1 binding loss ($\text{coef} = 0.2$)

---

## 3. Method 2: DQ-CGPv3 (DETR Query CGP v3)

### Source File Provenance
- Source directory: `/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main`
- Copied/adapted to: `experiments/dq_cgp_semantic_generalization/dq_cgp_v3/`

| Replicated Component | Source File | Destination File | Description |
| :--- | :--- | :--- | :--- |
| **DETRQueryCGP** | `experiments/vmr_cgp/query_cgp.py` | `dq_cgp_v3/query_cgp.py` | Interlayer candidate query adapter module |
| **DQCGPv3 Model** | `models/moment_detr_gmr/moment_detr.py` | `dq_cgp_v3/model.py` | MomentDETR integration with interlayer query adapter |
| **Decoder Interlayer Hook**| `models/moment_detr_gmr/moment_transformer.py` | `models/moment_detr_gmr/moment_transformer.py` | TransformerDecoder forward hook after layer 0 |
| **Training Routine** | `scripts/train_dq_cgp_v3.sh` & `train.py` | `dq_cgp_v3/train.py` | Training script conforming to GMR_Unseen matched 100-epoch protocol |

### Method Invariants Preserved
- Candidate-specific DETR query adaptation between decoder layers 0 and 1
- Candidate-specific temporal cross-attention: $A_{\text{temp}} = \text{softmax}((Q_{\text{cand}} + Q_{\text{sem}}) K_{\text{vid}}^T / \sqrt{d})$
- Basis routing over 16 prompt bases of length 6
- FRF feature refinement and fixed-beta residual injection ($\beta = 0.05$)
- Interlayer binding loss ($\text{coef} = 0.2$) and route entropy loss ($\text{coef} = 0.01$)
- Independent GMR existence head on $hs[-1]$
