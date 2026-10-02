# Compatibility Analysis: Soccer-GMR vs. Charades-STA Semantic Novelty × Existence

This document details the interface and protocol differences between the original Soccer-GMR codebase (where DQ-CGP was originally developed) and the Charades-STA Semantic Novelty × Event Existence benchmark (GMR_Unseen), together with the exact adaptations implemented to guarantee fair, matched-protocol evaluation.

---

## 1. Protocol Comparison Matrix

| Protocol Parameter | Soccer-GMR (DQ-CGP Original) | Charades-STA Baseline (GMR_Unseen) | Matched Protocol for Transfer (This Work) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Domain** | Soccer broadcast | Daily indoor activities (Charades-STA) | Daily indoor activities (Charades-STA) | Core benchmark target |
| **Novelty Axis** | Event types across match splits | 5 frozen holdouts (3 action, 2 composition) | 5 frozen holdouts (A1, A2_alt, A3, C1, C2_alt) | Answer the core scientific question |
| **Training Budget** | 400 epochs | 100 epochs | **100 epochs** | Matched budget; no extra training steps |
| **Early Stopping** | Enabled (varies 20–40) | Disabled (`max_es_cnt = -1`) | **Disabled (`max_es_cnt = -1`)** | Exact parity with GMR_Unseen baseline |
| **Random Seed** | 2023 | 3407 | **3407** | Matched seed for reproducibility |
| **Batch Size** | 16 (train), 16 (eval) | 16 (train), 16 (eval) | **16 (train), 16 (eval)** | Identical batch size |
| **Learning Rate** | `5e-5` / `1e-4` | Base Moment `1e-4` (lr_drop 400) | **`1e-4`** | Matched base optimizer & scheduler |
| **Saliency Supervision**| `mr_only=False`, `lw_saliency=1` | `mr_only=True`, `lw_saliency=0` | **`mr_only=True`, `lw_saliency=0`** | Must not introduce extra saliency supervision |
| **Empty GT (S−)** | Present in exist experiments | Present (1,500 training S−) | **Present (`keep_empty_gt=True`)** | Trains existence discrimination |
| **Existence Head** | Independent MLP on $h_q$ | Independent `GMRAdapter` on $h_q$ | **`GMRAdapter(d_model, d_model, pool='max')`** | Matched existence architecture |
| **Existence Loss Coef**| 1.0 | 1.0 | **1.0** | Matched existence loss weight |
| **Selection Metric** | MR mAP on val | MR-full-mAP on `val_seen` | **MR-full-mAP on `val_seen`** | Checkpoint selection strictly on seen val |
| **Threshold Tuning** | Fixed 0.3 / 0.4 | Calibrated on `val_seen` balanced acc | **Calibrated on `val_seen` balanced acc** | Evaluated via `analyze_semantic_existence.py` |

---

## 2. Text Feature & Semantic Mask Compatibility

### Background
- In Soccer-GMR, some scripts relied on fine-grained token masks or semantic attention masks.
- In Charades-STA (`GMR_Unseen`), CLIP text features are pre-extracted and saved as `last_hidden_state` NPZ files with shape `(L_txt, 512)` along with token mask `src_txt_mask`.
- Plan Section 6 states:
  > *Text feature 不要重新提取... 优先使用原 dataloader 已有的 `src_txt_mask` 作为有效 token mask。如果代码支持：`src_txt_semantic_mask = None`，则让方法退回使用：`src_txt_mask`。*

### Resolution
- Both **LS-DQCGP** and **DQ-CGPv3** check whether `src_txt_semantic_mask` is provided. If `src_txt_semantic_mask is None`:
  ```python
  semantic_mask = src_txt_mask.bool()
  ```
- The global query static semantic $E_{\text{static}}$ is obtained by masked average pooling over valid tokens:
  ```python
  semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
  query_semantic = (src_txt * semantic_mask.unsqueeze(-1).to(src_txt.dtype)).sum(dim=1) / semantic_count
  ```
- This ensures 100% feature compatibility with the baseline without re-extracting text features.

---

## 3. Empty-GT (S−) Handling in Mixed Batches

### Background
- In standard moment retrieval, all training queries contain ground-truth windows ($S^+$).
- In GMR, negative queries ($S^-$ and $U^-$) represent queries where the event is absent in the video. The ground truth spans for $S^-$ are empty (`len(relevant_windows) == 0`).
- Mixed batches contain both $S^+$ and $S^-$ samples.

### Hungarian Matcher Compatibility
- When `targets[i]["spans"]` has size $(0, 2)$, Hungarian matching correctly returns empty matching pairs:
  `src_indices = tensor([], dtype=int64)`, `target_indices = tensor([], dtype=int64)`.

### Binding Loss Safeguards
- **LS-DQCGP** (`native_matched_binding_loss`):
  ```python
  for batch_index, (src_indices, target_indices) in enumerate(indices):
      if src_indices.numel() == 0:
          continue
      ...
  return torch.cat(terms).mean() if terms else attention.sum() * 0.0
  ```
  If all samples in a batch or a given sample are $S^-$, `src_indices.numel() == 0` causes it to be skipped. If no matched positive spans exist, the loss returns a zero tensor with grad connected to `attention`, avoiding NaN or disconnected gradients.
- **DQ-CGPv3** (`loss_query_cgp`):
  ```python
  for batch_index, (src_indices, target_indices) in enumerate(indices):
      if src_indices.numel() == 0:
          continue
      ...
  if binding_terms:
      ...
  else:
      binding_loss = attention.sum() * 0.0
      route_loss = basis_weights.sum() * 0.0
  ```
  Similarly skips empty GT queries and safely outputs zero loss.

### Existence Loss Handling
- The existence loss is binary cross-entropy between `pred_exist_logits` and existence target (1 for $S^+$, 0 for $S^-$).
- This loss is calculated over all samples in the batch, guaranteeing that $S^-$ samples actively supervise the existence branch while not contributing spurious localization or binding loss.

---

## 4. Evaluation and Output Formatting

The evaluation script `scripts/analyze_semantic_existence.py` expects every prediction item in the test submission file to adhere to the following schema:
- `qid`: Query ID (string or integer)
- `pred_exist_score`: Float in $[0, 1]$ representing existence probability
- `pred_relevant_windows`: Ranked list of $[start, end, score]$ after applying the existence gate
- `pred_relevant_windows_pre_exist`: Ranked list of $[start, end, score]$ raw localization windows before applying the gate

Both model implementations conform strictly to this format.
