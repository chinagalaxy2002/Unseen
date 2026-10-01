"""Compatibility adapter for the GMR hooks used by FlashVTG inference.

The released inference code imports an ``eval_v1_3`` module that is not part
of this repository.  The repository does contain the current GMR ground-truth
normalizer, so this adapter reuses it and implements the small binary
existence-metric interface expected by the training loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from eval.normalization import load_ts_window_cfg, normalize_ground_truth


def _load_ts_window_cfg(path: Optional[str]):
    """Backward-compatible name expected by FlashVTG-GMR inference."""

    return load_ts_window_cfg(path)


def _existence_score(prediction: Dict[str, Any], pred_topk: int) -> float:
    """Prefer the explicit existence head and fall back to window confidence."""

    if "pred_exist_score" in prediction:
        try:
            return float(prediction["pred_exist_score"])
        except (TypeError, ValueError):
            pass

    scores = []
    for window in (prediction.get("pred_relevant_windows") or [])[:pred_topk]:
        if isinstance(window, (list, tuple)) and len(window) >= 3:
            try:
                scores.append(float(window[2]))
            except (TypeError, ValueError):
                continue
    return max(scores, default=0.0)


def compute_gmr_cls_metrics(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    *,
    pred_topk: int = 10,
    pred_score_thd: float = 0.5,
) -> Dict[str, float]:
    """Compute positive recall, negative recall, and balanced accuracy.

    Values are percentages, matching the other metrics stored in the
    FlashVTG evaluation ``brief`` dictionary.
    """

    qid_to_prediction = {
        item["qid"]: item
        for item in submission
        if isinstance(item, dict) and "qid" in item
    }
    tp = tn = fp = fn = 0
    for target in ground_truth:
        is_positive = bool(target.get("relevant_windows"))
        score = _existence_score(qid_to_prediction.get(target["qid"], {}), pred_topk)
        predicts_positive = score >= pred_score_thd
        if is_positive and predicts_positive:
            tp += 1
        elif is_positive:
            fn += 1
        elif predicts_positive:
            fp += 1
        else:
            tn += 1

    tpr = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    tnr = 100.0 * tn / (tn + fp) if tn + fp else 0.0
    return {
        "TPR": round(tpr, 2),
        "TNR": round(tnr, 2),
        "BalancedAcc": round((tpr + tnr) / 2.0, 2),
    }
