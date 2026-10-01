"""Four-quadrant and matched-pair diagnostics for one GMR prediction file."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def iou(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0


def top1_hit(row, prediction, raw=False):
    key = "pred_relevant_windows_pre_exist" if raw else "pred_relevant_windows"
    windows = prediction.get(key, prediction.get("pred_relevant_windows", []))
    if not windows:
        return False
    best = max(windows, key=lambda w: w[2])
    return max(iou(best, gt) for gt in row["relevant_windows"]) >= 0.5


def choose_threshold(rows, predictions):
    items = [(r, predictions[r["qid"]]) for r in rows if r["qid"] in predictions]
    positive = np.array([bool(r["relevant_windows"]) for r, _ in items])
    score = np.array([float(p["pred_exist_score"]) for _, p in items])
    thresholds = np.unique(score)
    best = (-1, 0.5)
    for threshold in thresholds:
        predicted = score >= threshold
        balanced = .5 * (predicted[positive].mean() + (~predicted[~positive]).mean())
        if balanced > best[0]:
            best = (balanced, float(threshold))
    return best[1]


def analyze(args):
    release = Path(args.release)
    val = read_jsonl(release / "val.jsonl")
    test = read_jsonl(release / "test.jsonl")
    pairs = read_jsonl(release / "matched_u_pairs.jsonl")
    val_pred = {str(p["qid"]): p for p in read_jsonl(args.val_predictions)}
    test_pred = {str(p["qid"]): p for p in read_jsonl(args.test_predictions)}
    val_seen = [r for r in val if r["partition"] in ("S+", "S-")]
    missing_val = [r["qid"] for r in val_seen if str(r["qid"]) not in val_pred]
    missing_test = [r["qid"] for r in test if str(r["qid"]) not in test_pred]
    if missing_val or missing_test:
        raise ValueError(f"Missing predictions: val={len(missing_val)}, test={len(missing_test)}")
    threshold = choose_threshold(val_seen, val_pred)
    by_part = {}
    for part in ("S+", "S-", "U+", "U-"):
        subset = [r for r in test if r["partition"] == part]
        scores = np.array([test_pred[str(r["qid"])]["pred_exist_score"] for r in subset], dtype=float)
        by_part[part] = {"n": len(subset), "mean_exist_score": float(scores.mean())}
        if part.endswith("+"):
            raw = np.array([top1_hit(r, test_pred[str(r["qid"])], raw=True) for r in subset])
            accepted = scores >= threshold
            by_part[part].update({
                "false_refusal": float((~accepted).mean()),
                "raw_R1_iou05": float(raw.mean()),
                "gated_R1_iou05": float((raw & accepted).mean()),
            })
        else:
            by_part[part]["rejection_rate"] = float((scores < threshold).mean())
    auc = {}
    for prefix, pos, neg in (("seen", "S+", "S-"), ("unseen", "U+", "U-")):
        subset = [r for r in test if r["partition"] in (pos, neg)]
        labels = [r["partition"] == pos for r in subset]
        scores = [test_pred[str(r["qid"])]["pred_exist_score"] for r in subset]
        auc[prefix] = float(roc_auc_score(labels, scores))
    pair_scores = []
    for pair in pairs:
        positive = test_pred[str(pair["positive_qid"])]["pred_exist_score"]
        negative = test_pred[str(pair["negative_qid"])]["pred_exist_score"]
        pair_scores.append((positive > negative) + .5 * (positive == negative))
    result = {
        "threshold_source": "val_seen balanced accuracy",
        "threshold": threshold,
        "quadrants": by_part,
        "AUROC": auc,
        "AUROC_gap": auc["seen"] - auc["unseen"],
        "over_refusal_gap": by_part["U+"]["false_refusal"] - by_part["S+"]["false_refusal"],
        "matched_pair_accuracy": float(np.mean(pair_scores)),
        "matched_pair_n": len(pair_scores),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--output", required=True)
    analyze(parser.parse_args())
