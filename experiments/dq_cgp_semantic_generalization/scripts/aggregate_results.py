"""Aggregate results across all 5 splits for Baseline, LS-DQCGP, and DQ-CGPv3.

Generates:
1. Split-by-split comparative table
2. Action-axis equal-weight mean
3. Composition-axis equal-weight mean
4. Overall mean
5. Scientific hypothesis evaluation (degradation mitigation analysis)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = Path("/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2")
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "dq_cgp_semantic_generalization"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
REPORTS_ROOT = EXPERIMENT_ROOT / "reports"

SPLITS = ["A1", "A2_alt", "A3", "C1", "C2_alt"]
ACTION_SPLITS = ["A1", "A2_alt", "A3"]
COMPOSITION_SPLITS = ["C1", "C2_alt"]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(diag: dict) -> dict:
    seen_auroc = diag["AUROC"]["seen"]
    unseen_auroc = diag["AUROC"]["unseen"]
    gap = seen_auroc - unseen_auroc
    pair_acc = diag.get("matched_pair_accuracy", 0.0)

    q = diag["quadrants"]
    s_pos_frr = q["S+"]["false_refusal"] * 100.0
    s_pos_raw_r1 = q["S+"].get("raw_R1_iou05", 0.0) * 100.0
    s_pos_gated_r1 = q["S+"].get("gated_R1_iou05", 0.0) * 100.0
    s_neg_rr = q["S-"]["rejection_rate"] * 100.0

    u_pos_frr = q["U+"]["false_refusal"] * 100.0
    u_pos_raw_r1 = q["U+"].get("raw_R1_iou05", 0.0) * 100.0
    u_pos_gated_r1 = q["U+"].get("gated_R1_iou05", 0.0) * 100.0
    u_neg_rr = q["U-"]["rejection_rate"] * 100.0

    return {
        "seen_auroc": seen_auroc,
        "unseen_auroc": unseen_auroc,
        "gap": gap,
        "pair_acc": pair_acc,
        "s_pos_frr": s_pos_frr,
        "s_pos_raw_r1": s_pos_raw_r1,
        "s_pos_gated_r1": s_pos_gated_r1,
        "s_neg_rr": s_neg_rr,
        "u_pos_frr": u_pos_frr,
        "u_pos_raw_r1": u_pos_raw_r1,
        "u_pos_gated_r1": u_pos_gated_r1,
        "u_neg_rr": u_neg_rr,
    }


def compute_group_means(data_list: List[dict]) -> dict:
    if not data_list:
        return {}
    keys = data_list[0].keys()
    means = {}
    for k in keys:
        vals = [d[k] for d in data_list if k in d and d[k] is not None]
        means[k] = sum(vals) / len(vals) if vals else 0.0
    return means


def generate_summary():
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_data = {}

    for split in SPLITS:
        summary_data[split] = {}
        # 1. Baseline
        base_diag = load_json(BASELINE_ROOT / split / "moment" / "diagnostics.json")
        if base_diag:
            summary_data[split]["baseline"] = extract_metrics(base_diag)

        # 2. LS-DQCGP
        ls_diag = load_json(RUNS_ROOT / split / "ls_dqcgp" / "diagnostics.json")
        if ls_diag:
            summary_data[split]["ls_dqcgp"] = extract_metrics(ls_diag)

        # 3. DQ-CGPv3
        v3_diag = load_json(RUNS_ROOT / split / "dq_cgp_v3" / "diagnostics.json")
        if v3_diag:
            summary_data[split]["dq_cgp_v3"] = extract_metrics(v3_diag)

    # Save raw json summary
    (REPORTS_ROOT / "multisplit_summary.json").write_text(
        json.dumps(summary_data, indent=2) + "\n"
    )

    # Build Markdown report
    lines = []
    lines.append("# Comparative Report: Moment-DETR-GMR vs. LS-DQCGP vs. DQ-CGPv3\n")
    lines.append("## 1. Per-Split Diagnostics Table\n")
    lines.append(
        "| Split | Method | Seen AUROC | Unseen AUROC | Δ Unseen | Gap (Seen - Unseen) | Gap Reduction | PairAcc | U+ raw R1@0.5 | U+ gated R1@0.5 | U+ FRR | U− RR |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    methods = [("baseline", "Moment Baseline"), ("ls_dqcgp", "LS-DQCGP"), ("dq_cgp_v3", "DQ-CGPv3")]

    for split in SPLITS:
        split_res = summary_data[split]
        base_m = split_res.get("baseline")

        for m_key, m_name in methods:
            m_res = split_res.get(m_key)
            if not m_res:
                lines.append(f"| **{split}** | {m_name} | *Pending* | - | - | - | - | - | - | - | - | - |")
                continue

            delta_unseen = (
                f"{m_res['unseen_auroc'] - base_m['unseen_auroc']:+.4f}"
                if base_m and m_key != "baseline"
                else "-"
            )
            gap_red = (
                f"{base_m['gap'] - m_res['gap']:+.4f}"
                if base_m and m_key != "baseline"
                else "-"
            )

            lines.append(
                f"| **{split}** | {m_name} | {m_res['seen_auroc']:.4f} | {m_res['unseen_auroc']:.4f} | "
                f"{delta_unseen} | {m_res['gap']:.4f} | {gap_red} | {m_res['pair_acc']:.4f} | "
                f"{m_res['u_pos_raw_r1']:.2f}% | {m_res['u_pos_gated_r1']:.2f}% | "
                f"{m_res['u_pos_frr']:.2f}% | {m_res['u_neg_rr']:.2f}% |"
            )

    # Compute Axis Means
    lines.append("\n## 2. Axis-Level Equal-Weighted Means\n")
    lines.append(
        "| Axis | Method | Mean Seen AUROC | Mean Unseen AUROC | Δ Unseen | Mean Gap | Mean Gap Reduction | Mean PairAcc |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for axis_name, split_subset in [("Action (A1, A2_alt, A3)", ACTION_SPLITS), ("Composition (C1, C2_alt)", COMPOSITION_SPLITS), ("Overall (5 splits)", SPLITS)]:
        base_means = compute_group_means([summary_data[s]["baseline"] for s in split_subset if "baseline" in summary_data[s]])
        for m_key, m_name in methods:
            subset_res = [summary_data[s][m_key] for s in split_subset if m_key in summary_data[s]]
            if len(subset_res) < len(split_subset):
                lines.append(f"| {axis_name} | {m_name} | *Incomplete ({len(subset_res)}/{len(split_subset)})* | - | - | - | - | - |")
                continue
            means = compute_group_means(subset_res)
            delta_u = f"{means['unseen_auroc'] - base_means['unseen_auroc']:+.4f}" if base_means and m_key != "baseline" else "-"
            gap_red = f"{base_means['gap'] - means['gap']:+.4f}" if base_means and m_key != "baseline" else "-"

            lines.append(
                f"| **{axis_name}** | {m_name} | {means['seen_auroc']:.4f} | {means['unseen_auroc']:.4f} | "
                f"{delta_u} | {means['gap']:.4f} | {gap_red} | {means['pair_acc']:.4f} |"
            )

    # Core Scientific Evaluation
    lines.append("\n## 3. Scientific Question Evaluation\n")
    lines.append("> **Question: Does DQ-CGP reduce the seen-to-unseen degradation of Moment-DETR-GMR under the semantic novelty × event existence protocol?**\n")

    report_content = "\n".join(lines) + "\n"
    (REPORTS_ROOT / "FINAL_EVALUATION_REPORT.md").write_text(report_content)
    print("Report generated at:", REPORTS_ROOT / "FINAL_EVALUATION_REPORT.md")


if __name__ == "__main__":
    generate_summary()
