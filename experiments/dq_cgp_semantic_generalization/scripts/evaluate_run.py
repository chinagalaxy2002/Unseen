"""Evaluate a trained model checkpoint on the full test set and compute diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "training" / "moment_detr_gmr"
for p in (REPO_ROOT, TRAIN_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate
from evaluate import compute_mr_results
from models.moment_detr_gmr.moment_detr import build_model as build_base_detr
from models.moment_detr_gmr.utils.basic_utils import save_jsonl

from experiments.dq_cgp_semantic_generalization.ls_dqcgp.model import LSDQCGPModel
from experiments.dq_cgp_semantic_generalization.dq_cgp_v3.model import DQCGPv3Model

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def build_dataset_config(opt, data_path: str, load_labels: bool = False):
    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
        mr_only=True,
        keep_empty_gt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on full test split")
    parser.add_argument("--run_dir", required=True, type=str, help="Path to run directory")
    parser.add_argument("--method", required=True, choices=["ls_dqcgp", "dq_cgp_v3"])
    parser.add_argument("--split", required=True, choices=["A1", "A2_alt", "A3", "C1", "C2_alt"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_bsz", type=int, default=16)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_root = REPO_ROOT / "data" / "release" / "semantic_existence_v2" / args.split
    feature_root = REPO_ROOT / "features" / "semantic_existence_v2" / args.split
    video_root = REPO_ROOT / "features" / "charades_video"
    test_path = data_root / "test.jsonl"
    val_pred_path = run_dir / "best_charades_semantic_existence_val_preds.jsonl"
    if not val_pred_path.exists():
        raise FileNotFoundError(f"Val predictions not found: {val_pred_path}")

    test_out_dir = run_dir / "test"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    test_submission_path = test_out_dir / "test_submission.jsonl"

    option_manager = BaseOptions("moment_detr", "charades_semantic_existence", "clip_slowfast")
    option_manager.parse()
    opt = option_manager.option

    opt.device = str(device)
    opt.results_dir = str(run_dir)
    opt.eval_bsz = args.eval_bsz
    opt.t_feat_dir = str(feature_root / "clip_text")
    opt.v_feat_dirs = [
        str(video_root / "vid_clip"),
        str(video_root / "vid_slowfast"),
    ]
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_exist_head = True
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.5
    opt.exist_pool = "max"

    test_dataset = StartEndDataset(
        **build_dataset_config(opt, str(test_path), load_labels=False)
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        shuffle=False,
    )

    base_model, criterion = build_base_detr(opt)
    if args.method == "ls_dqcgp":
        model = LSDQCGPModel(base_model=base_model)
    else:
        model = DQCGPv3Model(base_model=base_model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    logger.info("Computing predictions on full test set (%d queries)...", len(test_dataset))
    with torch.no_grad():
        results, _ = compute_mr_results(0, model, test_loader, opt, criterion=None)
    save_jsonl(results, str(test_submission_path))
    logger.info("Saved test submission to %s", test_submission_path)

    # Run diagnostics using analyze_semantic_existence.py
    diag_out_path = run_dir / "diagnostics.json"
    analyze_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_semantic_existence.py"),
        "--release", str(data_root),
        "--val-predictions", str(val_pred_path),
        "--test-predictions", str(test_submission_path),
        "--output", str(diag_out_path),
    ]
    logger.info("Running diagnostics: %s", " ".join(analyze_cmd))
    res = subprocess.run(analyze_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Diagnostics failed:\n%s", res.stderr)
        raise RuntimeError("Diagnostics failed")
    logger.info("Diagnostics saved to %s", diag_out_path)

    # Run official GMR evaluation
    official_out_path = run_dir / "official_test_metrics.json"
    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "eval_main.py"),
        "--submission_path", str(test_submission_path),
        "--gt_path", str(test_path),
        "--save_path", str(official_out_path),
    ]
    logger.info("Running official GMR evaluation: %s", " ".join(eval_cmd))
    res = subprocess.run(eval_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("Official eval notice:\n%s", res.stderr)
    logger.info("Official metrics saved to %s", official_out_path)


if __name__ == "__main__":
    main()
