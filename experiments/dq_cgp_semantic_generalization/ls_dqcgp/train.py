"""Training and evaluation script for LS-DQCGP on Semantic Novelty × Existence Benchmark.

Conforms strictly to the GMR_Unseen matched 100-epoch protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "training" / "moment_detr_gmr"
for p in (REPO_ROOT, TRAIN_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import eval_epoch, setup_model
from models.moment_detr_gmr.moment_detr import build_model as build_base_detr
from models.moment_detr_gmr.utils.basic_utils import (
    AverageMeter,
    rename_latest_to_best,
    save_checkpoint,
    save_json,
    write_log,
)
from models.moment_detr_gmr.utils.model_utils import count_parameters, ModelEMA

from experiments.dq_cgp_semantic_generalization.ls_dqcgp.model import (
    LSDQCGPModel,
    install_ls_dq_cgp_loss,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seed(seed: int, use_cuda: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def build_dataset_config(opt, data_path: str, load_labels: bool = True, keep_empty_gt: bool = False):
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
        mr_only=bool(getattr(opt, "mr_only", True)),
        keep_empty_gt=keep_empty_gt,
    )


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info("[Epoch %d]", epoch_i + 1)
    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_i + 1} Iteration", leave=False):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    model_ema = None
    if opt.model_ema:
        logger.info("Using model EMA")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = -1.0
    es_cnt = 0
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"

    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval != 0:
            continue

        with torch.no_grad():
            eval_model = model_ema.module if model_ema is not None else model
            metrics, eval_loss_meters, latest_file_paths = eval_epoch(
                epoch_i,
                eval_model,
                val_dataset,
                opt,
                save_submission_filename,
                criterion,
            )

        write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
        logger.info("metrics %s", pprint.pformat(metrics["brief"], indent=4))
        stop_score = metrics["brief"].get("MR-full-mAP", 0)

        if stop_score > prev_best_score:
            prev_best_score = stop_score
            save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
            rename_latest_to_best(latest_file_paths)
            es_cnt = 0
            logger.info("Updated best checkpoint with val score: %.4f", prev_best_score)
        else:
            es_cnt += 1
            if int(opt.max_es_cnt) >= 0 and es_cnt >= int(opt.max_es_cnt):
                logger.info("Early stopping at epoch %d. Best score %.4f", epoch_i + 1, prev_best_score)
                break


def main():
    parser = argparse.ArgumentParser(description="Train LS-DQCGP on GMR_Unseen splits")
    parser.add_argument("--split", required=True, choices=["A1", "A2_alt", "A3", "C1", "C2_alt"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_es_cnt", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    # LS-DQCGP specific frozen parameters
    parser.add_argument("--native_bind_coef", type=float, default=0.2)
    parser.add_argument("--num_basis", type=int, default=16)
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--frf_hidden_dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = REPO_ROOT / "data" / "release" / "semantic_existence_v2" / args.split
    feature_root = REPO_ROOT / "features" / "semantic_existence_v2" / args.split
    video_root = REPO_ROOT / "features" / "charades_video"

    if args.results_dir is None:
        results_dir = (
            REPO_ROOT
            / "experiments"
            / "dq_cgp_semantic_generalization"
            / "runs"
            / args.split
            / "ls_dqcgp"
        )
    else:
        results_dir = Path(args.results_dir)

    if results_dir.exists() and any(results_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists and not empty: {results_dir}")
        for item in results_dir.iterdir():
            if item.name != "console.log":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration for charades_semantic_existence
    option_manager = BaseOptions("moment_detr", "charades_semantic_existence", "clip_slowfast")
    option_manager.parse()
    opt = option_manager.option

    opt.seed = args.seed
    opt.device = str(device)
    opt.n_epoch = args.n_epoch
    opt.bsz = args.bsz
    opt.eval_bsz = args.eval_bsz
    opt.lr = args.lr
    opt.max_es_cnt = args.max_es_cnt
    opt.results_dir = str(results_dir)
    opt.ckpt_filepath = str(results_dir / opt.ckpt_filename)
    opt.train_log_filepath = str(results_dir / opt.train_log_filename)
    opt.eval_log_filepath = str(results_dir / opt.eval_log_filename)

    opt.train_path = str(data_root / "train.jsonl")
    opt.eval_path = str(feature_root / "val_seen.jsonl")
    opt.t_feat_dir = str(feature_root / "clip_text")
    opt.v_feat_dirs = [
        str(video_root / "vid_clip"),
        str(video_root / "vid_slowfast"),
    ]

    # Matched GMR_Unseen baseline protocol
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_exist_head = True
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.5
    opt.exist_pool = "max"

    set_seed(opt.seed, use_cuda=(opt.device == "cuda"))

    # Log metadata
    run_meta = {
        "method": "LS-DQCGP",
        "split": args.split,
        "seed": args.seed,
        "n_epoch": args.n_epoch,
        "lr": args.lr,
        "bsz": args.bsz,
        "native_bind_coef": args.native_bind_coef,
        "num_basis": args.num_basis,
        "prompt_length": args.prompt_length,
        "router_hidden_dim": args.router_hidden_dim,
        "frf_hidden_dim": args.frf_hidden_dim,
        "temperature": args.temperature,
        "started": str(Path("/proc/sys/kernel/hostname").read_text().strip()),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    train_dataset = StartEndDataset(
        **build_dataset_config(opt, opt.train_path, load_labels=True, keep_empty_gt=True)
    )
    val_dataset = StartEndDataset(
        **build_dataset_config(opt, opt.eval_path, load_labels=True, keep_empty_gt=True)
    )

    base_model, criterion = build_base_detr(opt)
    model = LSDQCGPModel(
        base_model=base_model,
        num_basis=args.num_basis,
        prompt_length=args.prompt_length,
        router_hidden_dim=args.router_hidden_dim,
        frf_hidden_dim=args.frf_hidden_dim,
        temperature=args.temperature,
    )
    install_ls_dq_cgp_loss(criterion, model, coefficient=args.native_bind_coef)

    model.to(device)
    criterion.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    count_parameters(model)
    logger.info("Starting LS-DQCGP training on %s with seed %d...", args.split, args.seed)
    train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)


if __name__ == "__main__":
    main()
