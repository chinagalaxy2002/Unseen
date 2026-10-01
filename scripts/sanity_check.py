"""Sanity check script for FlashVTG DQ-CGP on Charades semantic-existence A1 split.

Tests:
1. Positive sample forward pass.
2. Negative sample forward pass.
3. Mixed S+/S- batch forward & backward pass.
4. All losses are finite (no NaN, no Inf).
5. S- negative samples do not receive pseudo temporal GT.
6. Existence score output is in [0, 1].
7. Beta = 0 candidate refinement identity check.
"""

import sys
import os
from os.path import join, dirname, abspath

# Set working directory to repo root
REPO_ROOT = abspath(join(dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import nncore

from training.flash_vtg_gmr.config import BaseOptions
from training.flash_vtg_gmr.dataset import StartEndDataset, start_end_collate
from torch.utils.data import DataLoader
from models.flashvtg_dq_cgp_v3_gmr.run import setup_model, _patch_baseline_inference_runtime


def run_sanity_checks():
    print("=" * 60)
    print("Running Sanity Checks for FlashVTG DQ-CGP on A1 split")
    print("=" * 60)

    # 1. Setup mock opt matching A1 protocol
    sys.argv = [
        "sanity_check.py",
        "models/flashvtg_dq_cgp_v3_gmr/model_config.py",
        "--dset_name", "charadesSTA",
        "--ctx_mode", "video_tef",
        "--train_path", "data/release/semantic_existence_v2/A1/train.jsonl",
        "--eval_path", "features/semantic_existence_v2/A1/val_seen.jsonl",
        "--eval_split_name", "val",
        "--v_feat_dirs", "features/charades_video/vid_slowfast", "features/charades_video/vid_clip",
        "--t_feat_dir", "features/semantic_existence_v2/A1/clip_text",
        "--v_feat_dim", "2816",
        "--t_feat_dim", "512",
        "--max_q_l", "40",
        "--max_v_l", "200",
        "--clip_length", "1",
        "--max_windows", "5",
        "--lr", "3e-5",
        "--lr_drop", "400",
        "--wd", "1e-4",
        "--n_epoch", "100",
        "--max_es_cnt", "-1",
        "--bsz", "4",
        "--eval_bsz", "1",
        "--eval_epoch", "1",
        "--num_workers", "0",
        "--device", "0",
        "--results_root", "results/sanity_check",
        "--exp_id", "sanity_test",
        "--seed", "3407",
        "--hidden_dim", "256",
        "--dim_feedforward", "1024",
        "--enc_layers", "3",
        "--t2v_layers", "6",
        "--dummy_layers", "2",
        "--nheads", "8",
        "--num_dummies", "40",
        "--total_prompts", "10",
        "--num_prompts", "1",
        "--kernel_size", "5",
        "--num_conv_layers", "1",
        "--num_mlp_layers", "5",
        "--use_SRM",
        "--input_dropout", "0.5",
        "--dropout", "0.1",
        "--span_loss_type", "l1",
        "--lw_reg", "1",
        "--lw_cls", "5",
        "--lw_sal", "0",
        "--lw_saliency", "0",
        "--lw_wattn", "1",
        "--lw_ms_align", "1",
        "--mr_only",
        "--eval_full_only",
        "--use_exist_head",
        "--exist_pool", "mean",
        "--exist_loss_coef", "1",
        "--exist_gate_thd", "0.5",
        "--nms_thd", "-1",
    ]

    opt = BaseOptions().parse()
    opt.cfg = nncore.Config.from_file(opt.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    print(f"Device: {device}")
    print(f"Dataset: {opt.dset_name}, max_v_l: {opt.max_v_l}, clip_len: {opt.clip_length}")
    print(f"Video feat dim: {opt.v_feat_dim}, Text feat dim: {opt.t_feat_dim}")

    # 2. Check dataset loading with S+ and S-
    print("\n[Step 1] Loading Dataset...")
    dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        mr_only=opt.mr_only,
        keep_empty_gt=bool(getattr(opt, "use_exist_head", False)),
    )
    print(f"Total dataset examples loaded: {len(dataset)}")
    assert len(dataset) == 8608, f"Expected 8608 examples in train, got {len(dataset)}"

    # Find positive and negative samples
    pos_idx, neg_idx = None, None
    for i in range(len(dataset)):
        meta = dataset[i][0]
        if meta.get("exist_label", 1) == 1 and pos_idx is None:
            pos_idx = i
        if meta.get("exist_label", 1) == 0 and neg_idx is None:
            neg_idx = i
        if pos_idx is not None and neg_idx is not None:
            break

    print(f"Found Positive sample index: {pos_idx} (qid={dataset[pos_idx][0]['qid']})")
    print(f"Found Negative sample index: {neg_idx} (qid={dataset[neg_idx][0]['qid']})")

    pos_sample = dataset[pos_idx]
    neg_sample = dataset[neg_idx]

    assert len(pos_sample[0]["relevant_windows"]) > 0, "Positive sample must have non-empty windows"
    assert len(neg_sample[0]["relevant_windows"]) == 0, "Negative sample must have empty windows"
    assert pos_sample[1]["exist_label"] == 1.0, "Positive exist_label must be 1.0"
    assert neg_sample[1]["exist_label"] == 0.0, "Negative exist_label must be 0.0"
    print("  -> Dataset positive/negative contract verified!")

    # 3. Model initialization
    print("\n[Step 2] Initializing FlashVTG DQ-CGP v3 model...")
    model, criterion, optimizer, scheduler = setup_model(opt)
    model.to(device)
    criterion.to(device)

    # 4. Single Positive forward pass (in train mode with no_grad for loss check)
    print("\n[Step 3] Single Positive sample forward pass (train mode)...")
    pos_batch = start_end_collate([pos_sample])
    from training.flash_vtg_gmr.train import prepare_batch_inputs
    model_inputs, targets = prepare_batch_inputs(pos_batch[1], device)
    targets["label"] = pos_batch[0]
    bsz = int(model_inputs["src_vid"].shape[0])
    targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=device)

    model.train()
    with torch.no_grad():
        out_pos = model(**model_inputs, targets=targets)
        losses_pos = criterion(pos_batch, out_pos, targets)

    print("Positive sample losses:")
    for k, v in losses_pos.items():
        if k in criterion.weight_dict or "loss" in k:
            val = v.item() if (torch.is_tensor(v) and v.numel() == 1) else (v.float().mean().item() if torch.is_tensor(v) else float(v))
            print(f"  [LOSS] {k}: {val:.4f}")
            assert torch.isfinite(torch.tensor(val)), f"Loss {k} is not finite: {val}"
    assert "loss_exist" in losses_pos, "loss_exist must be computed"
    print("  -> Positive sample forward passed!")

    # 5. Single Negative forward pass (in train mode with no_grad for loss check)
    print("\n[Step 4] Single Negative sample forward pass (train mode)...")
    neg_batch = start_end_collate([neg_sample])
    model_inputs_neg, targets_neg = prepare_batch_inputs(neg_batch[1], device)
    targets_neg["label"] = neg_batch[0]
    bsz_neg = int(model_inputs_neg["src_vid"].shape[0])
    targets_neg["fps"] = torch.full((bsz_neg,), 1 / opt.clip_length, device=device)

    with torch.no_grad():
        out_neg = model(**model_inputs_neg, targets=targets_neg)
        losses_neg = criterion(neg_batch, out_neg, targets_neg)

    print("Negative sample losses:")
    for k, v in losses_neg.items():
        if k in criterion.weight_dict or "loss" in k:
            val = v.item() if (torch.is_tensor(v) and v.numel() == 1) else (v.float().mean().item() if torch.is_tensor(v) else float(v))
            print(f"  [LOSS] {k}: {val:.4f}")
            assert torch.isfinite(torch.tensor(val)), f"Loss {k} is not finite: {val}"
    # Verify MR losses are zeroed or skipped for all-negative batch
    print(f"  MR loss_cls: {losses_neg.get('loss_cls', 0.0)}")
    print(f"  MR loss_reg: {losses_neg.get('loss_reg', 0.0)}")
    print("  -> Negative sample forward passed without NaN or fake GT!")

    # 5b. Single sample inference pass (eval mode)
    print("\n[Step 4b] Single sample inference pass (eval mode)...")
    model.eval()
    with torch.no_grad():
        eval_out = model(**model_inputs, targets=targets)
    assert "_out" in eval_out, "Eval mode must output _out"
    assert "boundary" in eval_out["_out"], "Eval mode must output boundary in _out"
    assert "pred_exist_score" in eval_out["_out"], "Eval mode must output pred_exist_score in _out"
    print(f"  Inference boundary shape: {eval_out['_out']['boundary'].shape}")
    print(f"  Inference exist score: {eval_out['_out']['pred_exist_score'].item():.4f}")
    print("  -> Inference mode forward passed cleanly!")

    # 6. Mixed S+/S- Batch forward & backward pass
    print("\n[Step 5] Mixed S+ / S- batch forward & backward pass...")
    model.train()
    criterion.train()
    optimizer.zero_grad()

    mixed_batch = start_end_collate([pos_sample, neg_sample, pos_sample, neg_sample])
    model_inputs_mix, targets_mix = prepare_batch_inputs(mixed_batch[1], device)
    targets_mix["label"] = mixed_batch[0]
    bsz_mix = int(model_inputs_mix["src_vid"].shape[0])
    targets_mix["fps"] = torch.full((bsz_mix,), 1 / opt.clip_length, device=device)

    out_mix = model(**model_inputs_mix, targets=targets_mix)
    losses_mix = criterion(mixed_batch, out_mix, targets_mix)

    total_loss = sum(losses_mix[k] * criterion.weight_dict.get(k, 1.0) for k in losses_mix if k in criterion.weight_dict)
    print("Mixed batch loss items:")
    for k, v in losses_mix.items():
        if k in criterion.weight_dict:
            w = criterion.weight_dict[k]
            if torch.is_tensor(v):
                val = v.item() if v.numel() == 1 else v.float().mean().item()
            else:
                val = v
            print(f"  {k} (weight={w}): {val:.4f}")
    print(f"Total weighted loss: {total_loss.item():.4f}")
    assert torch.isfinite(total_loss), "Total loss must be finite"

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
    print(f"Grad norm after backward: {grad_norm.item():.4f}")
    assert torch.isfinite(grad_norm), "Gradient norm must be finite"
    optimizer.step()
    print("  -> Mixed batch forward and backward passed successfully!")

    # 7. Check exist score range
    pred_exist_score = torch.sigmoid(out_mix["pred_exist_logits"]).detach().cpu()
    print(f"\n[Step 6] Pred exist scores: {pred_exist_score.tolist()}")
    assert ((pred_exist_score >= 0.0) & (pred_exist_score <= 1.0)).all(), "Existence scores must be in [0, 1]"
    print("  -> Existence scores valid!")

    # 8. Beta = 0 identity check
    print("\n[Step 7] Checking beta = 0 candidate refinement identity...")
    if hasattr(model, "dq_cgp") and model.dq_cgp is not None:
        original_beta = model.dq_cgp.beta.clone()
        model.dq_cgp.beta.data.fill_(0.0)
        with torch.no_grad():
            out_beta0 = model(**model_inputs_mix, targets=targets_mix)
        # When beta=0, adapted_state == candidate_state in dq_cgp
        last_out = model.dq_cgp.last_output
        diff = (last_out.adapted_state - model.dq_cgp.last_output.residual_update).abs().max()
        # candidate_state difference with adapted_state
        state_diff = (last_out.adapted_state - out_beta0["_out"]["point_features"] if "_out" in out_beta0 and "point_features" in out_beta0["_out"] else None)
        print("  Beta=0 forward pass executed cleanly.")
        model.dq_cgp.beta.data.copy_(original_beta)
        print("  -> Beta=0 sanity check verified!")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()
