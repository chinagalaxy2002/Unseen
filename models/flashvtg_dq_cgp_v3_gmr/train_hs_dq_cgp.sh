#!/usr/bin/env bash
set -euo pipefail

# Standalone sparse DQ-CGP-v3 launcher. It leaves all earlier experiment trees
# untouched. Override any DQ_V3_* environment value for an ablation.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DQ_V3_PYTHON="${DQ_V3_PYTHON:-python}"
DQ_V3_GPU="${DQ_V3_GPU:-0}"
DQ_V3_EPOCHS="${DQ_V3_EPOCHS:-400}"
DQ_V3_OUTPUT="${DQ_V3_OUTPUT:-outputs/flashvtg_dq_cgp_v3_topk4_seed2024}"
DQ_V3_TRAIN_LABELS="${DQ_V3_TRAIN_LABELS:-data/label/Standard/train.jsonl}"
DQ_V3_VAL_LABELS="${DQ_V3_VAL_LABELS:-data/label/Standard/val.jsonl}"
DQ_V3_TEXT_FEATURES="${DQ_V3_TEXT_FEATURES:-Soccergmr/clip_text}"
DQ_V3_CLIP_FEATURES="${DQ_V3_CLIP_FEATURES:-Soccergmr/clip}"
DQ_V3_SLOWFAST_FEATURES="${DQ_V3_SLOWFAST_FEATURES:-Soccergmr/slowfast}"

CUDA_VISIBLE_DEVICES="${DQ_V3_GPU}" "${DQ_V3_PYTHON}" \
  -m models.flashvtg_dq_cgp_v3_gmr.run train \
  models/flashvtg_dq_cgp_v3_gmr/model_config.py \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path "${DQ_V3_TRAIN_LABELS}" \
  --eval_path "${DQ_V3_VAL_LABELS}" \
  --eval_split_name val \
  --v_feat_dirs "${DQ_V3_SLOWFAST_FEATURES}" "${DQ_V3_CLIP_FEATURES}" \
  --t_feat_dir "${DQ_V3_TEXT_FEATURES}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --max_q_l 40 \
  --max_v_l 75 \
  --clip_length 2 \
  --max_windows 5 \
  --lr 3e-5 \
  --lr_drop 400 \
  --wd 1e-4 \
  --n_epoch "${DQ_V3_EPOCHS}" \
  --max_es_cnt 80 \
  --bsz 8 \
  --eval_bsz 1 \
  --eval_epoch 1 \
  --num_workers 0 \
  --device 0 \
  --results_root "${DQ_V3_OUTPUT}" \
  --exp_id soccer_gmr \
  --seed 2024 \
  --hidden_dim 256 \
  --dim_feedforward 1024 \
  --enc_layers 3 \
  --t2v_layers 6 \
  --dummy_layers 2 \
  --nheads 8 \
  --num_dummies 40 \
  --total_prompts 10 \
  --num_prompts 1 \
  --kernel_size 5 \
  --num_conv_layers 1 \
  --num_mlp_layers 5 \
  --use_SRM \
  --input_dropout 0.5 \
  --dropout 0.1 \
  --span_loss_type l1 \
  --lw_reg 1.0 \
  --lw_cls 5.0 \
  --lw_sal 0.0 \
  --lw_saliency 0.0 \
  --lw_wattn 1.0 \
  --lw_ms_align 1.0 \
  --mr_only \
  --eval_full_only \
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1.0 \
  --exist_gate_thd 0.5 \
  --nms_thd 0.7
