#!/usr/bin/env bash
# Train FlashVTG DQ-CGP on semantic existence splits (default: A1)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SPLIT="${1:-A1}"
VARIANT="${VARIANT:-v3}"  # v3 (default, recommended in plan.md) or v2 (corresponding to epoch32 ckpt)
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEMANTIC_SEED:-3407}"
EPOCHS="${EPOCHS:-100}"
BSZ="${BSZ:-8}"

PYTHON="${FLASH_PYTHON:-/home/guoxiangyu/miniconda3/envs/univtg/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/release/semantic_existence_v2/$SPLIT}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT_DIR/features/semantic_existence_v2/$SPLIT}"
VIDEO_ROOT="${VIDEO_ROOT:-$ROOT_DIR/features/charades_video}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/results/dq_cgp_semantic_existence/$SPLIT}"

mkdir -p "${RUN_ROOT}"

if [[ "${VARIANT}" == "v3" ]]; then
  MODEL_MODULE="models.flashvtg_dq_cgp_v3_gmr.run"
  CONFIG_PATH="models/flashvtg_dq_cgp_v3_gmr/model_config.py"
elif [[ "${VARIANT}" == "v2" ]]; then
  MODEL_MODULE="models.flashvtg_dq-cgp-gmr-v2.run"
  CONFIG_PATH="models/flashvtg_dq-cgp-gmr-v2/model_config.py"
else
  echo "Unknown variant: ${VARIANT}. Use v3 or v2." >&2
  exit 2
fi

echo "=== Training FlashVTG DQ-CGP ==="
echo "Split: ${SPLIT}"
echo "Variant: ${VARIANT}"
echo "Config: ${CONFIG_PATH}"
echo "GPU: ${GPU}, Seed: ${SEED}, Epochs: ${EPOCHS}, Batch size: ${BSZ}"
echo "Data root: ${DATA_ROOT}"
echo "Run root: ${RUN_ROOT}"

cat <<EOF > "${RUN_ROOT}/run_metadata.txt"
split=${SPLIT}
variant=${VARIANT}
seed=${SEED}
epochs=${EPOCHS}
batch_size=${BSZ}
started=$(date -Is)
EOF

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m "${MODEL_MODULE}" train \
  "${CONFIG_PATH}" \
  --dset_name charadesSTA \
  --ctx_mode video_tef \
  --train_path "${DATA_ROOT}/train.jsonl" \
  --eval_path "${FEATURE_ROOT}/val_seen.jsonl" \
  --eval_split_name val \
  --v_feat_dirs "${VIDEO_ROOT}/vid_slowfast" "${VIDEO_ROOT}/vid_clip" \
  --t_feat_dir "${FEATURE_ROOT}/clip_text" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --max_q_l 40 \
  --max_v_l 200 \
  --clip_length 1 \
  --max_windows 5 \
  --lr 3e-5 \
  --lr_drop 400 \
  --wd 1e-4 \
  --n_epoch "${EPOCHS}" \
  --max_es_cnt -1 \
  --bsz "${BSZ}" \
  --eval_bsz 1 \
  --eval_epoch 1 \
  --num_workers 0 \
  --device 0 \
  --results_root "${RUN_ROOT}" \
  --exp_id "dq_cgp_${VARIANT}_seed${SEED}_${EPOCHS}ep" \
  --seed "${SEED}" \
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
  --lw_reg 1 \
  --lw_cls 5 \
  --lw_sal 0 \
  --lw_saliency 0 \
  --lw_wattn 1 \
  --lw_ms_align 1 \
  --mr_only \
  --eval_full_only \
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1 \
  --exist_gate_thd 0.5 \
  --nms_thd -1 \
  > "${RUN_ROOT}/console.log" 2>&1

status=$?
printf '%s\n' "${status}" > "${RUN_ROOT}/exit_code"
printf 'finished=%s\nexit_code=%s\n' "$(date -Is)" "${status}" >> "${RUN_ROOT}/run_metadata.txt"
exit "${status}"
