#!/usr/bin/env bash
# Inference and 4-quadrant evaluation script for FlashVTG DQ-CGP
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SPLIT="${1:-A1}"
VARIANT="${VARIANT:-v3}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${FLASH_PYTHON:-/home/guoxiangyu/miniconda3/envs/univtg/bin/python}"
GMR_PYTHON="${GMR_PYTHON:-/home/guoxiangyu/miniconda3/envs/univtg/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/release/semantic_existence_v2/$SPLIT}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT_DIR/features/semantic_existence_v2/$SPLIT}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/results/dq_cgp_semantic_existence/$SPLIT}"

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

run_dir="$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'charadesSTA-*' | sort | tail -1)"
if [[ -z "${run_dir}" || ! -f "${run_dir}/model_best.ckpt" ]]; then
  echo "Error: Best checkpoint not found in ${RUN_ROOT}" >&2
  exit 1
fi

best_ckpt="${run_dir}/model_best.ckpt"
val_preds="${run_dir}/best_charadesSTA_val_preds.jsonl"
test_dir="${RUN_ROOT}/test"
test_preds="${test_dir}/hl_test_submission.jsonl"

mkdir -p "${test_dir}"

echo "=== Running Inference for FlashVTG DQ-CGP ==="
echo "Checkpoint: ${best_ckpt}"
echo "Test file: ${DATA_ROOT}/test.jsonl"
echo "Results dir: ${test_dir}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m "${MODEL_MODULE}" infer \
  "${CONFIG_PATH}" \
  --resume "${best_ckpt}" \
  --eval_split_name test \
  --eval_path "${DATA_ROOT}/test.jsonl" \
  --eval_results_dir "${test_dir}" \
  --device 0 \
  --nms_thd -1 \
  > "${RUN_ROOT}/test_console.log" 2>&1

echo "=== Running 4-Quadrant & Matched-Pair Analysis ==="
"${GMR_PYTHON}" scripts/analyze_semantic_existence.py \
  --release "${DATA_ROOT}" \
  --val-predictions "${val_preds}" \
  --test-predictions "${test_preds}" \
  --output "${RUN_ROOT}/diagnostics.json" \
  > "${RUN_ROOT}/diagnostics_console.log" 2>&1

echo "=== Running Official GMR Evaluation ==="
"${GMR_PYTHON}" eval/eval_main.py \
  --submission_path "${test_preds}" \
  --gt_path "${DATA_ROOT}/test.jsonl" \
  --save_path "${RUN_ROOT}/official_test_metrics.json" \
  > "${RUN_ROOT}/official_console.log" 2>&1

echo "=== Diagnostics Summary ==="
cat "${RUN_ROOT}/diagnostics.json"
