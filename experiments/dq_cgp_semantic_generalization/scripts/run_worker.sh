#!/usr/bin/env bash
# Worker script to sequentially train and evaluate splits on a specified GPU
set -euo pipefail

METHOD="${1:?Usage: bash run_worker.sh METHOD GPU_ID [SPLITS...]}"
GPU_ID="${2:?Usage: bash run_worker.sh METHOD GPU_ID [SPLITS...]}"
shift 2
SPLITS=("$@")
if [ ${#SPLITS[@]} -eq 0 ]; then
  SPLITS=("A1" "A2_alt" "A3" "C1" "C2_alt")
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="/home/guoxiangyu/miniconda3/envs/gmr/bin/python"

echo "=========================================================="
echo "Starting worker for method: ${METHOD} on GPU: ${GPU_ID}"
echo "Splits to process: ${SPLITS[*]}"
echo "=========================================================="

for split in "${SPLITS[@]}"; do
  run_dir="${ROOT_DIR}/experiments/dq_cgp_semantic_generalization/runs/${split}/${METHOD}"
  log_file="${run_dir}/console.log"
  diag_file="${run_dir}/diagnostics.json"

  if [ -f "$diag_file" ]; then
    echo "Split ${split} for ${METHOD} already completed (${diag_file} exists). Skipping."
    continue
  fi

  echo "----------------------------------------------------------"
  echo "[$(date -Is)] Starting ${METHOD} on split ${split} (GPU ${GPU_ID})"
  echo "Output directory: ${run_dir}"
  echo "----------------------------------------------------------"

  mkdir -p "$run_dir"

  # Train 100 epochs
  train_script="${ROOT_DIR}/experiments/dq_cgp_semantic_generalization/${METHOD}/train.py"
  "$PYTHON" "$train_script" \
    --split "$split" \
    --gpu "$GPU_ID" \
    --seed 3407 \
    --n_epoch 100 \
    --bsz 16 \
    --eval_bsz 16 \
    --lr 1e-4 \
    --results_dir "$run_dir" \
    --overwrite \
    > "$log_file" 2>&1

  echo "[$(date -Is)] Training finished for ${METHOD} on ${split}. Running evaluation..."

  # Evaluate on full test set and generate diagnostics
  eval_script="${ROOT_DIR}/experiments/dq_cgp_semantic_generalization/scripts/evaluate_run.py"
  "$PYTHON" "$eval_script" \
    --run_dir "$run_dir" \
    --method "$METHOD" \
    --split "$split" \
    --gpu "$GPU_ID" \
    >> "$log_file" 2>&1

  echo "[$(date -Is)] Evaluation complete for ${METHOD} on ${split}."

  # Update aggregate summary
  "$PYTHON" "${ROOT_DIR}/experiments/dq_cgp_semantic_generalization/scripts/aggregate_results.py"

  echo "status=completed" > "${run_dir}/status.txt"
  echo "finished=$(date -Is)" >> "${run_dir}/status.txt"
done

echo "=========================================================="
echo "[$(date -Is)] All splits completed for method: ${METHOD} on GPU ${GPU_ID}"
echo "=========================================================="
