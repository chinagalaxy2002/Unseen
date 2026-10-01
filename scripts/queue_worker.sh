#!/usr/bin/env bash
# Dynamic queue worker for multi-split FlashVTG DQ-CGP training & evaluation
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${1:-0}"
QUEUE_DIR="$ROOT_DIR/results/dq_cgp_semantic_existence/queue"
mkdir -p "$QUEUE_DIR"
LOCK_DIR="$QUEUE_DIR/lock"

get_next_task() {
  local max_attempts=120
  local attempt=0
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 0.5
    ((attempt++))
    if [[ $attempt -ge $max_attempts ]]; then
      rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
  done
  
  local task=""
  for split in A2_alt A3 C2_alt; do
    if [[ ! -f "$QUEUE_DIR/${split}.claimed" && ! -f "$QUEUE_DIR/${split}.done" && ! -f "$QUEUE_DIR/${split}.failed" ]]; then
      touch "$QUEUE_DIR/${split}.claimed"
      echo "$GPU" > "$QUEUE_DIR/${split}.gpu"
      printf 'claimed=%s\ngpu=%s\n' "$(date -Is)" "$GPU" >> "$QUEUE_DIR/${split}.claimed"
      task="$split"
      break
    fi
  done
  
  rmdir "$LOCK_DIR" 2>/dev/null || true
  echo "$task"
}

echo "[Worker GPU $GPU] Started at $(date -Is)"
while true; do
  split="$(get_next_task)"
  if [[ -z "$split" ]]; then
    echo "[Worker GPU $GPU] No more unassigned tasks in queue. Exiting at $(date -Is)."
    break
  fi
  
  echo "[Worker GPU $GPU] ========================================="
  echo "[Worker GPU $GPU] Claimed split: $split at $(date -Is)"
  echo "[Worker GPU $GPU] ========================================="
  
  # 1. Training
  echo "[Worker GPU $GPU] [1/2] Launching training for $split on GPU $GPU..."
  if CUDA_VISIBLE_DEVICES="$GPU" bash scripts/train_dq_cgp_semantic_existence.sh "$split"; then
    echo "[Worker GPU $GPU] Training for $split succeeded at $(date -Is)"
  else
    echo "[Worker GPU $GPU] ERROR: Training for $split failed" >&2
    touch "$QUEUE_DIR/${split}.failed"
    continue
  fi
  
  # 2. Inference & Evaluation
  echo "[Worker GPU $GPU] [2/2] Launching inference & evaluation for $split on GPU $GPU..."
  if CUDA_VISIBLE_DEVICES="$GPU" bash scripts/infer_dq_cgp_semantic_existence.sh "$split"; then
    echo "[Worker GPU $GPU] Inference for $split succeeded at $(date -Is)"
    touch "$QUEUE_DIR/${split}.done"
  else
    echo "[Worker GPU $GPU] ERROR: Inference for $split failed" >&2
    touch "$QUEUE_DIR/${split}.failed"
  fi
  
  echo "[Worker GPU $GPU] Finished pipeline for $split at $(date -Is)"
done
