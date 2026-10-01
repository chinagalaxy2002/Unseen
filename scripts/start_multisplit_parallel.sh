#!/usr/bin/env bash
# Start parallel multi-split workers across GPU 0 and GPU 1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE_DIR="$ROOT_DIR/results/dq_cgp_semantic_existence/queue"
mkdir -p "$QUEUE_DIR"

# Clean any previous queue metadata for the pending splits if requested
for split in A2_alt A3 C2_alt; do
  if [[ ! -d "$ROOT_DIR/results/dq_cgp_semantic_existence/$split" ]]; then
    rm -f "$QUEUE_DIR/${split}.claimed" "$QUEUE_DIR/${split}.done" "$QUEUE_DIR/${split}.failed" "$QUEUE_DIR/${split}.gpu"
  fi
done
rmdir "$QUEUE_DIR/lock" 2>/dev/null || true

# Kill existing worker tmux sessions if running
tmux kill-session -t dq_worker_gpu0 2>/dev/null || true
tmux kill-session -t dq_worker_gpu1 2>/dev/null || true

echo "Starting Worker 0 on GPU 0 (tmux: dq_worker_gpu0)..."
tmux new-session -d -s dq_worker_gpu0 "bash scripts/queue_worker.sh 0 2>&1 | tee results/dq_cgp_semantic_existence/queue/worker_gpu0.log"

echo "Starting Worker 1 on GPU 1 (tmux: dq_worker_gpu1)..."
tmux new-session -d -s dq_worker_gpu1 "bash scripts/queue_worker.sh 1 2>&1 | tee results/dq_cgp_semantic_existence/queue/worker_gpu1.log"

echo "Both workers launched successfully."
echo "Active tmux sessions: dq_worker_gpu0, dq_worker_gpu1"
echo "Queue logs in: results/dq_cgp_semantic_existence/queue/"
