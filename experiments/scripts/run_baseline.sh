#!/bin/bash
# Script chạy baseline evaluation (no Budget Forcing)
# Sử dụng: bash run_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="r1-distill-7B"      # Thay đổi nếu cần
BENCHMARK="math500"
N_SAMPLES=50               # Tăng lên 200 hoặc 500 khi có đủ GPU

echo "=== Running Baseline (no BF) ==="
echo "Model: $MODEL | Benchmark: $BENCHMARK | Samples: $N_SAMPLES"

python "$ROOT_DIR/experiments/evaluation/run_eval.py" \
    --model "$MODEL" \
    --benchmark "$BENCHMARK" \
    --n_wait 0 \
    --n_samples "$N_SAMPLES" \
    --output_dir "$ROOT_DIR/experiments/results" \
    --trigger "Wait"

echo "Done. Results in $ROOT_DIR/experiments/results"
