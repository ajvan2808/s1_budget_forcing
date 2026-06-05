#!/bin/bash
# Script chạy Budget Forcing với n_wait = 0, 1, 2, 4
# Sử dụng: bash run_budget_forcing.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="r1-distill-7B"
BENCHMARK="math500"
N_SAMPLES=50
TRIGGER="Wait"             # Trigger phrase mặc định từ paper

echo "=== Running Budget Forcing Experiments ==="
echo "Model: $MODEL | Benchmark: $BENCHMARK | Trigger: '$TRIGGER'"
echo "n_wait values: 0 1 2 4"

python "$ROOT_DIR/experiments/evaluation/run_eval.py" \
    --model "$MODEL" \
    --benchmark "$BENCHMARK" \
    --n_wait 0 1 2 4 \
    --n_samples "$N_SAMPLES" \
    --output_dir "$ROOT_DIR/experiments/results" \
    --trigger "$TRIGGER"

echo ""
echo "=== Running with alternative trigger: 'Hmm' ==="
python "$ROOT_DIR/experiments/evaluation/run_eval.py" \
    --model "$MODEL" \
    --benchmark "$BENCHMARK" \
    --n_wait 0 1 2 4 \
    --n_samples "$N_SAMPLES" \
    --output_dir "$ROOT_DIR/experiments/results" \
    --trigger "Hmm, let me reconsider"

echo "Done. Check $ROOT_DIR/experiments/results"
