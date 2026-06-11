#!/bin/bash
# Phase 2 orchestration: cross-model x cross-benchmark Budget Forcing sweep.
# Usage:
#   bash experiments/scripts/run_phase2_generalizability.sh
# Optional overrides:
#   MODELS="qwen2.5-3B gemma4-E2B-it llama3-8B phi4-reasoning" BENCHMARKS="gsm8k arc_challenge" N_SAMPLES=20 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_MODELS=("qwen2.5-3B" "gemma4-E2B-it" "llama3-8B" "phi4-reasoning")
DEFAULT_BENCHMARKS=("gsm8k" "arc_challenge")
DEFAULT_N_WAIT=(0 1 2 4)
DEFAULT_TRIGGER="Wait"
DEFAULT_N_SAMPLES=10

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS_ARR=(${MODELS})
else
  MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

if [[ -n "${BENCHMARKS:-}" ]]; then
  # shellcheck disable=SC2206
  BENCHMARKS_ARR=(${BENCHMARKS})
else
  BENCHMARKS_ARR=("${DEFAULT_BENCHMARKS[@]}")
fi

if [[ -n "${N_WAIT_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  N_WAIT_ARR=(${N_WAIT_LIST})
else
  N_WAIT_ARR=("${DEFAULT_N_WAIT[@]}")
fi

TRIGGER="${TRIGGER:-$DEFAULT_TRIGGER}"
N_SAMPLES="${N_SAMPLES:-$DEFAULT_N_SAMPLES}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/experiments/results/phase2_$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "=== Phase 2 Generalizability Sweep ==="
echo "Output dir : $OUT_DIR"
echo "Models     : ${MODELS_ARR[*]}"
echo "Benchmarks : ${BENCHMARKS_ARR[*]}"
echo "n_wait     : ${N_WAIT_ARR[*]}"
echo "Trigger    : $TRIGGER"
echo "n_samples  : $N_SAMPLES"
echo "seed       : $SEED"
echo

for model in "${MODELS_ARR[@]}"; do
  for benchmark in "${BENCHMARKS_ARR[@]}"; do
    echo "--- Running model=$model | benchmark=$benchmark ---"
    python "$ROOT_DIR/experiments/evaluation/run_eval.py" \
      --model "$model" \
      --benchmark "$benchmark" \
      --n_wait "${N_WAIT_ARR[@]}" \
      --n_samples "$N_SAMPLES" \
      --output_dir "$OUT_DIR" \
      --trigger "$TRIGGER" \
      --seed "$SEED" \
      ${EXTRA_ARGS}
  done
done

echo
echo "Sweep finished. Results saved in: $OUT_DIR"
echo "Next: python $ROOT_DIR/experiments/results/summary_phase2.py --results_dir $OUT_DIR"
