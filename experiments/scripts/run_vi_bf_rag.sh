#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_vi_bf_rag.sh — Sweep script for Vietnamese BF+RAG experiments
#
# Controlled entirely by environment variables (no positional args needed).
# All env vars have sensible defaults so you can override just what you need.
#
# Quick examples:
#   # Smoke test (5 samples, no 4-bit, 1 model)
#   MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
#   CONDITIONS='BF_only RAG_only BF_RAG' \
#   N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
#   EXTRA_ARGS='--max_tokens 512 --no_4bit' \
#   bash experiments/scripts/run_vi_bf_rag.sh
#
#   # Small matrix (2 models × 2 benchmarks × 20 samples)
#   MODELS='qwen2.5-3B vinallama-7b' BENCHMARKS='vi_gsm8k vimmlu' \
#   CONDITIONS='BF_only RAG_only BF_RAG' \
#   N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
#   bash experiments/scripts/run_vi_bf_rag.sh
#
# Environment variables:
#   MODELS          Space-separated model keys (default: qwen2.5-3B)
#   BENCHMARKS      Space-separated benchmark keys (default: vi_gsm8k)
#   CONDITIONS      Space-separated conditions (default: BF_only RAG_only BF_RAG)
#   N_WAIT_LIST     Space-separated n_wait values (default: 0 1 2)
#   N_SAMPLES       Questions per run (default: 100)
#   OUTPUT_DIR      Root output directory (default: experiments/results)
#   RAG_INDEX_DIR   Path to FAISS index (optional; auto-detected if omitted)
#   EXTRA_ARGS      Extra args forwarded to run_eval_vi.py (e.g. '--no_4bit')
#   SEED            Random seed (default: 42)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODELS="${MODELS:-qwen2.5-3B}"
BENCHMARKS="${BENCHMARKS:-vi_gsm8k}"
CONDITIONS="${CONDITIONS:-BF_only RAG_only BF_RAG}"
N_WAIT_LIST="${N_WAIT_LIST:-0 1 2}"
N_SAMPLES="${N_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SEED="${SEED:-42}"

# ── Locate repo root (where this script is called from) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/experiments/evaluation/run_eval_vi.py"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "[ERROR] run_eval_vi.py not found at: $EVAL_SCRIPT"
    echo "        Run this script from the repo root or check your paths."
    exit 1
fi

echo "======================================================================"
echo "Vietnamese BF+RAG Sweep"
echo "  Models:     $MODELS"
echo "  Benchmarks: $BENCHMARKS"
echo "  Conditions: $CONDITIONS"
echo "  n_wait:     $N_WAIT_LIST"
echo "  n_samples:  $N_SAMPLES"
echo "  output_dir: $OUTPUT_DIR"
echo "  seed:       $SEED"
[[ -n "$RAG_INDEX_DIR" ]] && echo "  RAG index:  $RAG_INDEX_DIR"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra_args: $EXTRA_ARGS"
echo "======================================================================"
echo ""

# Build n_wait args (space-separated → repeated --n_wait args)
N_WAIT_ARGS=""
for nw in $N_WAIT_LIST; do
    N_WAIT_ARGS="$N_WAIT_ARGS $nw"
done
N_WAIT_ARGS="--n_wait $N_WAIT_ARGS"

# Build RAG index arg
RAG_ARGS=""
if [[ -n "$RAG_INDEX_DIR" ]]; then
    RAG_ARGS="--rag_index_dir $RAG_INDEX_DIR"
fi

# ── Sweep ─────────────────────────────────────────────────────────────────────
TOTAL_RUNS=0
FAILED_RUNS=0

for MODEL in $MODELS; do
    for BENCHMARK in $BENCHMARKS; do
        echo "----------------------------------------------------------------------"
        echo "  Model: $MODEL  |  Benchmark: $BENCHMARK"
        echo "----------------------------------------------------------------------"

        CMD="python $EVAL_SCRIPT \
            --model $MODEL \
            --benchmark $BENCHMARK \
            --conditions $CONDITIONS \
            $N_WAIT_ARGS \
            --n_samples $N_SAMPLES \
            --output_dir $OUTPUT_DIR \
            --seed $SEED \
            $RAG_ARGS \
            $EXTRA_ARGS"

        echo "  Running: $CMD"
        echo ""

        if eval "$CMD"; then
            echo ""
            echo "  [OK] $MODEL × $BENCHMARK completed."
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        else
            EXIT_CODE=$?
            echo ""
            echo "  [FAIL] $MODEL × $BENCHMARK exited with code $EXIT_CODE"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            # Continue with next combination (don't abort entire sweep)
        fi
        echo ""
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "Sweep complete: $((TOTAL_RUNS - FAILED_RUNS))/$TOTAL_RUNS runs succeeded."
echo ""
echo "Aggregate results with:"
echo "  python experiments/results/summary_vi.py \\"
echo "      --results_dir $OUTPUT_DIR"
echo "======================================================================"

if [[ $FAILED_RUNS -gt 0 ]]; then
    exit 1
fi
