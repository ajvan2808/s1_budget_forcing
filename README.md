# Budget Forcing - Evaluating the Generalizability across Open-Source Language Model Families.

This project studies **Budget Forcing** from *s1: Simple Test-Time Scaling* with the current research focus on **Gap 1: whether Budget Forcing generalizes beyond the Qwen model family used in the paper**.

The repository is no longer just a reproduction scaffold. It now has Phase 2 support for cross-model and cross-benchmark smoke/small-batch runs, but the final experiment artifacts still need to be generated on a machine with stable model-download bandwidth and suitable accelerator support.

## Research Scope

**Primary question:** Does Budget Forcing preserve positive test-time scaling when applied to open-source model families beyond Qwen?

**Current target families:**

| Family | Model key in repo | Role |
| --- | --- | --- |
| Qwen | `qwen2.5-3B`, `qwen2.5-7B`, `r1-distill-7B`, `r1-distill-14B` | Paper-family reference and reasoning-model baseline |
| Gemma | `gemma2-2B`, `gemma2-9B` | Non-Qwen family comparison |
| Phi | `phi3.5-mini`, `phi4-14B` | Non-Qwen family comparison |
| Llama | `llama3.1-8B` | Optional gated-model comparison |

**Current benchmarks:** `math500`, `aime24`, `gsm8k`, `arc_challenge`.

## Project Structure

```text
.
├── README.md
├── PROJECT_BRIEF.md
├── s1. Simple Test-Time Scaling Analysis.md
├── docs/
│   ├── brainstorm.md
│   ├── phase-2-4-taskboard.md
│   ├── sprint-1/
│   ├── sprint-2/
│   ├── sprint-3/
│   └── sprint-4/
├── experiments/
│   ├── budget_forcing/
│   │   ├── decoding.py
│   │   └── metrics.py
│   ├── data/download_s1k.py
│   ├── evaluation/run_eval.py
│   ├── models/model_loader.py
│   ├── notebooks/
│   ├── results/
│   │   └── summary_phase2.py
│   └── scripts/
│       ├── run_baseline.sh
│       ├── run_budget_forcing.sh
│       └── run_phase2_generalizability.sh
└── report/
    ├── main.tex
    ├── outline.md
    ├── references.bib
    └── sections/
```

## Current Status

Updated: 2026-06-11

- Implemented: Budget Forcing decoder, metric helpers, model registry, benchmark registry, evaluation CLI, Phase 2 orchestration script, Phase 2 summary script.
- Implemented model families: Qwen, DeepSeek-R1-Distill-Qwen, Gemma, Phi, optional Llama.
- Implemented benchmarks: MATH500, AIME24, GSM8K, ARC-Challenge.
- Pending: successful Phase 2 run artifacts under `experiments/results/phase2_*`.
- Known blocker: current macOS workspace has MPS but no CUDA; a prior smoke run reached Hugging Face model download and was interrupted by slow transfer.

## Quickstart

Install dependencies:

```bash
uv sync
```

Check the evaluation CLI:

```bash
uv run python experiments/evaluation/run_eval.py --help
```

Run a tiny Gap 1 smoke test:

```bash
MODELS='qwen2.5-3B' \
BENCHMARKS='gsm8k' \
N_WAIT_LIST='0 1' \
N_SAMPLES=2 \
EXTRA_ARGS='--max_tokens 256 --no_4bit' \
bash experiments/scripts/run_phase2_generalizability.sh
```

Run the planned small Phase 2 matrix:

```bash
MODELS='qwen2.5-3B gemma2-2B phi3.5-mini' \
BENCHMARKS='gsm8k arc_challenge' \
N_WAIT_LIST='0 1 2' \
N_SAMPLES=10 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_phase2_generalizability.sh
```

Summarize a completed run directory:

```bash
python experiments/results/summary_phase2.py --results_dir experiments/results/phase2_YYYYMMDD_HHMMSS
```

## Evaluation Metrics

| Metric | Meaning | Current implementation note |
| --- | --- | --- |
| Control | Whether actual compute matches target compute | Full token-budget control needs explicit target/actual token logging; current runs log `n_wait` and thinking-token details. |
| Scaling | Accuracy trend as extra `n_wait` compute is added | Computed from the accuracy sweep. |
| Performance | Best accuracy observed across compute levels | Computed from the accuracy sweep. |

## Main References

- Paper: <https://arxiv.org/abs/2501.12599>
- Code: <https://github.com/simplescaling/s1>
- Dataset: `simplescaling/s1K` on Hugging Face
