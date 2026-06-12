# Agent A Handoff - Sprint 2 Gap 1

## Mission

Run the Gap 1 generalizability experiments and produce traceable artifacts.

## Read First

1. `PROJECT_BRIEF.md`
2. `README.md`
3. `docs/phase-2-4-taskboard.md`
4. `docs/sprint-2/progress.md`

## Required Commands

Tiny smoke run:

```bash
MODELS='qwen2.5-3B' BENCHMARKS='gsm8k' N_WAIT_LIST='0 1' N_SAMPLES=2 EXTRA_ARGS='--max_tokens 256 --no_4bit' bash experiments/scripts/run_phase2_generalizability.sh
```

Small matrix run after smoke succeeds:

```bash
MODELS='qwen2.5-3B gemma4-E2B-it llama3-8B phi4-reasoning' BENCHMARKS='gsm8k arc_challenge' N_WAIT_LIST='0 1 2' N_SAMPLES=10 EXTRA_ARGS='--max_tokens 512 --no_4bit' bash experiments/scripts/run_phase2_generalizability.sh
```

Aggregation:

```bash
python experiments/results/summary_phase2.py --results_dir experiments/results/phase2_YYYYMMDD_HHMMSS
```

## Deliver Back

- Machine/runtime details.
- Exact commands.
- Artifact paths.
- `summary_phase2.csv` and `summary_phase2.md` paths.
- Blocker details if execution fails.

## Rules

- Keep the trigger fixed as `Wait` for this phase.
- Reduce model/benchmark/sample size before changing code.
- Do not fabricate missing results.
