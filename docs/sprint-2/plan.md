# Sprint 2 Plan - Gap 1 Generalizability

Updated: 2026-06-11

## Objective

Produce the first real evidence for Gap 1: whether Budget Forcing generalizes beyond Qwen-family models.

## Deliverables

- At least one successful `experiments/results/phase2_*` run directory.
- Raw JSON outputs for each model/benchmark pair that completes.
- `summary_phase2.csv` and `summary_phase2.md` generated from the run directory.
- Report-ready table schema for model, benchmark, `n_wait`, accuracy, scaling, and performance.

## Completed Implementation

- [x] Model registry includes Qwen, Gemma, Phi, optional Llama.
- [x] Benchmark registry includes MATH500, AIME24, GSM8K, ARC-Challenge.
- [x] Phase 2 sweep script exists.
- [x] Phase 2 summary script exists.

## Execution Tasks

1. Run the tiny smoke test.
2. If it succeeds, run the small matrix.
3. Aggregate the run directory with `summary_phase2.py`.
4. Record the exact command, machine, output path, and blockers in `docs/sprint-2/progress.md`.
5. Hand `summary_phase2.md` to the report workflow.

## Non-Goals

- Trigger phrase optimization.
- Anti-repetition decoding changes.
- New training or SFT.
- Broad benchmark expansion before the first Phase 2 artifact exists.
