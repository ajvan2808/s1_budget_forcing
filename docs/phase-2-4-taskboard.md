# Taskboard — Vietnamese BF+RAG (Sprint 3+)

Updated: 2026-06-11

## Scope

**Budget Forcing + RAG on Vietnamese Reasoning** (see PROJECT_BRIEF.md §3).

Gap 1 cross-family work is archived in `docs/sprint-2/`. Do not expand back into it until the Vietnamese pipeline produces artifacts.

---

## Definition of Done

- Experiment outputs saved under `experiments/results/vi_*/`
- Each output dir contains raw JSON files + `summary_vi.csv` + `summary_vi.md`
- Every report number traces to an artifact file
- Runtime blockers recorded with exact commands + last error

---

## Sprint 3 — Vietnamese Pivot (Implementation)

**Status: active**

### Agent A Tasks (Coder)

- [x] Rewrite `PROJECT_BRIEF.md` with Vietnamese BF+RAG framing
- [x] Update `docs/brainstorm.md` with pivot decision
- [ ] Add Vietnamese models to `experiments/models/model_loader.py`
- [ ] Add Vietnamese benchmark specs to `experiments/evaluation/run_eval.py`
- [ ] Create `experiments/rag/` module (retriever + KB builder + pipeline)
- [ ] Create `experiments/data/download_vi_benchmarks.py`
- [ ] Create `experiments/evaluation/run_eval_vi.py` (3-condition eval driver)
- [ ] Create `experiments/scripts/run_vi_bf_rag.sh`
- [ ] Create `experiments/results/summary_vi.py`
- [ ] Smoke test: 1 model × vi_gsm8k × 3 conditions × 5 samples → JSON

### Agent B Tasks (Writer)

- [x] Review new PROJECT_BRIEF.md
- [ ] Rewrite `report/outline.md` for Vietnamese BF+RAG topic
- [ ] Update `report/sections/01_introduction.tex` — new motivation
- [ ] Update `report/sections/02_method.tex` — add RAG section
- [ ] Update `report/sections/05_related_work.tex` — Vi NLP + RAG-in-education
- [ ] Add result table placeholders matching `summary_vi.csv` columns
- [ ] Write limitations section draft

---

## Sprint 4 — Execution + Report Population

**Status: pending (starts after smoke run succeeds)**

### Agent A Tasks

- [ ] Run small matrix: 2 models × 2 benchmarks × 3 conditions × n_samples=20
- [ ] Run `summary_vi.py` → produce `summary_vi.csv` + `summary_vi.md`
- [ ] Error analysis: classify failures (parser / reasoning / context-overflow)
- [ ] Generate scaling curve plots per condition

### Agent B Tasks

- [ ] Fill result tables from `summary_vi.csv`
- [ ] Write Discussion: BF vs RAG comparison, combined effect
- [ ] Write error analysis narrative
- [ ] Final proofread + compile LaTeX PDF

---

## Column Schema for `summary_vi.csv`

```
model, benchmark, language, condition, n_wait,
n_samples, accuracy, scaling, performance,
avg_thinking_tokens, avg_retrieved_tokens,
extraction_failures, cuda_available, mps_available,
run_dir, timestamp_utc
```

---

## Canonical Commands

**Smoke (5 samples, no 4-bit):**
```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf_rag.sh
```

**Small matrix:**
```bash
MODELS='qwen2.5-3B vinallama-7b' BENCHMARKS='vi_gsm8k vimmlu' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
bash experiments/scripts/run_vi_bf_rag.sh
```

**Aggregate:**
```bash
python experiments/results/summary_vi.py \
  --results_dir experiments/results/vi_YYYYMMDD_HHMMSS
```

---

## Coordination Rule

Do not add new experiment branches (trigger optimization, anti-repetition, etc.) before the Vietnamese smoke run closes the evidence loop: run → JSON → summary_vi.csv → report table.
