# Sprint 3 Plan — Vietnamese BF+RAG Pivot

Updated: 2026-06-11

## Goal

Deliver a runnable 3-condition evaluation pipeline for Vietnamese reasoning tasks.
Success = smoke run produces JSON + summary_vi.csv.

## Duration

Sprint 3: ~5 days implementation (see taskboard for status tracking).

## Work Streams

| Stream | Owner | Output |
|--------|-------|--------|
| Code infrastructure | Agent A | All new/updated Python files + shell script |
| Report framing | Agent B | outline.md + intro/method section drafts |

These run in parallel. Agent B does NOT need experiment results to start writing;
it writes from prior literature + hypotheses. Results fill placeholders in Sprint 4.

---

## Agent A Checklist (this sprint)

1. Update `experiments/models/model_loader.py`
   - Add: `vinallama-7b`, `vistral-7b`, `seallm-7b`

2. Update `experiments/evaluation/run_eval.py`
   - Add: `vi_gsm8k`, `vimmlu` benchmark specs to `BENCHMARK_REGISTRY`

3. Create `experiments/rag/` module
   - `__init__.py`
   - `retriever.py` — FAISS + multilingual-MiniLM retriever
   - `knowledge_base.py` — Vi Wikipedia index builder
   - `rag_pipeline.py` — end-to-end retrieval + prompt augmentation

4. Create `experiments/data/download_vi_benchmarks.py`
   - Download MGSM-vi, ViMMLU; inspect + print sample

5. Create `experiments/evaluation/run_eval_vi.py`
   - 3-condition driver: BF_only / RAG_only / BF+RAG
   - Outputs JSON per (model, benchmark, condition, n_wait)

6. Create `experiments/scripts/run_vi_bf_rag.sh`
   - Sweeps models × benchmarks × conditions

7. Create `experiments/results/summary_vi.py`
   - Aggregates JSONs → summary_vi.csv + summary_vi.md

8. Smoke test
   - Command: `MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' CONDITIONS='BF_only RAG_only BF_RAG' N_WAIT_LIST='0 1 2' N_SAMPLES=5 EXTRA_ARGS='--max_tokens 512 --no_4bit' bash experiments/scripts/run_vi_bf_rag.sh`
   - Goal: JSON files in `experiments/results/vi_*/`, no crash, summary_vi.csv populated

---

## Agent B Checklist (this sprint)

1. Rewrite `report/outline.md` — Vietnamese BF+RAG 8-section structure
2. Draft `report/sections/01_introduction.tex` — Vi NLP motivation, BF+RAG gap
3. Update `report/sections/02_method.tex` — add RAG subsection
4. Update `report/sections/05_related_work.tex` — Vi NLP + RAG-in-education refs
5. Add placeholder tables matching `summary_vi.csv` columns in `06_experiments.tex`

---

## Definition of Done (Sprint 3)

- [ ] All code files exist and pass basic syntax check (`python -c "import ..."`)
- [ ] Smoke run: 1 model × vi_gsm8k × 3 conditions × 5 samples → JSON
- [ ] `summary_vi.csv` generated from smoke JSONs
- [ ] `report/outline.md` rewritten
- [ ] Handoff docs written and committed

---

## Sprint 4 Preview

- Run small matrix (2 models × 2 benchmarks × n_samples=20)
- Fill report tables from summary_vi.csv
- Error analysis + scaling plots
- Final LaTeX compile → PDF
