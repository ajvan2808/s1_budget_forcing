# Budget Forcing + RAG on Vietnamese Reasoning

This project evaluates **Budget Forcing (BF)** on Vietnamese-language reasoning tasks and compares it head-to-head with **Retrieval-Augmented Generation (RAG)**. It is the first published comparison of test-time-scaling-via-decoding against retrieve-more-context on Vietnamese benchmarks.

Based on *s1: Simple Test-Time Scaling* (EMNLP 2025). Repository: `s1_budget_forcing`.

## Research Question

> Does Budget Forcing work for Vietnamese-language reasoning? Where does thinking longer (BF) beat knowing more (RAG), and can combining them (BF+RAG) produce further gains?

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| `BF_only` | Budget Forcing decoding (`n_wait` ∈ {0,1,2}), no retrieval |
| `RAG_only` | Top-3 Vietnamese Wikipedia passages prepended, greedy decoding |
| `BF+RAG`  | Retrieved context + Budget Forcing (`n_wait` ∈ {1,2}) |

**Benchmarks:** `vi_gsm8k` (MGSM-vi, 250 problems) · `vimmlu` (ViMMLU, multi-domain)

**Models:** `qwen2.5-3B` / `qwen2.5-7B` (multilingual baseline) · `vinallama-7b` · `vistral-7b` · `seallm-7b`

## Project Structure

```text
s1_budget_forcing/
├── PROJECT_BRIEF.md                   ← scope, RQs, risk register
├── docs/
│   ├── brainstorm.md                  ← pivot history + benchmark/model rationale
│   ├── phase-2-4-taskboard.md         ← sprint checklist (Agent A + B)
│   ├── sprint-1/                      ← archived: repo scaffold
│   ├── sprint-2/                      ← archived: Gap 1 cross-family (superseded)
│   └── sprint-3/                      ← active: Vietnamese pipeline
│       ├── plan.md
│       ├── agent-a-handoff.md         ← coder mission
│       └── agent-b-handoff.md         ← writer mission
├── experiments/
│   ├── budget_forcing/
│   │   ├── decoding.py                ← BudgetForcingDecoder (language-agnostic)
│   │   └── metrics.py                 ← control / scaling / performance metrics
│   ├── rag/
│   │   ├── retriever.py               ← FAISS + multilingual-MiniLM retriever
│   │   ├── knowledge_base.py          ← Vietnamese Wikipedia index builder
│   │   └── rag_pipeline.py            ← retrieval + prompt augmentation
│   ├── models/
│   │   └── model_loader.py            ← model registry (multilingual + Vi-specific)
│   ├── evaluation/
│   │   ├── run_eval.py                ← base registry (BenchmarkSpec, answer logic)
│   │   └── run_eval_vi.py             ← 3-condition evaluation driver ← main entry point
│   ├── data/
│   │   └── download_vi_benchmarks.py  ← validate HF benchmark access
│   ├── scripts/
│   │   └── run_vi_bf_rag.sh           ← sweep orchestration
│   └── results/
│       └── summary_vi.py              ← JSON → summary_vi.csv + summary_vi.md
└── report/
    ├── outline.md
    ├── main.tex
    ├── references.bib
    └── sections/
        ├── 01_introduction.tex
        ├── 02_method.tex
        ├── 03_metrics.tex
        ├── 04_analysis.tex
        ├── 05_related_work.tex
        ├── 06_experiments.tex
        └── 07_conclusion.tex
```

## Quickstart

Install dependencies:

```bash
uv sync
```

Validate Vietnamese benchmarks are accessible:

```bash
uv run python experiments/data/download_vi_benchmarks.py
```

Build a small RAG index (smoke-friendly, ~10k articles):

```bash
uv run python experiments/rag/knowledge_base.py \
    --output_dir experiments/data/vi_wiki_index_smoke \
    --max_docs 10000
```

**Smoke test** — 1 model × vi_gsm8k × 3 conditions × 5 samples:

```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf_rag.sh
```

**Small matrix** — 2 models × 2 benchmarks × 20 samples:

```bash
MODELS='qwen2.5-3B vinallama-7b' BENCHMARKS='vi_gsm8k vimmlu' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
bash experiments/scripts/run_vi_bf_rag.sh
```

Aggregate results:

```bash
uv run python experiments/results/summary_vi.py \
    --results_dir experiments/results/vi_YYYYMMDD_HHMMSS
```

## Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Scaling** | Slope of accuracy vs `n_wait` — positive = thinking longer helps |
| **Performance** | Max accuracy across `n_wait` values |
| **Control** | % of runs reaching target thinking-token budget (approximated by `n_wait`) |
| **Extraction failure rate** | % of outputs where answer parser found no match |

Comparison axis: condition (`BF_only` / `RAG_only` / `BF+RAG`) × benchmark × model.

## Pre-registered Hypotheses

| Hypothesis | Prediction | Reasoning |
|-----------|------------|-----------|
| H1 | BF > RAG on vi_gsm8k | Multi-step math; retrieved text rarely contains calculation steps |
| H2 | RAG ≥ BF on vimmlu | Factual recall; Wikipedia directly resolves knowledge gaps |
| H3 | BF+RAG ≥ max(BF, RAG) | Combined benefit when context window not saturated |

## Sprint Status

| Sprint | Status | Focus |
|--------|--------|-------|
| Sprint 1 | done | Repo scaffold, BF baseline |
| Sprint 2 | archived | Gap 1 cross-family (superseded) |
| **Sprint 3** | **active** | Vietnamese pipeline — RAG module + Vi benchmarks + 3-condition eval |
| Sprint 4 | pending | Run experiments, populate report |

**Success criteria (minimum):** smoke run → JSON → `summary_vi.csv` with 3+ rows, no crash.

## References

- Paper: <https://arxiv.org/abs/2501.12599>
- Code: <https://github.com/simplescaling/s1>
- MGSM benchmark: <https://huggingface.co/datasets/juletxara/mgsm>
- ViMMLU benchmark: <https://huggingface.co/datasets/vilm/vimmlu>
