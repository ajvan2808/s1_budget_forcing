# Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?

This project evaluates **Budget Forcing (BF)** — a decoding-time test-time scaling intervention from the s1 paper — on Vietnamese-language reasoning benchmarks. It is the first study to test whether BF's positive scaling signal transfers to Vietnamese and to Vietnamese-specialized model families.

Based on *s1: Simple Test-Time Scaling* (EMNLP 2025). Repository: `s1_budget_forcing`.

## Research Question

> Does Budget Forcing work for Vietnamese-language reasoning?
> Does the scaling signal transfer across both multilingual and Vietnamese-specialized model families?

## Experimental Design

**Condition:** BF-only — n_wait ∈ {0, 1, 2} where n_wait=0 is the greedy baseline.  
No retrieval. RAG is off-scope for this study.

**Benchmarks:**

| Benchmark | Type | Source |
|-----------|------|---------|
| `vi_gsm8k` (MGSM-vi) | Math reasoning | `juletxara/mgsm`, lang=vi, 250 problems |
| `vimmlu` | Factual multi-domain | `vilm/vimmlu`, ~4000 questions |

**Models:**

| Model | Type | Role |
|-------|------|------|
| `qwen2.5-3B` | Multilingual instruction | Smoke-test baseline |
| `r1-distill-7B` | Reasoning-specialized (Qwen base) | Closest to paper's s1-32B |
| `vinallama-7b` | Vietnamese-specialized (LLaMA-2 base) | BF on Vi-fine-tuned model |
| `vistral-7b` | Vietnamese-specialized (Mistral base) | Second Vi comparison |
| `seallm-7b` | SEA multilingual | Regional baseline |

**Trigger phrase:** `"Chờ một chút"` (Vietnamese; removes language-mixing confound vs. paper's `"Wait"`)

## Pre-registered Hypotheses

| Hypothesis | Prediction | Rationale |
|-----------|------------|-----------|
| H1 | Positive scaling on vi_gsm8k | Multi-step math rewards extended thinking |
| H2 | Weak/no scaling on vimmlu | Factual recall doesn't benefit from longer thinking |
| H3 | Vi-specialized models show different BF response | Not trained with long-CoT; EoT token differs (`</s>`) |

## Project Structure

```text
s1_budget_forcing/
├── PROJECT_BRIEF.md                   ← scope, RQs, risk register
├── docs/
│   ├── brainstorm.md                  ← pivot history + model/benchmark rationale
│   ├── sprint-1/                      ← archived: repo scaffold
│   ├── sprint-2/                      ← archived: Gap 1 cross-family (superseded)
│   └── sprint-3/                      ← active: Vietnamese BF pipeline
│       ├── plan.md
│       ├── agent-a-handoff.md
│       └── agent-b-handoff.md
├── experiments/
│   ├── budget_forcing/
│   │   ├── decoding.py                ← BudgetForcingDecoder (language-agnostic)
│   │   └── metrics.py                 ← control / scaling / performance metrics
│   ├── models/
│   │   └── model_loader.py            ← model registry (multilingual + Vi-specific)
│   ├── evaluation/
│   │   ├── run_eval.py                ← base registry (BenchmarkSpec, answer logic)
│   │   └── run_eval_vi.py             ← BF-only evaluation driver ← main entry point
│   ├── data/
│   │   └── download_vi_benchmarks.py  ← validate HF benchmark access
│   ├── scripts/
│   │   └── run_vi_bf.sh               ← BF-only sweep orchestration
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

**Smoke test** — qwen2.5-3B × vi_gsm8k × n_wait {0,1,2} × 5 samples:

```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf.sh
```

**Small matrix** — 2 models × 2 benchmarks × 20 samples:

```bash
MODELS='qwen2.5-3B r1-distill-7B' BENCHMARKS='vi_gsm8k vimmlu' \
N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
bash experiments/scripts/run_vi_bf.sh
```

**Full matrix** — all 5 models × 2 benchmarks × 100 samples:

```bash
MODELS='qwen2.5-3B r1-distill-7B vinallama-7b vistral-7b seallm-7b' \
BENCHMARKS='vi_gsm8k vimmlu' \
N_WAIT_LIST='0 1 2' N_SAMPLES=100 \
bash experiments/scripts/run_vi_bf.sh
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

## Sprint Status

| Sprint | Status | Focus |
|--------|--------|-------|
| Sprint 1 | done | Repo scaffold, BF baseline |
| Sprint 2 | archived | Gap 1 cross-family (superseded) |
| **Sprint 3** | **active** | Vietnamese BF pipeline — evaluation driver + model registry |
| Sprint 4 | pending | Run experiments, populate report |

**Success criteria (minimum):** smoke run → JSON → `summary_vi.csv` with 3+ rows, no crash.

## References

- Paper: <https://arxiv.org/abs/2501.12599>
- Code: <https://github.com/simplescaling/s1>
- DeepSeek-R1-Distill-Qwen-7B: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>
- MGSM benchmark: <https://huggingface.co/datasets/juletxara/mgsm>
- ViMMLU benchmark: <https://huggingface.co/datasets/vilm/vimmlu>
- Vinallama: <https://huggingface.co/vilm/vinallama-7b-chat>
- Vistral: <https://huggingface.co/vilm/vistral-7b-chat>
- SeaLLM: <https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat>
