# Project Brief

Updated: 2026-06-12 (BF-only Focus)

## 1. Project Overview

- **Project:** Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?
- **Goal:** Evaluate whether Budget Forcing (BF) — a test-time scaling via decoding intervention — transfers to Vietnamese-language reasoning tasks and across multilingual + Vietnamese-specialized model families.
- **Research Gap:** No published study tests whether BF's positive scaling signal generalizes to Vietnamese. First evaluation of BF on Vietnamese-specialized models (VinAllama, ViStral). Novel, UIT-relevant, and complements existing s1 paper work.
- **Repository:** `s1_budget_forcing`
- **Deliverable:** reproducible experiment artifacts + paper-style report

## 2. Research Questions

**Primary RQ:**
> Does Budget Forcing work for Vietnamese-language reasoning tasks?  
> Does the scaling signal transfer across both multilingual and Vietnamese-specialized model families?

**Secondary RQs:**

- **RQ1:** Does BF produce positive scaling on Vietnamese math (vi-GSM8K)?
- **RQ2:** Does BF produce weak/no scaling on Vietnamese factual tasks (ViMMLU)?
- **RQ3:** Do Vietnamese-specialized models show different BF responses vs. multilingual models?
- **RQ4:** What is the interaction between model family (LLaMA, Mistral, Qwen, SEA) and BF effectiveness?

## 3. Experimental Design

**Single condition: BF-only** (no retrieval; RAG is off-scope)

| Component | Details |
|-----------|---------|
| **Decoding** | Budget Forcing with n_wait ∈ {0, 1, 2}; n_wait=0 is greedy baseline |
| **Trigger phrase** | `"Chờ một chút"` (Vietnamese; avoids language-mixing confound) |
| **Benchmarks** | vi_gsm8k (math), vimmlu (factual) — 2 task types |

**Benchmarks (Vietnamese):**

- `vi_gsm8k` — MGSM Vietnamese split (`juletxara/mgsm`, lang=`vi`), 250 problems, math reasoning
- `vimmlu` — Vietnamese MMLU subset (`vilm/vimmlu`), ~4000 questions, factual multi-domain

**Models (5 total):**

- `qwen2.5-3B` — Multilingual instruction baseline (smoke test)
- `r1-distill-7B` — Reasoning-specialized distill (Qwen base); closest to s1-32B in spirit
- `vinallama-7b` — Vietnamese-specialized LLaMA (`vilm/vinallama-7b-chat`)
- `vistral-7b` — Vietnamese-specialized Mistral (`vilm/vistral-7b-chat`)
- `seallm-7b` — SEA multilingual (`SeaLLMs/SeaLLMs-v3-7B-Chat`)

## 4. Code Implementation Status

| Module | Status | Details |
|--------|--------|---------|
| `budget_forcing/decoding.py` | ✅ Reused | Language-agnostic BudgetForcingDecoder |
| `budget_forcing/metrics.py` | ✅ Reused | Scaling/performance/control metrics |
| `models/model_loader.py` | ✅ Extended | Added Vietnamese models + r1-distill-7B |
| `evaluation/run_eval.py` | ✅ Extended | Added vi_gsm8k, vimmlu benchmark specs to registry |
| `evaluation/run_eval_vi.py` | ✅ Complete | BF-only evaluation driver (primary entry point) |
| `data/download_vi_benchmarks.py` | ✅ Complete | Benchmark validation script |
| `scripts/run_vi_bf.sh` | ✅ Complete | BF-only sweep orchestration (simplified from original 3-condition plan) |
| `results/summary_vi.py` | ✅ Complete | JSON → summary_vi.csv + summary_vi.md aggregator |

## 5. Architecture

```text
Vietnamese Benchmarks              Budget Forcing Decoder
(vi-GSM8K / ViMMLU)               (n_wait sweep)
          |                              |
          └──────────────┬───────────────┘
                         v
           evaluation/run_eval_vi.py
                  (BF-only driver)
                         |
                         v
          results/vi_*/  ← JSON per (model, benchmark, n_wait)
                         |
                         v
   results/summary_vi.py → summary_vi.csv + summary_vi.md

[RAG modules available but off-scope for current study]
```

**Data Flow (BF-only):**

1. Load model (multilingual or Vietnamese-specialized)
2. Load benchmark (vi_gsm8k or vimmlu)
3. For each sample and each n_wait ∈ {0, 1, 2}:
   - Generate response using BudgetForcingDecoder
   - Extract answer using language-appropriate parser
   - Evaluate correctness
4. Output JSON with thinking_tokens, extracted_answer, correct, n_wait
5. Aggregate across all runs → summary_vi.csv (model × benchmark × n_wait table)

## 6. Canonical Run Commands

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

**Full matrix** — 5 models × 2 benchmarks × 100 samples:

```bash
MODELS='qwen2.5-3B r1-distill-7B vinallama-7b vistral-7b seallm-7b' \
BENCHMARKS='vi_gsm8k vimmlu' \
N_WAIT_LIST='0 1 2' N_SAMPLES=100 \
bash experiments/scripts/run_vi_bf.sh
```

**Aggregation** — convert JSON results to CSV + markdown summary:

```bash
python experiments/results/summary_vi.py \
  --results_dir experiments/results/vi_YYYYMMDD_HHMMSS
```

## 7. Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Scaling** | Slope of accuracy vs n_wait — positive = longer thinking helps |
| **Performance** | Max accuracy across n_wait ∈ {0, 1, 2} |
| **Control** | % of runs reaching target thinking-token budget (approximated by n_wait) |
| **Extraction failure rate** | % of outputs where answer parser found no match (language-specific issue) |

## 8. Hypotheses

| Hypothesis | Prediction | Rationale |
|-----------|------------|-----------|
| **H1** | Positive scaling on vi_gsm8k | Multi-step math rewards extended thinking |
| **H2** | Weak/no scaling on vimmlu | Factual recall doesn't benefit from longer thinking |
| **H3** | Vi-specialized models show different BF response | Not trained with long-CoT; EoT token differs (`</s>` vs model-specific) |
| **H4** | Multilingual models generalize better than Vi-specific | Broader training data despite domain specificity loss |

## 9. Team Roles

| Agent | Role | Deliverable |
|-------|------|-------------|
| **Agent A** | Coder / Experiment owner | BF-only pipeline + JSON artifacts |
| **Agent B** | Writer / Report owner | LaTeX report, Vietnamese BF framing |

## 10. Sprint Status

| Sprint | Status | Focus |
|--------|--------|-------|
| Sprint 1 | done | Repo scaffold, initial BF baseline (Gap 1) |
| Sprint 2 | archived | Gap 1 cross-family (superseded by Vietnamese pivot) |
| **Sprint 3** | **active** | Vietnamese BF pipeline — evaluation driver + model registry + benchmark specs |
| Sprint 4 | pending | Run full experiments, populate report, error analysis, scaling curves |

## 11. Success Criteria

**Minimum (MVP):**

- Smoke run: 1 model × vi-GSM8K × n_wait {0, 1, 2} × 5 samples → JSON + summary_vi.csv

**Good:**

- 2 models × 2 benchmarks × n_samples ≥ 20
- BF scaling table in report
- Accuracy vs n_wait plot for at least one model-benchmark pair

**Strong:**

- All 5 models × 2 benchmarks × n_samples ≥ 100
- Error analysis (parser failure rate vs reasoning failures in Vietnamese)
- Scaling curves per model family (multilingual vs Vietnamese-specialized)
- Hypothesis validation for H1–H4

## 12. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Vietnamese benchmarks unavailable on HF | High | MGSM-vi confirmed; ViMMLU confirmed; pre-validate via download_vi_benchmarks.py |
| Vietnamese models slow on consumer GPU/CPU | Medium | Qwen2.5-3B as smoke-test baseline; 4-bit quantization for 7B models |
| Vietnamese answer extraction fails | Medium | Implement Vietnamese-aware parser; log extraction failures separately by language |
| Model parameter differences (EoT token) | Low | Documented in model_loader.py; tested in smoke run |
| Report write-up delayed | Low | Agent B writes from literature + hypotheses; results fill placeholders in Sprint 4 |
