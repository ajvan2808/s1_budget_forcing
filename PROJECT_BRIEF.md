# Project Brief

Updated: 2026-06-11 (Pivot to Vietnamese BF+RAG)

## 1. Project Overview

- **Project:** Budget Forcing + RAG on Vietnamese Reasoning
- **Goal:** Evaluate whether Budget Forcing (BF) improves reasoning on Vietnamese-language tasks, whether Retrieval-Augmented Generation (RAG) improves it, and — whether **BF+RAG combined outperforms either alone**.
- **Research Gap:** No published study compares test-time scaling via decoding (BF) against retrieve-more-context (RAG) on Vietnamese. Novel, UIT-relevant, and connects to existing `adaptive-rag-hotpotqa` work.
- **Repository:** `s1_budget_forcing`
- **Deliverable:** reproducible experiment artifacts + paper-style report

## 2. Research Questions

**Primary RQ:**
> Does Budget Forcing work for Vietnamese-language reasoning tasks? Where does test-time-scaling-via-decoding (BF) beat retrieve-more-context (RAG), and can combining them (BF+RAG) produce further gains?

**Secondary RQs:**
- **RQ1:** Does BF produce positive scaling on Vietnamese math (vi-GSM8K, ZaloAI Math)?
- **RQ2:** Does BF produce positive scaling on Vietnamese knowledge tasks (ViMMLU)?
- **RQ3:** Does RAG-augmented context reduce BF steps needed to reach peak accuracy?
- **RQ4:** Are there task types where RAG dominates BF, and vice versa?

## 3. Experimental Conditions

**Three conditions per question:**

| Condition | Description |
|-----------|-------------|
| `BF_only` | Budget Forcing decoding (n_wait ∈ {0,1,2}), no external retrieval |
| `RAG_only` | Retrieve top-k Vietnamese passages, augment prompt, no BF (n_wait=0) |
| `BF+RAG` | Retrieve + augment prompt, then apply BF (n_wait ∈ {1,2}) |

**Benchmarks (Vietnamese):**
- `vi_gsm8k` — MGSM Vietnamese split (`juletxara/mgsm`, lang=`vi`), 250 problems
- `vimmlu` — Vietnamese MMLU subset (`vilm/vimmlu`), multi-domain
- `zaloai_math` — ZaloAI Math 2023 (curated locally)

**Models:**
- `qwen2.5-3B` / `qwen2.5-7B` — multilingual baseline (already in registry)
- `vinallama-7b` — Vietnamese LLaMA (`vilm/vinallama-7b-chat`)
- `vistral-7b` — Vietnamese Mistral (`vilm/vistral-7b-chat`)
- `seallm-7b` — SEA multilingual (`SeaLLMs/SeaLLMs-v3-7B-Chat`)

**RAG Knowledge Base:**
- Source: Vietnamese Wikipedia (`wikimedia/wikipedia`, `20231101.vi`)
- Index: FAISS + `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Top-k: 3 passages, each ≤128 tokens

## 4. Code Reuse Map (~60% from Gap 1)

| Module | Reuse | Change |
|--------|-------|--------|
| `budget_forcing/decoding.py` | ✅ 100% | Language-agnostic, unchanged |
| `budget_forcing/metrics.py` | ✅ 100% | Unchanged |
| `models/model_loader.py` | ✅ ~70% | Add Vietnamese model entries |
| `evaluation/run_eval.py` | ✅ registry only | Add `vi_gsm8k`, `vimmlu` benchmark specs |
| `evaluation/run_eval_vi.py` | 🆕 new | 3-condition evaluation driver |
| `rag/` module | 🆕 new | retriever + KB builder + pipeline |
| `data/download_vi_benchmarks.py` | 🆕 new | Download & inspect Vi benchmarks |
| `scripts/run_vi_bf_rag.sh` | 🆕 new | 3-condition sweep script |
| `results/summary_vi.py` | 🆕 new | 3-condition aggregator |

## 5. Architecture

```text
Vietnamese Benchmarks              Vietnamese Knowledge Base
(vi-GSM8K / ViMMLU / ZaloAI)      (Vi Wikipedia → FAISS index)
          |                                    |
          v                                    v
  evaluation/run_eval_vi.py  ←──  rag/rag_pipeline.py
          |
          ├─ BF_only  → BudgetForcingDecoder(n_wait>0),  no RAG
          ├─ RAG_only → RAG-augmented prompt,             n_wait=0
          └─ BF+RAG   → RAG-augmented prompt + BudgetForcingDecoder(n_wait>0)
          |
          v
  results/vi_*/  ← JSON per (model, benchmark, condition, n_wait)
          |
          v
  results/summary_vi.py → summary_vi.csv + summary_vi.md → report tables
```

## 6. Canonical Run Commands

**Smoke test:**
```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf_rag.sh
```

**Small matrix (2 models × 2 benchmarks):**
```bash
MODELS='qwen2.5-3B vinallama-7b' BENCHMARKS='vi_gsm8k vimmlu' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
bash experiments/scripts/run_vi_bf_rag.sh
```

**Aggregation:**
```bash
python experiments/results/summary_vi.py \
  --results_dir experiments/results/vi_YYYYMMDD_HHMMSS
```

## 7. Team Roles

| Agent | Role | Deliverable |
|-------|------|-------------|
| **Agent A** | Coder / Experiment owner | 3-condition pipeline + JSON artifacts |
| **Agent B** | Writer / Report owner | LaTeX report, Vietnamese BF+RAG framing |

## 8. Sprint Structure

| Sprint | Focus |
|--------|-------|
| Sprint 1 (done) | Repo scaffold, initial BF baseline |
| Sprint 2 (archived) | Gap 1 cross-family (superseded by pivot) |
| **Sprint 3 (active)** | Vietnamese pivot: RAG module + Vi benchmarks + 3-condition eval |
| Sprint 4 | Run experiments, collect artifacts, populate report |

## 9. Success Criteria

**Minimum:** smoke run 1 model × vi-GSM8K × 3 conditions × 5 samples → JSON + summary CSV

**Good:** 2 models × 2 benchmarks × 3 conditions, n_samples ≥ 20, BF vs RAG table in report

**Strong:** 3+ models × 3 benchmarks, error analysis (parser vs reasoning failures in Vietnamese), scaling curves

## 10. Key Risks

| Risk | Mitigation |
|------|------------|
| Vi benchmark not on HF | MGSM-vi confirmed; ZaloAI curated locally as fallback |
| Vi model slow on macOS | Qwen2.5 (multilingual) as primary; Vi-specific models secondary |
| RAG index build time | Pre-build once → `data/vi_wiki_index/`; skip if unavailable |
| Vi answer extraction fails | Vi-specific parser + log extraction failures separately |
| BF+RAG prompt overflow | Cap retrieved text at 2×128 tokens |
