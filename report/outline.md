# Report Outline: Vietnamese BF Transfer Study

Updated: 2026-06-12 (Pivot from BF+RAG to BF-only transfer study)

## Working Title

**Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?**

## Central Claim to Test

Budget Forcing (BF) — a decoding-time intervention that suppresses end-of-thought tokens to
extend chain-of-thought reasoning — was originally validated on English benchmarks with the
Qwen model family. This report is the first study to evaluate whether BF's positive test-time
scaling signal transfers to (1) Vietnamese-language benchmarks and (2) Vietnamese-specialized
model families that were not trained with long chain-of-thought data.

---

## 1. Introduction

- Vietnamese as an underrepresented language in reasoning benchmarks and test-time scaling research.
- Budget Forcing: simple, inference-only intervention from s1 paper — no fine-tuning required.
- Research gap: original BF study was English-only on Qwen family. Transfer to other languages and model families untested.
- Contributions:
  1. First BF evaluation on Vietnamese reasoning benchmarks (MGSM-vi, ViMMLU)
  2. First cross-family BF study including Vietnamese-specialized models (Vinallama, Vistral, SeaLLM)
  3. Pre-registered hypotheses on scaling behavior by benchmark type and model family

## 2. Background

### 2.1 Budget Forcing (s1 paper)

- s1K data curation; chain-of-thought distillation.
- BF mechanics: enforce minimum (suppress EoT + append `Wait`), enforce maximum (inject `Final Answer:`).
- Positive scaling signal in the original study (Qwen family, English benchmarks).
- Reported limitations: model-family and language coverage gaps.

### 2.2 Retrieval-Augmented Generation

- RAG overview: Lewis et al. 2020 foundational paper.
- Dense retrieval (DPR, FAISS) vs sparse BM25.
- RAG for reasoning: when external context helps vs hurts.
- Multilingual dense retrieval: multilingual-MiniLM, mDPR.

### 2.3 Vietnamese Language Models and Benchmarks

- Vinallama, Vistral, SeaLLM — architecture and training overview.
- Qwen2.5 as multilingual baseline.
- MGSM-vi (Shi et al., 2023): parallel GSM8K in 11 languages.
- ViMMLU: Vietnamese MMLU; multi-domain knowledge.

## 3. Metrics

- **Control:** % of runs that reach a target thinking-token budget (approximate via n_wait).
- **Scaling:** slope of accuracy vs n_wait curve. Positive = thinking longer helps.
- **Performance:** max accuracy across n_wait values.
- **Extraction failure rate:** % of outputs where answer parser found no match.
- Comparison axis: model family × benchmark × n_wait.

## 4. Method

### 4.1 Budget Forcing Decoding

- BudgetForcingDecoder implementation (from s1 codebase, language-agnostic).
- EoT suppression tokens: `["<|im_end|>", "</think>", "####", "\n\nFinal Answer:", "\n\nAnswer:", "</s>"]`.
  - `</s>` added for LLaMA/Mistral-based Vi models (Vinallama, Vistral).
- n_wait sweep: {0, 1, 2} where n_wait=0 is greedy baseline.
- Trigger phrase: `"Chờ một chút"` (Vietnamese; removes language-mixing confound vs. paper's `"Wait"`).

### 4.2 Experimental Design

| Variable | Values |
|----------|--------|
| Models | qwen2.5-3B, r1-distill-7B, vinallama-7b, vistral-7b, seallm-7b |
| Benchmarks | vi_gsm8k (MGSM-vi), vimmlu |
| n_wait | {0, 1, 2} |
| Total runs | 5 models × 2 benchmarks × 3 n_wait = 30 |

### 4.3 Hypotheses (pre-registered before results)

- **H1 (Positive scaling on vi_gsm8k):** Multi-step math rewards extended thinking. BF produces positive scaling slope on reasoning-specialized models (r1-distill-7B, qwen2.5-3B).
- **H2 (Weak/no scaling on vimmlu):** Factual recall tasks don't benefit from longer thinking; scaling slope near zero or negative.
- **H3 (Vi-specialized ≠ reasoning-specialized):** Vinallama/Vistral (no long-CoT training) show different, likely weaker, BF response than r1-distill-7B.

## 5. Related Work

### 5.1 Test-Time Scaling

- s1: Simple Test-Time Scaling (EMNLP 2025).
- DeepSeek-R1 and R1-Distill.
- Budget-aware decoding, adaptive compute.
- Cross-family generalizability of BF (Yong et al., 2025 — partially addresses Gap 1).

### 5.2 Vietnamese NLP

- PhoGPT, Vinallama, Vistral, SeaLLM — model family survey.
- ViMMLU benchmark paper.
- Multilingual reasoning: MGSM (Shi et al., 2023).

### 5.3 RAG for Reasoning

- RAG foundational (Lewis et al., 2020).
- Distractor vs helpful context in reasoning (Shi et al., 2023 — irrelevant context hurts).
- RAG + chain-of-thought: He et al., 2022 (iterative retrieval).
- Vietnamese educational AI context.

## 6. Experiments

### 6.1 Setup

- Models: qwen2.5-3B (smoke), r1-distill-7B, vinallama-7b, vistral-7b, seallm-7b.
- Benchmarks: vi_gsm8k (250 problems), vimmlu (~4000 questions).
- Compute: 4-bit NF4 quantization (BitsAndBytes); Kaggle T4 GPU.
- n_samples: 5 (smoke), 20 (small matrix), 100+ (full run).
- Hardware constraints: logged in `experiments/results/BLOCKERS.md`.

### 6.2 Main Results Table

<!-- Populated in Sprint 4 from summary_vi.csv. DO NOT fabricate. -->

| Model | Benchmark | n_wait=0 | n_wait=1 | n_wait=2 | Scaling | Performance |
|-------|-----------|----------|----------|----------|---------|-------------|
| qwen2.5-3B | vi_gsm8k | -- | -- | -- | -- | -- |
| r1-distill-7B | vi_gsm8k | -- | -- | -- | -- | -- |
| vinallama-7b | vi_gsm8k | -- | -- | -- | -- | -- |
| vistral-7b | vi_gsm8k | -- | -- | -- | -- | -- |
| seallm-7b | vi_gsm8k | -- | -- | -- | -- | -- |
| qwen2.5-3B | vimmlu | -- | -- | -- | -- | -- |
| ... | ... | ... | ... | ... | ... | ... |

### 6.3 Scaling Curves by Model Family

### 6.4 Scaling Curves

<!-- Accuracy vs n_wait per condition per model — line plots. -->

### 6.5 Error Analysis

- Parser failure rate by model and language.
- Failure taxonomy: (a) answer extraction fails, (b) reasoning error despite sufficient thinking, (c) context overflow / irrelevant retrieval.
- Vietnamese-specific extraction patterns.

## 7. Discussion

- **BF on Vietnamese:** Does extended thinking produce positive scaling on vi_gsm8k? Is the scaling slope consistent across n_wait values?
- **Benchmark type effect:** Does BF help more on math reasoning (vi_gsm8k) than factual recall (vimmlu)? Does this match H1/H2?
- **Model family effects:** Do Vi-specialized models (Vinallama/Vistral — no long-CoT training) show different BF response than reasoning-specialized r1-distill-7B?
- **Failure modes:** Parser failure rate by model; EoT token mismatch for LLaMA-based models; repetitive loop detection.
- **Generalizability claim:** Under what conditions can we claim BF "transfers" to Vietnamese?

## 8. Conclusion

- Answer primary RQ: does BF work for Vietnamese reasoning?
- Answer secondary RQs (RQ1–RQ4 from PROJECT_BRIEF.md §2).
- Limitations: small sample sizes; limited model coverage; trigger phrase in English.
- Future work: Vietnamese-trigger BF ("Đợi đã"), SFT on s1K-vi, harder benchmarks (ZaloAI Math), human evaluation of reasoning trace quality.

---

## References (to fill)

- `\cite{TODO:s1}` — s1: Simple Test-Time Scaling
- `\cite{TODO:deepseek-r1}` — DeepSeek-R1
- `\cite{TODO:mgsm}` — Shi et al., 2023: MGSM multilingual benchmark
- `\cite{TODO:vimmlu}` — ViMMLU benchmark paper
- `\cite{TODO:lewis2020rag}` — Lewis et al., 2020: RAG
- `\cite{TODO:vinallama}` — Vinallama model card / paper
- `\cite{TODO:vistral}` — Vistral model card / paper
- `\cite{TODO:seallm}` — SeaLLM model paper
- `\cite{TODO:yong2025}` — Yong et al., 2025: cross-family BF generalizability

---

## Artifact Traceability

All numbers in §6 must trace to a file in `experiments/results/vi_*/`:
- Accuracy numbers ← `summary_vi.csv` rows
- Error analysis ← `*.json` details arrays
- Scaling values ← `summary_vi.csv` `scaling` column

Do not write numbers into the report that are not present in the CSV.
