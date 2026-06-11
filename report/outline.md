# Report Outline: Vietnamese BF+RAG Study

Updated: 2026-06-11 (Pivot from Gap 1 to Vietnamese BF+RAG)

## Working Title

**Budget Forcing on Vietnamese Reasoning: Test-Time Scaling vs. Retrieval Augmentation**

## Central Claim to Test

On Vietnamese-language reasoning tasks, two complementary strategies can improve LLM outputs:
(1) **Budget Forcing (BF)** — think longer via decoding-time extension; and
(2) **RAG** — know more via retrieved Vietnamese Wikipedia context.
This report is the first published head-to-head comparison of these two strategies on
Vietnamese benchmarks, and tests whether combining them (BF+RAG) produces further gains.

---

## 1. Introduction

- Vietnamese as an underrepresented language in reasoning benchmarks.
- Two complementary scaling strategies: BF (thinking longer) vs RAG (knowing more).
- Research gap: no prior BF study on Vietnamese; no BF vs RAG comparison on any low-resource language.
- Connection to existing `adaptive-rag-hotpotqa` work (RAG module reuse).
- Contributions:
  1. First BF evaluation on Vietnamese reasoning (MGSM-vi, ViMMLU)
  2. First direct BF vs RAG head-to-head on Vietnamese tasks
  3. Combined BF+RAG evaluation under controlled conditions

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
- Comparison axis: condition (BF_only / RAG_only / BF+RAG) × benchmark × model family.

## 4. Method

### 4.1 Budget Forcing Decoding

- BudgetForcingDecoder implementation (from s1 codebase, language-agnostic).
- EoT suppression tokens for Vietnamese models.
- n_wait sweep: {0, 1, 2}.
- Trigger phrase: "Wait" (English baseline; "Đợi đã" deferred to future work).

### 4.2 RAG Pipeline

- Knowledge base: Vietnamese Wikipedia (wikimedia/wikipedia, 20231101.vi).
- Embedding: `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, FAISS FlatL2.
- Retrieval: top-3 passages, ≤128 tokens each.
- Augmentation template:
  ```
  [Ngữ cảnh tham khảo]
  (1) {passage_1}
  (2) {passage_2}
  (3) {passage_3}

  Câu hỏi: {question}
  ```

### 4.3 Experimental Conditions

| Condition | BF | RAG | n_wait |
|-----------|----|----|--------|
| BF_only   | ✓  | ✗  | {0, 1, 2} |
| RAG_only  | ✗  | ✓  | 0 |
| BF+RAG    | ✓  | ✓  | {1, 2} |

### 4.4 Hypotheses (pre-registered before results)

- **H1 (BF > RAG on math):** MGSM-vi requires multi-step reasoning; retrieved passages rarely contain calculation steps. Extended thinking helps more than context.
- **H2 (RAG ≥ BF on knowledge):** ViMMLU factual questions are directly resolvable with retrieved Wikipedia text; extra thinking on incorrect prior knowledge does not help.
- **H3 (BF+RAG ≥ max):** Combined condition benefits from both — but only when context window is not saturated and retrieved passages are relevant.

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

- Models: Qwen2.5-3B (smoke), Qwen2.5-7B, Vinallama-7b, Vistral-7b, SeaLLM-7b.
- Benchmarks: vi_gsm8k (250 problems), vimmlu (~4000 questions).
- Compute: 4-bit NF4 quantization (BitsAndBytes); CPU fallback for Qwen2.5-3B.
- n_samples: 5 (smoke), 20 (small matrix), 100+ (full run).
- Hardware constraints: logged in `experiments/results/BLOCKERS.md`.

### 6.2 Main Results Table

<!-- Populated in Sprint 4 from summary_vi.csv. DO NOT fabricate. -->

| Model | Benchmark | Condition | n_wait=0 | n_wait=1 | n_wait=2 | Scaling | Performance |
|-------|-----------|-----------|----------|----------|----------|---------|-------------|
| qwen2.5-3B | vi_gsm8k | BF_only | -- | -- | -- | -- | -- |
| qwen2.5-3B | vi_gsm8k | RAG_only | -- | -- | -- | -- | -- |
| qwen2.5-3B | vi_gsm8k | BF+RAG  | -- | -- | -- | -- | -- |
| qwen2.5-7B | vi_gsm8k | BF_only | -- | -- | -- | -- | -- |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 6.3 BF vs RAG Comparison (per benchmark)

<!-- Add bar/violin plots from summary_vi.csv when artifacts exist. -->

### 6.4 Scaling Curves

<!-- Accuracy vs n_wait per condition per model — line plots. -->

### 6.5 Error Analysis

- Parser failure rate by model and language.
- Failure taxonomy: (a) answer extraction fails, (b) reasoning error despite sufficient thinking, (c) context overflow / irrelevant retrieval.
- Vietnamese-specific extraction patterns.

## 7. Discussion

- **BF on Vietnamese:** Does extended thinking produce positive scaling on Vietnamese math? Is it consistent across model families?
- **RAG on Vietnamese:** Does Wikipedia retrieval reliably improve factual accuracy? Are retrieved passages in Vietnamese consistently relevant?
- **BF vs RAG:** Which strategy wins by benchmark type? Does result match H1/H2?
- **BF+RAG combined:** Does the combination beat either alone? What limits it (context overflow, irrelevant retrieval, reasoning interference)?
- **Model family effects:** Do Vinallama/Vistral (Vi-specialized) benefit more from BF than Qwen (multilingual)?

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
