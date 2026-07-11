# Report Outline: s1 Analysis + Vietnamese Budget Forcing Transfer Study

Updated: 2026-06-20

## Working Title

**Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?**

## Report Purpose

This report has two main objectives:

1. **Introduce and analyze the source paper**: explain the key ideas of *s1: Simple Test-Time Scaling*, including s1K data curation, Budget Forcing, test-time scaling metrics, reported results, and limitations.
2. **Conduct a Vietnamese transfer experiment**: evaluate whether the Budget Forcing mechanism from s1 transfers to Vietnamese-language reasoning benchmarks and Vietnamese/open-source model families.

The report should not read like a pure implementation report. It should first teach the reader what the s1 paper does and why it matters, then present our experiment as a targeted extension of one central s1 mechanism: Budget Forcing.

---

## Abstract

Write the abstract in two halves:

- First half: summarize the s1 paper and its recipe: a small curated reasoning dataset (s1K), supervised fine-tuning, and Budget Forcing as a decoding-time method for controlling thinking length.
- Second half: summarize our extension: adapting Budget Forcing to Vietnamese prompts, Vietnamese answer formats, and a model/benchmark matrix covering multilingual, reasoning-distilled, Vietnamese-specialized, and Vietnamese-native reasoning models.

Do **not** claim final accuracy improvements until full-run results are available. Use cautious wording such as "we evaluate", "we test", and "preliminary results suggest" when necessary.

---

## 1. Introduction

### 1.1 Motivation: Test-Time Scaling

- Explain test-time scaling as a new paradigm: improving reasoning by spending more inference-time compute rather than only increasing model size or training data.
- Contrast train-time scaling with inference-time/test-time scaling.
- Mention that o1-style reasoning models made this direction important, but their methodology is not fully open.

### 1.2 Why s1 Matters

- Introduce *s1: Simple Test-Time Scaling* as an open-source attempt to find the simplest recipe for test-time scaling.
- Emphasize the two central components:
  1. s1K: 1,000 carefully curated reasoning examples.
  2. Budget Forcing: a decoding-time intervention that lengthens or shortens reasoning traces.
- State that this report studies the source paper first, then extends its Budget Forcing component to Vietnamese.

### 1.3 Research Gap

- s1 evaluates mainly English reasoning benchmarks and the Qwen/s1 model family.
- It is not clear whether Budget Forcing transfers across:
  - language: English to Vietnamese
  - benchmark type: math, factual, applied exam questions
  - model family: multilingual, English-distilled reasoning, Vietnamese-specialized, Vietnamese-native reasoning

### 1.4 Objectives and Contributions

State the two objectives explicitly:

1. Analyze the s1 paper: its motivation, method, metrics, findings, and limitations.
2. Evaluate Budget Forcing on Vietnamese models and benchmarks.

Contributions:

- We provide a structured analysis of the s1 paper, focusing on why Budget Forcing can create sequential test-time scaling.
- We adapt Budget Forcing to Vietnamese by using the trigger phrase `"Chờ một chút"`, adding end-of-thinking delimiters for Vietnamese-specialized and Vietnamese-native reasoning models, and handling Vietnamese benchmark answer formats.
- We evaluate a 2x2 model-family matrix over Vietnamese reasoning benchmarks.
- We analyze whether increased thinking tokens translate into improved accuracy, separating mechanical controllability from useful reasoning transfer.

### 1.5 Report Organization

- Section 2 analyzes the s1 source paper.
- Section 3 derives the Vietnamese transfer research question and hypotheses.
- Section 4 describes our experimental design.
- Section 5 reports results.
- Section 6 discusses the findings in relation to s1.
- Section 7 gives limitations and future work.
- Section 8 concludes.

---

## 2. Source Paper Analysis: s1 Simple Test-Time Scaling

This section should be a real analytical section, not only related work.

### 2.1 Problem Setting in s1

- Define test-time scaling.
- Explain why the field wanted an open alternative to opaque o1-style reasoning systems.
- State s1's guiding question: what is the simplest approach to achieve test-time scaling and strong reasoning performance?

### 2.2 The s1 Recipe

Explain s1 as a combination of:

- small reasoning-data curation (`s1K`)
- supervised fine-tuning on a Qwen2.5-32B-Instruct base model
- Budget Forcing at inference time

Clarify that our project transfers/evaluates the Budget Forcing component, not the full s1 training pipeline.

### 2.3 s1K Data Curation

Explain the 59K-to-1K selection process:

- quality filtering
- difficulty filtering using model performance and reasoning-trace length
- diversity filtering using domain labels
- distillation of reasoning traces from Gemini Thinking

Key analytical point: s1 argues that careful selection can outperform simply using more data.

### 2.4 Budget Forcing Method

Explain both directions:

- **Enforce maximum**: end thinking and force final answer after a budget.
- **Enforce minimum**: suppress end-of-thinking and append `"Wait"` when the model tries to stop too early.

Explain why this can produce self-correction: the model is forced to continue reasoning after a premature stopping point.

### 2.5 Test-Time Scaling Metrics in s1

Define:

- **Control**: how well the method reaches a desired thinking-token budget.
- **Scaling**: slope of accuracy as thinking budget increases.
- **Performance**: best accuracy achieved across budgets.

These metrics motivate the metrics used in our Vietnamese experiments.

### 2.6 Main Findings of s1

Summarize:

- s1-32B is sample-efficient compared with other open reasoning models.
- Budget Forcing gives positive test-time scaling on reasoning-intensive benchmarks such as AIME24, MATH500, and GPQA Diamond.
- Sequential scaling via Budget Forcing is compared against other test-time scaling methods.
- Scaling eventually flattens or degrades due to loops/context limits.

### 2.7 Limitations and Open Questions from s1

Use this subsection as the bridge to our work:

- The original evaluation is mostly English.
- The main model family is Qwen/s1.
- It is unclear whether Budget Forcing works on Vietnamese text.
- It is unclear whether the method helps Vietnamese-specialized models that were not trained with long reasoning traces.
- It is unclear whether factual or exam-style multiple-choice tasks benefit from longer thinking.

---

## 3. Research Gap and Vietnamese Transfer Study

### 3.1 Main Research Question

> Does Budget Forcing transfer to Vietnamese-language reasoning tasks and Vietnamese-specialized model families?

### 3.2 Subquestions

- Does Budget Forcing improve Vietnamese math reasoning?
- Does it help or hurt factual/applied multiple-choice benchmarks?
- Do reasoning-trained models respond differently from non-reasoning instruction/chat models?
- Does a Vietnamese-native reasoning model respond better than an English-distilled reasoning model?

### 3.3 Hypotheses

- **H1 (Vietnamese math):** Multi-step math should benefit most from extended thinking, especially on reasoning-trained models.
- **H2 (Factual/applied knowledge):** ViMMLU and VNHSGE may show weak, flat, or negative scaling because longer reasoning does not add missing knowledge.
- **H3 (Vietnamese-specialized non-reasoning models):** Vinallama, Vistral, and SeaLLM may show weaker BF response because they were not trained for long chain-of-thought behavior.
- **H4 (Vietnamese-native reasoning model):** GreenMind-14B-R1 may respond better than English-distilled R1-Distill because its reasoning format and training are aligned with Vietnamese.

---

## 4. Experimental Design

### 4.1 Budget Forcing Adaptation for Vietnamese

- Reuse the language-agnostic BudgetForcingDecoder.
- Replace the English trigger `"Wait"` with Vietnamese `"Chờ một chút"`.
- Add/handle model-specific end-of-thinking delimiters:
  - `<|im_end|>` for Qwen-style models
  - `</think>` for reasoning models
  - `</answer>` for GreenMind
  - `</s>` for LLaMA/Mistral-style Vietnamese models
- Sweep `n_wait ∈ {0, 1, 2}` with `n_wait=0` as greedy baseline.
- Mention KV-cache optimization as an implementation requirement for feasible long runs, not as the main research contribution.

### 4.2 Model Matrix

Use the 2x2 structure:

| | English-primary | Vietnamese-primary |
|---|---|---|
| **Non-reasoning** | qwen2.5-3B | vinallama-7b, vistral-7b, seallm-7b |
| **Reasoning** | r1-distill-7B | greenmind-14b-r1 |

Explain why this design tests both model-family transfer and language transfer.

### 4.3 Benchmarks

| Benchmark | Task type | HF path | Split | Answer format |
|-----------|-----------|---------|-------|---------------|
| vi_gsm8k | multi-step math | `hllj/vi_gsm8k` | test | numeric |
| vimmlu | factual knowledge MC | `tridm/VMLU` | validation | A/B/C/D |
| vnhsge | Vietnamese high-school exam MC | `roshansk23/Vietnam_HighSchool_Exam_Dataset` | train | 1-indexed numeric answers converted to A/B/C/D |

### 4.4 Evaluation Metrics

- Accuracy
- Scaling slope across `n_wait`
- Performance = max accuracy over `n_wait`
- Average thinking tokens
- Extraction failure rate

Emphasize the difference between:

- **mechanical control**: BF increases thinking tokens
- **useful transfer**: increased thinking improves accuracy

### 4.5 Run Plan and Artifacts

- Smoke: small local/Colab checks
- Small matrix: n=20 preliminary observations
- Full runs: n=100 per model/benchmark/n_wait cell
- Results source of truth: `experiments/results/vi_*/summary_vi.csv` and JSON details

Do not fabricate numbers. All values in the report must trace to result artifacts.

---

## 5. Results

### 5.1 Preliminary Small-Matrix Results

Use only with explicit caveats:

- n=20
- qwen2.5-3B and r1-distill-7B
- vi_gsm8k and vimmlu

Do not use preliminary numbers as final claims.

### 5.2 Main Results Table

Populate from `summary_vi.csv` after full runs.

Suggested columns:

| Model | Benchmark | n_wait=0 | n_wait=1 | n_wait=2 | Scaling | Performance |

### 5.3 Scaling Curves

- Accuracy vs `n_wait`
- One figure per benchmark or grouped by model family

### 5.4 Thinking Token Analysis

- Average thinking tokens vs `n_wait`
- Show whether BF mechanically increased thinking even when accuracy did not improve.

### 5.5 Error and Extraction Analysis

- Parser failure rate by model and benchmark
- Common failure modes:
  - no final answer
  - wrong answer format
  - trigger phrase contaminates final extraction
  - reasoning loops
  - EoT delimiter mismatch

---

## 6. Discussion

### 6.1 Does Vietnamese BF Replicate s1's Scaling Signal?

Compare directly to s1:

- s1 showed positive scaling on English reasoning benchmarks.
- Our experiment tests whether this signal survives language/model/benchmark transfer.

### 6.2 Benchmark-Type Effects

- vi_gsm8k: math reasoning
- vimmlu: factual knowledge
- vnhsge: applied/exam knowledge

Discuss whether longer reasoning is useful for each task type.

### 6.3 Model-Family Effects

Compare:

- qwen2.5-3B vs r1-distill-7B
- r1-distill-7B vs greenmind-14b-r1
- Vietnamese-specialized non-reasoning models vs reasoning-trained models

### 6.4 Mechanical Control vs Accuracy Gain

Important interpretive distinction:

- BF may increase thinking tokens.
- That does not guarantee better answers.

This section should explain failures where compute increases but accuracy is flat or worse.

### 6.5 What Our Results Say About s1

Return to the source paper:

- Which parts of s1 seem robust?
- Which parts may depend on model training, task type, or English/Qwen-specific behavior?
- Does BF appear universal, or conditional?

---

## 7. Limitations and Future Work

Limitations:

- n=100 per cell is still small.
- Limited compute and Kaggle session constraints.
- Single Vietnamese trigger phrase.
- No Vietnamese s1K-style fine-tuning.
- Automated answer extraction may bias results.
- No human evaluation of reasoning traces.

Future work:

- Test trigger variants such as `"Đợi đã"` or `"Suy nghĩ thêm"`.
- Create a Vietnamese s1K-style reasoning dataset.
- Fine-tune Vietnamese models before BF.
- Add harder Vietnamese math benchmarks.
- Human-evaluate reasoning trace quality.

---

## 8. Conclusion

Answer both report objectives:

1. What does the s1 paper contribute?
2. What does our Vietnamese transfer experiment show?

Suggested conclusion frame:

> The s1 paper demonstrates that a simple combination of carefully curated reasoning data and Budget Forcing can produce open test-time scaling on English reasoning benchmarks. Our project tests whether the Budget Forcing component transfers to Vietnamese models and benchmarks. The results show [pending], suggesting that test-time scaling transfer depends on [model family/task type/language-aligned reasoning training].

---

## Tables and Figures to Include

### Source Paper Analysis

- Figure/table: s1 recipe = s1K + SFT + Budget Forcing
- Figure: Budget Forcing mechanism (`Wait` / EoT suppression)
- Table: s1 metrics (Control, Scaling, Performance)

### Vietnamese Extension

- Table: model-family 2x2 matrix
- Table: benchmark summary
- Table: main results from `summary_vi.csv`
- Figure: accuracy vs `n_wait`
- Figure: thinking tokens vs `n_wait`
- Optional: extraction failure rate by model/benchmark

---

## References to Fill

- `\cite{TODO:s1}` — s1: Simple Test-Time Scaling
- `\cite{TODO:deepseek-r1}` — DeepSeek-R1
- `\cite{TODO:r1-distill}` — DeepSeek-R1-Distill-Qwen-7B
- `\cite{TODO:mgsm}` — MGSM / multilingual GSM
- `\cite{TODO:vimmlu}` — ViMMLU / VMLU benchmark
- `\cite{TODO:vnhsge}` — VNHSGE dataset
- `\cite{TODO:greenmind}` — GreenMind-14B-R1
- `\cite{TODO:vinallama}` — Vinallama
- `\cite{TODO:vistral}` — Vistral
- `\cite{TODO:seallm}` — SeaLLM

---

## Artifact Traceability Rule

All experiment numbers in Sections 5 and 6 must trace to files in `experiments/results/vi_*/`:

- accuracy numbers: `summary_vi.csv`
- scaling values: `summary_vi.csv`
- average thinking tokens: JSON result files or `summary_vi.csv`
- extraction failures: JSON result files or `summary_vi.csv`

Do not write final numerical claims without artifact support.
