<!-- markdownlint-disable MD007 MD060 -->

# Project Brief

Updated: 2026-07-02 (code-aligned rewrite)

## 1. Executive Summary

- **Project:** Does Test-Time Scaling via Budget Forcing Transfer to Vietnamese Language Reasoning?
- **Core idea:** Reproduce the *inference-time* part of the s1 paper on Vietnamese tasks by applying Budget Forcing (BF) at decoding time, without training a new Vietnamese s1-style model.
- **Actual repository scope:** **BF-only**. The current repo does **not** implement RAG, BF+RAG, or Vietnamese s1K-style supervised fine-tuning.
- **Primary contribution of this repo state:** a reproducible evaluation pipeline for testing whether the BF mechanism from s1 transfers across Vietnamese benchmarks and model families.
- **Deliverable target:** experiment artifacts, summary tables/figures, and documentation detailed enough to support a full scientific report.

## 2. Research Positioning

### 2.1 Relation to the s1 paper

The reference paper *s1: Simple Test-Time Scaling* (EMNLP 2025) combines two ingredients:

1. **Train-time ingredient:** supervised fine-tuning on a curated 1K reasoning dataset (s1K).
2. **Test-time ingredient:** Budget Forcing, a decoding intervention that prolongs or truncates reasoning.

This repository currently studies only the **second ingredient** in Vietnamese. In other words, this repo is **not** a full reproduction of s1. It is a transfer study of the BF decoding mechanism.

### 2.2 Research Gap

- The original s1 paper is English-centric and mostly validated on Qwen-family models.
- The current repo investigates whether BF generalizes to:
   - Vietnamese-language reasoning tasks.
   - Vietnamese-specialized chat models.
   - A broader mix of reasoning-native, multilingual, and Vietnamese-adapted model families.

### 2.3 Precise Claim Boundary

The strongest defensible claim from this repo is:

> Budget Forcing can be evaluated as a test-time intervention on Vietnamese tasks, but its effectiveness appears to be **model-family dependent rather than universally beneficial**.

The repo does **not** currently justify claims about:

- Vietnamese s1K distillation or fine-tuning.
- BF vs RAG.
- BF+RAG synergy.
- Cross-lingual transfer under a controlled train-time setup.

## 3. Current Research Questions

### 3.1 Primary RQ

> Does Budget Forcing improve Vietnamese-language reasoning performance, and is the scaling signal consistent across different model families?

### 3.2 Secondary RQs

- **RQ1:** On Vietnamese math reasoning (`vi_gsm8k`), does accuracy improve as `n_wait` increases?
- **RQ2:** On Vietnamese factual / multiple-choice benchmarks (`vimmlu`, `vnhsge`), does BF help or hurt?
- **RQ3:** Do reasoning-native models react differently from Vietnamese-specialized instruction models?
- **RQ4:** Is positive BF scaling a property of the language/task, or mainly of the model's training recipe?

## 4. Implemented Experimental Scope

### 4.1 What is implemented

The implemented condition is a single-condition sweep:

| Component | Actual implementation |
|-----------|------------------------|
| Condition | `BF-only` |
| Compute control variable | `n_wait ∈ {0, 1, 2}` |
| Baseline | `n_wait = 0` |
| Trigger phrase | `"Chờ một chút"` |
| Optional safeguard | `--max_thinking_tokens` to inject `Final Answer:` |
| Outputs | one JSON per `(model, benchmark, n_wait)` |

### 4.2 What is not implemented

- No retrieval pipeline.
- No RAG index.
- No BF+RAG condition.
- No Vietnamese s1K dataset construction.
- No supervised fine-tuning stage inside this repo.
- No paper-style control metric computed end-to-end from actual target token budgets in the Vietnamese pipeline.

## 5. Benchmarks in Code

The benchmark registry in `experiments/evaluation/run_eval.py` defines **three** Vietnamese benchmarks, not two.

| Key | HF dataset | Split used in evaluation | Size in registry | Answer type | Intended role |
|-----|------------|--------------------------|------------------|-------------|---------------|
| `vi_gsm8k` | `hllj/vi_gsm8k` | `test` | 250 | free-form numeric | Vietnamese math reasoning |
| `vimmlu` | `tridm/VMLU` | `validation` | 744 | multiple-choice (`A-E`) | Vietnamese factual / academic knowledge |
| `vnhsge` | `roshansk23/Vietnam_HighSchool_Exam_Dataset` | `train` | 6663 | multiple-choice (`1-5` mapped to `A-E`) | Vietnamese high-school exam benchmark |

### Important benchmark note

Earlier documentation in the repo referenced `vilm/vimmlu` and a RAG-oriented benchmark plan. That is no longer the code reality. The current evaluation code uses:

- `tridm/VMLU` for `vimmlu`.
- `validation` split for evaluation because labeled answers are available there.
- `vnhsge` as a third benchmark already integrated into the active pipeline.

## 6. Models in Scope

### 6.1 Main matrix currently supported

| Model key | HF model ID | Family / role | Status in repo narrative |
|-----------|-------------|---------------|---------------------------|
| `qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` | multilingual instruction | smoke-test baseline |
| `r1-distill-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | reasoning-specialized | main reasoning baseline |
| `vinallama-7b` | `vilm/vinallama-7b-chat` | Vietnamese-specialized LLaMA-family chat | main Vietnamese-specialized comparison |
| `vistral-7b` | `Viet-Mistral/Vistral-7B-Chat` | Vietnamese-specialized Mistral-family chat | main Vietnamese-specialized comparison |
| `seallm-7b` | `SeaLLMs/SeaLLMs-v3-7B-Chat` | SEA multilingual chat | regional multilingual baseline |
| `greenmind-14b-r1` | `GreenNode/GreenMind-Medium-14B-R1` | Vietnamese reasoning-native model | advanced comparison added later |

### 6.2 Additional models in registry but not part of the main write-up target

The loader also contains `s1-32B`, `r1-distill-14B`, `phi4-reasoning`, `qwen2.5-7B`, `llama3-8B`, `gemma4-E2B-it`. These are available in code but are not the center of the current Vietnamese experiment matrix.

## 7. Methodology Actually Used by the Repo

### 7.1 Prompting protocol

The Vietnamese evaluation driver (`experiments/evaluation/run_eval_vi.py`) does the following:

1. Builds a Vietnamese system prompt via `tokenizer.apply_chat_template(...)` when possible.
2. Uses a Vietnamese instruction that asks the model to reason carefully.
3. Appends `<think>` **only when `n_wait > 0`**.

This means the baseline `n_wait = 0` is not just "BF disabled"; it also avoids the explicit thinking tag used in BF conditions. Any report should describe this precisely.

### 7.2 Budget Forcing mechanics

The decoder in `experiments/budget_forcing/decoding.py` implements two paper-inspired interventions:

- **Enforce minimum compute:** when the model tries to emit an end-of-thinking token and `waits_triggered < n_wait`, the decoder suppresses that stop and appends `"\nChờ một chút"`.
- **Enforce maximum compute:** if `--max_thinking_tokens` is set and the reasoning trace grows too long, the decoder injects `"\n\nFinal Answer:"`.

Supported end-of-thinking markers include:

- `<|im_end|>`
- `</think>`
- `</answer>`
- `####`
- `\n\nFinal Answer:`
- `\n\nAnswer:`
- model `eos_token_id`

### 7.3 Answer extraction and checking

The repo does not use a single answer parser. It uses layered fallbacks in `experiments/evaluation/run_eval.py`:

- `<answer>...</answer>`
- LaTeX `\boxed{...}`
- `Final Answer:` / `The answer is`
- last clean multiple-choice line (`A-E`)
- last clean numeric line

Comparison logic supports:

- exact normalized string match,
- multiple-choice letter extraction,
- numeric label to letter mapping for `vnhsge` (`1 → A`, `2 → B`, ...),
- tolerant numeric comparison.

### 7.4 Aggregation logic

`experiments/results/summary_vi.py` computes:

- `accuracy`
- `scaling` as OLS slope of accuracy vs `n_wait`
- `performance` as max accuracy across `n_wait`
- `avg_thinking_tokens`
- `extraction_failures`

Important nuance:

- `summary_vi.py` groups by `(model, benchmark, condition)`, but `run_eval_vi.py` does not explicitly write a `condition` field.
- In practice this is harmless because the current repo contains only BF-only Vietnamese runs.

## 8. Code Status by Module

| Module | Status | What it really does |
|--------|--------|---------------------|
| `experiments/budget_forcing/decoding.py` | active | BF decoder with minimum/maximum compute intervention |
| `experiments/budget_forcing/metrics.py` | partially used | paper-style metrics helper; not the sole source for Vietnamese summaries |
| `experiments/models/model_loader.py` | active | model registry + device detection + optional 4-bit quantization |
| `experiments/evaluation/run_eval.py` | active | benchmark registry, prompt helpers, answer parsing |
| `experiments/evaluation/run_eval_vi.py` | active primary entrypoint | Vietnamese BF-only sweep driver |
| `experiments/data/download_vi_benchmarks.py` | active | dataset accessibility / field validation helper |
| `experiments/scripts/run_vi_bf.sh` | active | environment-driven sweep launcher |
| `experiments/results/summary_vi.py` | active | JSON aggregation into CSV/Markdown |
| `experiments/results/visualize.py` | active | publication-style figures from consolidated results |

## 9. Canonical Commands

### 9.1 Smoke test

```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf.sh
```

### 9.2 Small matrix

```bash
MODELS='qwen2.5-3B r1-distill-7B' BENCHMARKS='vi_gsm8k vimmlu' \
N_WAIT_LIST='0 1 2' N_SAMPLES=20 \
bash experiments/scripts/run_vi_bf.sh
```

### 9.3 Main matrix pattern

```bash
MODELS='qwen2.5-3B r1-distill-7B vinallama-7b vistral-7b seallm-7b greenmind-14b-r1' \
BENCHMARKS='vi_gsm8k vimmlu vnhsge' \
N_WAIT_LIST='0 1 2' N_SAMPLES=100 \
bash experiments/scripts/run_vi_bf.sh
```

### 9.4 Aggregate summaries

```bash
python experiments/results/summary_vi.py \
   --results_dir experiments/results/vi_YYYYMMDD_HHMMSS
```

### 9.5 Build consolidated figures

```bash
python experiments/results/visualize.py
```

## 10. Reproducibility and Artifact Schema

### 10.1 Output directory pattern

Each run creates a timestamped directory:

```text
experiments/results/vi_YYYYMMDD_HHMMSS/
```

Inside it, each file corresponds to one `(model, benchmark, n_wait)` combination:

```text
{model}__{benchmark}__nwait{n_wait}.json
```

### 10.2 Important fields actually saved

Each JSON payload includes at least:

- `model`
- `benchmark`
- `language`
- `n_wait`
- `trigger`
- `run_dir`
- `timestamp_utc` (name is misleading; see caveat below)
- `runtime`
- `load_in_4bit`
- `seed`
- `accuracy`
- `correct`
- `n_samples`
- `extraction_failures`
- `avg_thinking_tokens`
- `details`

Each element of `details` includes:

- `sample_idx`
- truncated `question`
- `ground_truth`
- `predicted`
- `correct`
- `thinking_tokens`
- `answer_text`
- `thinking_text`
- `elapsed_sec`
- `error`

### 10.3 Reproducibility caveats

These issues should be explicitly acknowledged in any report:

1. **Original dataset indices are not currently persisted in JSON.** Reproduction relies on `seed` and dataset order.
2. **`question` is truncated to 200 characters in saved details.** Full error-analysis reconstruction may require reloading the dataset with the same sampling seed.
3. **`timestamp_utc` is a misnamed field.** The string is generated in `Asia/Ho_Chi_Minh` local time format, not true UTC ISO timestamp.
4. **The baseline condition differs in prompt format.** `n_wait=0` does not receive the `<think>` cue that BF conditions do.

## 11. Current Artifact Inventory

### 11.1 Source-of-truth files for report writing

- Per-run JSON artifacts: `experiments/results/vi_*/`
- Consolidated summary table: `experiments/results/figures/summary_vi.csv`
- Human-readable consolidated summary: `experiments/results/summary_vi.md`
- Figure generator: `experiments/results/visualize.py`

### 11.2 Completed result coverage visible in the repo

The consolidated figure summary currently contains 100-sample results for:

- `r1-distill-7B`
- `seallm-7b`
- `vinallama-7b`
- `vistral-7b`
- `greenmind-14b-r1`

across:

- `vi_gsm8k`
- `vimmlu`
- `vnhsge`

The `qwen2.5-3B` model remains the documented smoke baseline, but it is not part of the main consolidated 100-sample table in `experiments/results/figures/summary_vi.csv`.

## 12. Preliminary Findings from Existing Artifacts

These observations come from `experiments/results/figures/summary_vi.csv` and should be described as **preliminary empirical findings**, not final polished conclusions.

### 12.1 Strongest positive BF responder in the current artifacts

- `r1-distill-7B` shows modest positive gains on all three benchmarks:
   - `vi_gsm8k`: `53 → 56 → 57`
   - `vimmlu`: `27 → 23 → 31`
   - `vnhsge`: `22 → 21 → 25`

### 12.2 Mixed responder

- `seallm-7b` is strongly mixed:
   - negative on `vi_gsm8k`: `66 → 50 → 47`
   - positive on `vimmlu`: `46 → 52 → 56`
   - positive on `vnhsge`: `47 → 51 → 53`

### 12.3 Vietnamese-specialized chat models often degrade under BF

- `vinallama-7b` declines on all three benchmarks.
- `vistral-7b` also declines, with especially large drops on `vimmlu` and `vnhsge`.

### 12.4 Reasoning-native Vietnamese model does not automatically benefit

- `greenmind-14b-r1` has very strong baselines, especially on `vi_gsm8k` (`80` at `n_wait=0`), but BF hurts it in the current settings.

### 12.5 Working hypothesis suggested by current results

The repo's current evidence suggests:

> Budget Forcing is not a universal improvement knob for Vietnamese reasoning. It appears to help some reasoning-oriented or regionally multilingual models, while harming several Vietnamese-specialized instruction models and even a strong Vietnamese reasoning-native model under the current trigger/setup.

## 13. Threats to Validity

| Type | Risk | Current mitigation / note |
|------|------|---------------------------|
| Construct | `n_wait` is only a proxy for compute | log `avg_thinking_tokens` for every run |
| Construct | prompt differs between baseline and BF runs | must be stated explicitly in report |
| Construct | parser errors may confound accuracy | log `extraction_failures` separately |
| Internal | end-of-thinking tokens vary by model family | decoder supports multiple EoT strings |
| Internal | quantization / device differences | runtime metadata includes CUDA/MPS and loader settings |
| Internal | some models may loop under BF | optional `--max_thinking_tokens` is available |
| External | no Vietnamese s1-style fine-tuning | claims limited to test-time transfer only |
| External | open-source model mix is heterogeneous | report should avoid over-generalizing across all Vietnamese LLMs |

## 14. Report-Writing Guidance for Other Agents

If another agent writes the paper/report from this repo state, it should:

1. Treat the study as a **Vietnamese BF transfer study**, not a full s1 reproduction.
2. State clearly that the repo evaluates **off-the-shelf or already-trained open models**.
3. Emphasize the three benchmark types now present in code: math, academic/factual MCQ, and high-school exam MCQ.
4. Use `experiments/results/figures/summary_vi.csv` as the consolidated numeric source of truth.
5. Separate **empirical findings already observed** from **planned future extensions**.
6. Explicitly document the reproducibility limitations noted above.

## 15. Recommended Next Steps

- Add original dataset indices to saved JSON payloads.
- Save a true UTC timestamp field alongside the local timestamp.
- Decide whether `n_wait=0` should share the same `<think>` prompt format for a cleaner ablation.
- Run a controlled trigger ablation (`Chờ một chút`, `Đợi đã`, `Hãy suy nghĩ lại`).
- Evaluate whether `--max_thinking_tokens` should be standardized for non-reasoning chat models.
