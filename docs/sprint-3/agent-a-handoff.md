# Agent A Handoff — Sprint 3

Role: Coder / Experiment owner
Updated: 2026-06-11

---

## Mission

Build the Vietnamese BF+RAG evaluation pipeline from scratch.
Previous sprint (Sprint 2) implemented Gap 1 cross-family BF — archived, do not expand.
This sprint's code is the primary deliverable.

---

## Starting State

Files you can reuse unchanged:
- `experiments/budget_forcing/decoding.py` — BudgetForcingDecoder (language-agnostic)
- `experiments/budget_forcing/metrics.py` — scaling/performance metrics (language-agnostic)

Files you must extend (do NOT rewrite from scratch):
- `experiments/models/model_loader.py` — add 3 Vietnamese models
- `experiments/evaluation/run_eval.py` — add 2 Vietnamese benchmark specs to BENCHMARK_REGISTRY

Files you must create (do not exist yet):
- `experiments/rag/__init__.py`
- `experiments/rag/retriever.py`
- `experiments/rag/knowledge_base.py`
- `experiments/rag/rag_pipeline.py`
- `experiments/data/download_vi_benchmarks.py`
- `experiments/evaluation/run_eval_vi.py`  ← most important
- `experiments/scripts/run_vi_bf_rag.sh`
- `experiments/results/summary_vi.py`

---

## Task Checklist

### T1 — Extend model_loader.py

Add to `SUPPORTED_MODELS`:
```python
"vinallama-7b": {
    "hf_name": "vilm/vinallama-7b-chat",
    "dtype": "float16",
    "context_len": 4096,
    "chat_template": "llama",
},
"vistral-7b": {
    "hf_name": "vilm/vistral-7b-chat",
    "dtype": "float16",
    "context_len": 4096,
    "chat_template": "mistral",
},
"seallm-7b": {
    "hf_name": "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "dtype": "float16",
    "context_len": 8192,
    "chat_template": "chatml",
},
```
Use 4-bit NF4 quantization (BitsAndBytes) for all 7B models.

### T2 — Extend run_eval.py BENCHMARK_REGISTRY

Add:
```python
"vi_gsm8k": BenchmarkSpec(
    dataset_name="juletxara/mgsm",
    dataset_config="vi",
    split="test",
    question_field="question",
    answer_field="answer_number",
    prompt_type="math",
    language="vi",
),
"vimmlu": BenchmarkSpec(
    dataset_name="vilm/vimmlu",
    dataset_config=None,
    split="test",
    question_field="question",
    answer_field="answer",
    prompt_type="mcq",
    language="vi",
),
```

### T3 — Create experiments/rag/ module

`retriever.py`: FAISS flat L2 index, `paraphrase-multilingual-MiniLM-L12-v2` embeddings,
top-k=3, max 128 tokens per passage.

`knowledge_base.py`: loads Vi Wikipedia (`wikimedia/wikipedia`, `20231101.vi`),
chunks to ≤128 tokens, builds FAISS index, saves to `data/vi_wiki_index/`.
Include `--max_docs` arg for dev/testing (e.g. 10000 docs for smoke).

`rag_pipeline.py`: given query + retriever, returns augmented prompt string.
Format:
```
[Ngữ cảnh tham khảo]
{passage_1}
{passage_2}
{passage_3}

Câu hỏi: {question}
```

### T4 — Create download_vi_benchmarks.py

Downloads and prints 3 sample rows from each benchmark.
Usage: `python experiments/data/download_vi_benchmarks.py --benchmark vi_gsm8k`
Acts as a validation script — confirms dataset exists and fields match BenchmarkSpec.

### T5 — Create run_eval_vi.py (primary deliverable)

Three conditions per (model, benchmark, n_wait):
- `BF_only`: n_wait > 0, no retrieval
- `RAG_only`: retrieved context, n_wait = 0, no BF
- `BF+RAG`: retrieved context + BF (n_wait > 0)

Output: one JSON file per run at `experiments/results/vi_{timestamp}/`:
```
{model}__{benchmark}__{condition}__nwait{n_wait}.json
```

Each JSON: list of dicts with keys:
```
question, reference_answer, model_output, extracted_answer, correct,
thinking_tokens, retrieved_tokens, condition, n_wait, model, benchmark
```

CLI: `python experiments/evaluation/run_eval_vi.py --model qwen2.5-3B --benchmark vi_gsm8k --conditions BF_only RAG_only BF_RAG --n_wait 0 1 2 --n_samples 5 --output_dir experiments/results/`

### T6 — Create run_vi_bf_rag.sh

Sweep script controlled by env vars:
- `MODELS` (space-separated)
- `BENCHMARKS` (space-separated)
- `CONDITIONS` (default: `BF_only RAG_only BF_RAG`)
- `N_WAIT_LIST` (default: `0 1 2`)
- `N_SAMPLES` (default: 100)
- `EXTRA_ARGS` (forwarded to run_eval_vi.py)

Logs each run to `experiments/results/vi_{timestamp}/run.log`.

### T7 — Create summary_vi.py

Reads all JSON files in a results directory, aggregates metrics,
outputs `summary_vi.csv` + `summary_vi.md`.

Column schema:
```
model, benchmark, language, condition, n_wait,
n_samples, accuracy, scaling, performance,
avg_thinking_tokens, avg_retrieved_tokens,
extraction_failures, cuda_available, mps_available,
run_dir, timestamp_utc
```

### T8 — Smoke test

```bash
MODELS='qwen2.5-3B' BENCHMARKS='vi_gsm8k' \
CONDITIONS='BF_only RAG_only BF_RAG' \
N_WAIT_LIST='0 1 2' N_SAMPLES=5 \
EXTRA_ARGS='--max_tokens 512 --no_4bit' \
bash experiments/scripts/run_vi_bf_rag.sh
```

Expected output:
- `experiments/results/vi_{timestamp}/` exists
- JSON files for each (condition, n_wait) combination
- `summary_vi.csv` has 3+ rows

---

## Key Implementation Notes

**BF for Vietnamese models:** The EoT token list in `decoding.py` covers `</think>` and `<|im_end|>`.
For vinallama/vistral (LLaMA/Mistral base), also check `</s>` as EoT.

**Answer extraction for Vietnamese:** MGSM-vi answers are numeric. Use the existing
`extract_answer()` boxed/digit pattern; add fallback for Vietnamese ordinals if needed.
ViMMLU answers are A/B/C/D letter choices — existing MCQ extractor should work.

**RAG index:** For smoke test, pass `--max_docs 5000` to `build_index` to avoid
downloading all of Vi Wikipedia. Full index (~1.5M articles) can be pre-built once and reused.

**Hardware fallback:** If CUDA unavailable, BitsAndBytes 4-bit will fail. Check `cuda_available`
and fall back to `float16` on CPU/MPS. Log this in the JSON output.

**Do not fabricate numbers.** If a run fails, record the error in the JSON and keep
`accuracy: null` in summary_vi.csv. Never fill in plausible-looking numbers.

---

## Blockers to Record

If any of these happen, document in `experiments/results/BLOCKERS.md`:
- Model not available on HF or requires gated access
- Dataset field mismatch (update BenchmarkSpec and note the change)
- OOM error (note model, GPU, batch size)
- Vietnamese answer extraction failure rate > 30% (flag for parser review)

---

## Handoff to Sprint 4

When smoke test passes, hand off to sprint-4/agent-a-handoff.md with:
- Path to smoke run results
- Any extraction failure patterns observed
- Confirmed model list (which models loaded successfully)
