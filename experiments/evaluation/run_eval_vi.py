"""
3-condition evaluation driver for Vietnamese BF+RAG experiments.

Three conditions evaluated per (model, benchmark, n_wait):
  BF_only  — Budget Forcing only, no retrieval (n_wait ∈ {0,1,2})
  RAG_only — Retrieved Vietnamese context, greedy decoding (n_wait=0)
  BF+RAG   — Retrieved context + Budget Forcing (n_wait ∈ {1,2})

Output: one JSON file per (condition, n_wait) in --output_dir.
File naming: {model}__{benchmark}__{condition}__nwait{n_wait}.json

Usage:
    # Single condition
    python experiments/evaluation/run_eval_vi.py \
        --model qwen2.5-3B \
        --benchmark vi_gsm8k \
        --conditions BF_only \
        --n_wait 0 1 2 \
        --n_samples 50

    # All 3 conditions (smoke test)
    python experiments/evaluation/run_eval_vi.py \
        --model qwen2.5-3B \
        --benchmark vi_gsm8k \
        --conditions BF_only RAG_only BF_RAG \
        --n_wait 0 1 2 \
        --n_samples 5 \
        --max_tokens 512 \
        --no_4bit

    # With custom RAG index
    python experiments/evaluation/run_eval_vi.py \
        --model vinallama-7b \
        --benchmark vi_gsm8k \
        --conditions BF_only RAG_only BF_RAG \
        --n_wait 0 1 2 \
        --n_samples 20 \
        --rag_index_dir experiments/data/vi_wiki_index
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_loader import SUPPORTED_MODELS, load_model_and_tokenizer
from budget_forcing import BudgetForcingDecoder

# Import registry from run_eval (reuse BenchmarkSpec + answer logic)
from evaluation.run_eval import (
    BENCHMARK_REGISTRY,
    BenchmarkSpec,
    extract_answer,
    check_answer,
    format_question,
)
from rag.rag_pipeline import RAGPipeline

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_CONDITIONS = ["BF_only", "RAG_only", "BF_RAG"]

DEFAULT_RAG_INDEX_DIR = "experiments/data/vi_wiki_index"
DEFAULT_RAG_SMOKE_INDEX_DIR = "experiments/data/vi_wiki_index_smoke"

VIETNAMESE_THINK_TRIGGER = "Wait"  # default; can override with --trigger


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt_vi(question: str, tokenizer, think: bool = True) -> str:
    """
    Format a Vietnamese question using the model's chat template.
    Optionally append <think> to force chain-of-thought.

    Args:
        question: The (potentially RAG-augmented) question text
        tokenizer: HF tokenizer with apply_chat_template support
        think: If True, append <think> token to prompt

    Returns:
        Formatted prompt string
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý thông minh, giải toán và trả lời câu hỏi bằng tiếng Việt. "
                "Hãy suy nghĩ kỹ trước khi đưa ra câu trả lời cuối cùng."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if think and "<think>" not in prompt:
        prompt += "<think>\n"

    return prompt


# ── Condition logic ───────────────────────────────────────────────────────────

def condition_needs_rag(condition: str) -> bool:
    return condition in ("RAG_only", "BF_RAG")

def condition_needs_bf(condition: str, n_wait: int) -> bool:
    """BF is active only when n_wait > 0 and condition is not RAG_only."""
    if condition == "RAG_only":
        return False
    return n_wait > 0


def effective_n_wait(condition: str, n_wait: int) -> int:
    """RAG_only always uses n_wait=0 regardless of the sweep value."""
    if condition == "RAG_only":
        return 0
    return n_wait


# ── Single run ────────────────────────────────────────────────────────────────

def run_condition(
    condition: str,
    n_wait: int,
    samples: list,
    cfg: BenchmarkSpec,
    model,
    tokenizer,
    rag_pipeline: RAGPipeline,
    max_new_tokens: int,
    trigger: str,
    model_name: str,
    benchmark: str,
) -> dict:
    """
    Run evaluation for one (condition, n_wait) combination.

    Returns:
        dict with accuracy, details list, metadata
    """
    eff_nwait = effective_n_wait(condition, n_wait)
    use_rag = condition_needs_rag(condition)
    use_bf = condition_needs_bf(condition, n_wait)

    decoder = BudgetForcingDecoder(model, tokenizer)

    correct = 0
    extraction_failures = 0
    details = []

    for i, sample in enumerate(tqdm(samples, desc=f"{condition}/nw={eff_nwait}")):
        question = format_question(sample, cfg)
        ground_truth = str(sample[cfg.answer_key])

        # RAG augmentation
        if use_rag and rag_pipeline.enabled:
            augmented_question = rag_pipeline.augment(question)
            retrieved_tokens = rag_pipeline.last_retrieved_tokens
        else:
            augmented_question = question
            retrieved_tokens = 0

        prompt = format_prompt_vi(augmented_question, tokenizer, think=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        t0 = time.time()
        error_msg = None
        try:
            output = decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_wait=eff_nwait,
                trigger=trigger,
            )
            elapsed = time.time() - t0

            predicted = extract_answer(output["answer_text"])
            if not predicted:
                extraction_failures += 1
                predicted = ""
            is_correct = check_answer(predicted, ground_truth)
            correct += int(is_correct)
            thinking_tokens = output.get("thinking_tokens", 0)

        except Exception as e:
            elapsed = time.time() - t0
            error_msg = str(e)
            predicted = ""
            is_correct = False
            thinking_tokens = 0
            print(f"\n  [WARN] Error on sample {i}: {e}")

        details.append({
            "sample_idx": i,
            "question": question[:200] + ("..." if len(question) > 200 else ""),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "thinking_tokens": thinking_tokens,
            "retrieved_tokens": retrieved_tokens,
            "elapsed_sec": round(elapsed, 2),
            "error": error_msg,
        })

    n = len(samples)
    accuracy = correct / n if n > 0 else 0.0

    return {
        "condition": condition,
        "n_wait": eff_nwait,
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "extraction_failures": extraction_failures,
        "avg_thinking_tokens": (
            round(sum(d["thinking_tokens"] for d in details) / n, 1) if n > 0 else 0
        ),
        "avg_retrieved_tokens": (
            round(sum(d["retrieved_tokens"] for d in details) / n, 1) if n > 0 else 0
        ),
        "details": details,
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation_vi(
    model_name: str,
    benchmark: str,
    conditions: List[str],
    n_wait_list: List[int],
    n_samples: int,
    output_dir: str,
    rag_index_dir: Optional[str] = None,
    trigger: str = VIETNAMESE_THINK_TRIGGER,
    load_in_4bit: bool = True,
    max_new_tokens: int = 2048,
    seed: int = 42,
) -> List[dict]:
    """
    Run all requested conditions × n_wait combinations.

    Returns:
        List of result dicts (one per (condition, n_wait))
    """
    import random
    random.seed(seed)

    # Setup output directory
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"vi_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[run_eval_vi] Started at {timestamp}")
    log(f"  model={model_name}, benchmark={benchmark}")
    log(f"  conditions={conditions}, n_wait_list={n_wait_list}, n_samples={n_samples}")
    log(f"  load_in_4bit={load_in_4bit}, max_new_tokens={max_new_tokens}")

    # Load benchmark
    from datasets import load_dataset
    cfg = BENCHMARK_REGISTRY[benchmark]
    log(f"[run_eval_vi] Loading benchmark: {cfg.hf_path}" + (f"/{cfg.hf_name}" if cfg.hf_name else ""))

    try:
        if cfg.hf_name:
            dataset = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.hf_split)
        else:
            dataset = load_dataset(cfg.hf_path, split=cfg.hf_split)
    except Exception as e:
        log(f"[FAIL] Cannot load benchmark {benchmark}: {e}")
        raise

    import random
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    log(f"[run_eval_vi] Sampled {len(samples)} questions")

    # Load model
    log(f"[run_eval_vi] Loading model: {model_name}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name, load_in_4bit=load_in_4bit
        )
    except Exception as e:
        log(f"[FAIL] Cannot load model {model_name}: {e}")
        raise

    # Determine if RAG is needed at all
    needs_rag = any(condition_needs_rag(c) for c in conditions)
    rag_pipeline: RAGPipeline

    if needs_rag:
        # Resolve index dir
        index_dirs_to_try = []
        if rag_index_dir:
            index_dirs_to_try.append(rag_index_dir)
        index_dirs_to_try += [DEFAULT_RAG_INDEX_DIR, DEFAULT_RAG_SMOKE_INDEX_DIR]

        loaded_rag = False
        for idx_dir in index_dirs_to_try:
            if Path(idx_dir).exists():
                log(f"[run_eval_vi] Loading RAG index from: {idx_dir}")
                try:
                    rag_pipeline = RAGPipeline.from_index(idx_dir)
                    loaded_rag = True
                    break
                except Exception as e:
                    log(f"  [WARN] Failed to load index at {idx_dir}: {e}")

        if not loaded_rag:
            log("[WARN] No RAG index found. RAG conditions will be skipped or use empty context.")
            log("       To build index: python experiments/rag/knowledge_base.py --max_docs 10000")
            rag_pipeline = RAGPipeline.disabled()
    else:
        rag_pipeline = RAGPipeline.disabled()

    # Run conditions
    all_results = []
    runtime_meta = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }

    for condition in conditions:
        # Determine which n_wait values to sweep for this condition
        if condition == "RAG_only":
            # RAG_only always uses n_wait=0; run once
            nwaits = [0]
        elif condition in ("BF_only", "BF_RAG"):
            nwaits = n_wait_list
        else:
            nwaits = n_wait_list

        for nw in nwaits:
            log(f"\n--- Condition: {condition}, n_wait={nw} ---")

            try:
                result = run_condition(
                    condition=condition,
                    n_wait=nw,
                    samples=samples,
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    rag_pipeline=rag_pipeline,
                    max_new_tokens=max_new_tokens,
                    trigger=trigger,
                    model_name=model_name,
                    benchmark=benchmark,
                )
            except Exception as e:
                log(f"[FAIL] condition={condition} n_wait={nw}: {e}")
                result = {
                    "condition": condition,
                    "n_wait": nw,
                    "error": str(e),
                    "accuracy": None,
                    "correct": 0,
                    "n_samples": len(samples),
                    "extraction_failures": 0,
                    "avg_thinking_tokens": 0,
                    "avg_retrieved_tokens": 0,
                    "details": [],
                }

            log(
                f"  accuracy={result.get('accuracy')}, "
                f"correct={result.get('correct')}/{len(samples)}, "
                f"extraction_failures={result.get('extraction_failures')}"
            )

            # Save individual JSON
            safe_model = model_name.replace("/", "_")
            safe_bench = benchmark
            safe_cond = condition
            fname = f"{safe_model}__{safe_bench}__{safe_cond}__nwait{nw}.json"

            payload = {
                "model": model_name,
                "benchmark": benchmark,
                "language": "vi",
                "condition": condition,
                "n_wait": nw,
                "trigger": trigger,
                "run_dir": str(run_dir),
                "timestamp_utc": timestamp,
                "runtime": runtime_meta,
                "rag_enabled": condition_needs_rag(condition),
                "rag_index_dir": rag_index_dir,
                "load_in_4bit": load_in_4bit,
                "seed": seed,
                **result,
            }

            out_path = run_dir / fname
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            log(f"  Saved: {out_path}")
            all_results.append(payload)

    log(f"\n[run_eval_vi] All conditions done. Results in: {run_dir}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_output_dir = str(Path(__file__).resolve().parents[1] / "results")

    parser = argparse.ArgumentParser(
        description="Vietnamese BF+RAG 3-condition evaluation"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-3B",
        help="Model key (from model_loader.SUPPORTED_MODELS) or HF model ID",
    )
    parser.add_argument(
        "--benchmark",
        default="vi_gsm8k",
        choices=list(BENCHMARK_REGISTRY.keys()),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Conditions to run (default: all 3)",
    )
    parser.add_argument(
        "--n_wait",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="n_wait values to sweep for BF conditions (default: 0 1 2)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)",
    )
    parser.add_argument(
        "--output_dir",
        default=default_output_dir,
        help="Root output directory (a timestamped subdir vi_YYYYMMDD_HHMMSS is created)",
    )
    parser.add_argument(
        "--rag_index_dir",
        default=None,
        help="Path to pre-built FAISS index. Defaults to experiments/data/vi_wiki_index/",
    )
    parser.add_argument(
        "--trigger",
        default=VIETNAMESE_THINK_TRIGGER,
        help=f"BF trigger phrase (default: '{VIETNAMESE_THINK_TRIGGER}')",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use float16/bfloat16)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate per question (default: 2048)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_evaluation_vi(
        model_name=args.model,
        benchmark=args.benchmark,
        conditions=args.conditions,
        n_wait_list=args.n_wait,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        rag_index_dir=args.rag_index_dir,
        trigger=args.trigger,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )
