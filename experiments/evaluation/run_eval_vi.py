"""
BF-only evaluation driver for Vietnamese test-time scaling experiments.

Research question: Does Budget Forcing (test-time scaling via decoding)
transfer to Vietnamese-language reasoning?

Conditions: BF_only — n_wait ∈ {0, 1, 2} (n_wait=0 is greedy baseline)
No retrieval. RAG is off-scope for this study.

Output: one JSON per n_wait in --output_dir.
File naming: {model}__{benchmark}__nwait{n_wait}.json

Usage:
    # Smoke test (5 samples, no 4-bit)
    python experiments/evaluation/run_eval_vi.py \\
        --model qwen2.5-3B \\
        --benchmark vi_gsm8k \\
        --n_wait 0 1 2 \\
        --n_samples 5 \\
        --max_tokens 512 \\
        --no_4bit

    # Full run
    python experiments/evaluation/run_eval_vi.py \\
        --model r1-distill-7B \\
        --benchmark vi_gsm8k \\
        --n_wait 0 1 2 \\
        --n_samples 100

    # Vietnamese-specialized model
    python experiments/evaluation/run_eval_vi.py \\
        --model vinallama-7b \\
        --benchmark vimmlu \\
        --n_wait 0 1 2 \\
        --n_samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import torch
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_loader import load_model_and_tokenizer
from budget_forcing import BudgetForcingDecoder

# Import registry from run_eval (reuse BenchmarkSpec + answer logic)
from evaluation.run_eval import (
    BENCHMARK_REGISTRY,
    BenchmarkSpec,
    extract_answer,
    check_answer,
    format_question,
)

# ── Constants ─────────────────────────────────────────────────────────────────

VIETNAMESE_THINK_TRIGGER = "Chờ một chút"  # Vietnamese; removes language-mixing confound

# EoT tokens that may appear in Vietnamese-specialized models (LLaMA/Mistral base)
EXTRA_EOT_TOKENS = ["</s>", "<|endoftext|>"]


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt_vi(question: str, tokenizer, think: bool = True) -> str:
    """
    Format a Vietnamese question using the model's chat template.
    Appends <think> to prompt when think=True to signal chain-of-thought.
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

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        prompt = (
            "Bạn là trợ lý thông minh, giải toán và trả lời câu hỏi bằng tiếng Việt.\n\n"
            f"Người dùng: {question}\nTrợ lý:"
        )

    if think and "<think>" not in prompt:
        prompt += "<think>\n"

    return prompt


# ── Single n_wait run ─────────────────────────────────────────────────────────

def run_bf(
    n_wait: int,
    samples: list,
    cfg: BenchmarkSpec,
    model,
    tokenizer,
    max_new_tokens: int,
    trigger: str,
    model_name: str,
    benchmark: str,
) -> dict:
    """
    Run BF evaluation for one n_wait value.
    n_wait=0 is the greedy baseline (BF disabled).

    Returns:
        dict with accuracy, details list, metadata
    """
    decoder = BudgetForcingDecoder(model, tokenizer)

    correct = 0
    extraction_failures = 0
    details = []

    for i, sample in enumerate(tqdm(samples, desc=f"n_wait={n_wait}")):
        question = format_question(sample, cfg)
        ground_truth = str(sample[cfg.answer_key])

        prompt = format_prompt_vi(question, tokenizer, think=(n_wait > 0))
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        t0 = time.time()
        error_msg = None
        try:
            output = decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_wait=n_wait,
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
            "elapsed_sec": round(elapsed, 2),
            "error": error_msg,
        })

    n = len(samples)
    accuracy = correct / n if n > 0 else 0.0

    return {
        "n_wait": n_wait,
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "extraction_failures": extraction_failures,
        "avg_thinking_tokens": (
            round(sum(d["thinking_tokens"] for d in details) / n, 1) if n > 0 else 0
        ),
        "details": details,
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation_vi(
    model_name: str,
    benchmark: str,
    n_wait_list: List[int],
    n_samples: int,
    output_dir: str,
    trigger: str = VIETNAMESE_THINK_TRIGGER,
    load_in_4bit: bool = True,
    max_new_tokens: int = 2048,
    seed: int = 42,
) -> List[dict]:
    """
    Run BF sweep across all n_wait values for one (model, benchmark).

    Returns:
        List of result dicts (one per n_wait)
    """
    import random
    random.seed(seed)

    # Setup output directory
    timestamp = datetime.now(tz=ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"vi_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[run_eval_vi] Started at {timestamp}")
    log(f"  model={model_name}, benchmark={benchmark}")
    log(f"  n_wait_list={n_wait_list}, n_samples={n_samples}")
    log(f"  trigger='{trigger}', load_in_4bit={load_in_4bit}, max_new_tokens={max_new_tokens}")

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

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    log(f"[run_eval_vi] Sampled {len(samples)} questions (seed={seed})")

    # Load model
    log(f"[run_eval_vi] Loading model: {model_name}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name, load_in_4bit=load_in_4bit
        )
    except Exception as e:
        log(f"[FAIL] Cannot load model {model_name}: {e}")
        raise

    runtime_meta = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }

    # Run n_wait sweep
    all_results = []

    for nw in n_wait_list:
        log(f"\n--- n_wait={nw} ---")

        try:
            result = run_bf(
                n_wait=nw,
                samples=samples,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                trigger=trigger,
                model_name=model_name,
                benchmark=benchmark,
            )
        except Exception as e:
            log(f"[FAIL] n_wait={nw}: {e}")
            result = {
                "n_wait": nw,
                "error": str(e),
                "accuracy": None,
                "correct": 0,
                "n_samples": len(samples),
                "extraction_failures": 0,
                "avg_thinking_tokens": 0,
                "details": [],
            }

        log(
            f"  accuracy={result.get('accuracy')}, "
            f"correct={result.get('correct')}/{len(samples)}, "
            f"extraction_failures={result.get('extraction_failures')}"
        )

        # Save JSON
        safe_model = model_name.replace("/", "_")
        fname = f"{safe_model}__{benchmark}__nwait{nw}.json"

        payload = {
            "model": model_name,
            "benchmark": benchmark,
            "language": "vi",
            "n_wait": nw,
            "trigger": trigger,
            "run_dir": str(run_dir),
            "timestamp_utc": timestamp,
            "runtime": runtime_meta,
            "load_in_4bit": load_in_4bit,
            "seed": seed,
            **result,
        }

        out_path = run_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        log(f"  Saved: {out_path}")
        all_results.append(payload)

    log(f"\n[run_eval_vi] Sweep done. Results in: {run_dir}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_output_dir = str(Path(__file__).resolve().parents[1] / "results")

    parser = argparse.ArgumentParser(
        description="Vietnamese Budget Forcing evaluation — BF-only n_wait sweep"
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
        "--n_wait",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="n_wait values to sweep (default: 0 1 2). n_wait=0 is greedy baseline.",
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
        "--trigger",
        default=VIETNAMESE_THINK_TRIGGER,
        help=f"BF trigger phrase appended when EoT suppressed (default: '{VIETNAMESE_THINK_TRIGGER}')",
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
        n_wait_list=args.n_wait,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        trigger=args.trigger,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )
