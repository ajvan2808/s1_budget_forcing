"""
Evaluation pipeline cho Budget Forcing experiments.
Chạy model trên benchmark (MATH500 / AIME24) với các cấu hình BF khác nhau.

Sử dụng:
    python run_eval.py \\
        --model qwen2.5-7B \\
        --benchmark math500 \\
        --n_wait 0 1 2 4 \\
        --n_samples 50 \\
        --output_dir ../results/
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from tqdm import tqdm

from datasets import load_dataset
import torch

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_loader import SUPPORTED_MODELS, load_model_and_tokenizer
from budget_forcing import BudgetForcingDecoder, compute_all_metrics


# ── Benchmark configs ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BenchmarkSpec:
    hf_path: str
    hf_split: str
    question_key: str
    answer_key: str
    n_total: int
    hf_name: str | None = None
    choices_key: str | None = None


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {
    # ── Vietnamese benchmarks (Sprint 3+) ──────────────────────────────────────
    "vi_gsm8k": BenchmarkSpec(
        hf_path="hllj/vi_gsm8k",
        hf_name=None,
        hf_split="test",
        question_key="question",
        answer_key="answer",   # "answer_number" absent in this dataset variant
        n_total=250,
    ),
    "vimmlu": BenchmarkSpec(
        hf_path="tridm/VMLU",
        hf_name=None,
        hf_split="validation",  # test split has answer=None; validation has 744 labelled samples
        question_key="question",
        answer_key="answer",
        choices_key="choices",
        n_total=744,
    ),
    "vnhsge": BenchmarkSpec(
        hf_path="roshansk23/Vietnam_HighSchool_Exam_Dataset",
        hf_name=None,
        hf_split="train",
        question_key="question",
        answer_key="answer",
        choices_key="options",  # flat list WITHOUT letter prefixes — added by format_question
        n_total=6663,           # train split size (answer is 1-indexed: "1"→A, "2"→B, ...)
    ),
}


# ── Prompt template ───────────────────────────────────────────────────────────

THINKING_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant that solves math problems step by step.
Think carefully before giving your final answer.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>
"""


def format_prompt(question: str, tokenizer: Any) -> str:
    """Sử dụng chat template của model và ép model bắt đầu bằng <think>."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems step by step. Think carefully before giving your final answer."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Ép model bắt đầu bằng <think> nếu chưa có
    if "<think>" not in prompt:
        prompt += "<think>\n"
    return prompt


def format_question(sample: dict, cfg: BenchmarkSpec) -> str:
    """Format câu hỏi, có hỗ trợ multiple-choice benchmark như VMLU, ARC-Challenge."""
    question = str(sample[cfg.question_key])
    if not cfg.choices_key:
        return question

    choices = sample.get(cfg.choices_key)

    # Flat list of choices — may or may not have letter prefixes
    if isinstance(choices, list) and choices:
        first = str(choices[0])
        if re.match(r'^[A-E]\.?\s', first):
            # Already prefixed: ["A. text", "B. text", ...] (VMLU style)
            return f"{question}\n\nOptions:\n" + "\n".join(str(c) for c in choices)
        else:
            # No prefixes: ["text1", "text2", ...] (VNHSGE style) — add A/B/C/D
            letters = "ABCDE"
            option_lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices)]
            return f"{question}\n\nOptions:\n" + "\n".join(option_lines)

    # ARC-Challenge / other dict format: {"label": [...], "text": [...]}
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if labels and texts:
            option_lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts)]
            return f"{question}\n\nOptions:\n" + "\n".join(option_lines)

    return question


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """
    Trích xuất đáp án cuối từ generated text.
    Hỗ trợ: \boxed{}, <answer>...</answer>, Final Answer:, last-line fallback.
    """
    if not text:
        return ""

    # GreenMind / VietCoMath style: <answer>42</answer>
    answer_tag = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if answer_tag:
        return answer_tag[-1].strip()

    # LaTeX boxed answer: \boxed{42}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # "Final Answer: 42" hoặc "The answer is 42"
    fa = re.findall(r"(?:Final Answer|The answer is)[:\s]+(.+?)(?:\n|$)", text, re.I)
    if fa:
        return fa[-1].strip()

    # Fallback: lấy dòng cuối không rỗng
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _numeric_candidates(text: str) -> list[float]:
    """Extract likely numeric answers from a string for tolerant comparison."""
    s = text.strip()
    candidates: list[float] = []

    for frac in re.findall(r"(?<!\d)([-+]?\d+\s*/\s*\d+)(?!\d)", s):
        try:
            candidates.append(float(Fraction(frac.replace(" ", ""))))
        except (ValueError, ZeroDivisionError):
            pass

    for number in re.findall(r"(?<!\w)([-+]?\d*\.?\d+)(?!\w)", s.replace(",", "")):
        try:
            candidates.append(float(number))
        except ValueError:
            pass

    return candidates


def _extract_choice_letter(text: str) -> str | None:
    """Extract chọn đáp án dạng A/B/C/D từ text dự đoán."""
    patterns = [
        r"(?:final answer|answer is|therefore|so)\s*[:\-]?\s*\(?([A-E])\)?",
        r"\boption\s*([A-E])\b",
        r"\(([A-E])\)",
        r"\b([A-E])\b",
    ]
    upper = text.upper()
    for pat in patterns:
        match = re.search(pat, upper, re.I)
        if match:
            return match.group(1).upper()
    return None


def check_answer(predicted: str, ground_truth: str) -> bool:
    """
    So sánh đáp án. Normalize để tránh false negative.
    Có thể extend bằng math-symbolic comparison (sympy).
    """
    def normalize(s: str) -> str:
        s = s.strip().lower()
        s = s.replace(",", "").replace(" ", "")
        s = s.replace("$", "").replace("\\", "")
        return s

    p_norm = normalize(predicted)
    g_norm = normalize(str(ground_truth))
    if p_norm == g_norm:
        return True

    # Multiple-choice: letter answer (e.g., ARC-Challenge answerKey = A/B/C/D)
    if len(g_norm) == 1 and g_norm in {"a", "b", "c", "d", "e"}:
        pred_choice = _extract_choice_letter(predicted)
        if pred_choice and pred_choice.lower() == g_norm:
            return True

    # Multiple-choice: 1-indexed numeric answer (e.g., VNHSGE: "1"→A, "2"→B, ...)
    if len(g_norm) == 1 and g_norm in {"1", "2", "3", "4", "5"}:
        letter = chr(ord("a") + int(g_norm) - 1)
        pred_choice = _extract_choice_letter(predicted)
        if pred_choice and pred_choice.lower() == letter:
            return True

    p_nums = _numeric_candidates(predicted)
    g_nums = _numeric_candidates(str(ground_truth))
    if p_nums and g_nums:
        return abs(p_nums[-1] - g_nums[-1]) <= 1e-6

    return False


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    model_name: str,
    benchmark: str,
    n_wait_list: list[int],
    n_samples: int,
    output_dir: str,
    trigger: str = "Wait",
    load_in_4bit: bool = True,
    max_new_tokens: int = 2048,
    seed: int = 42,
) -> dict:
    """
    Chạy evaluation với nhiều cấu hình n_wait.

    Returns:
        results dict với accuracy tại mỗi n_wait level
    """
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=load_in_4bit)
    decoder = BudgetForcingDecoder(model, tokenizer)

    # Load benchmark
    cfg = BENCHMARK_REGISTRY[benchmark]
    if cfg.hf_name:
        dataset = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.hf_split)
    else:
        dataset = load_dataset(cfg.hf_path, split=cfg.hf_split)

    # Sample n_samples câu hỏi
    import random
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    print(f"\nEvaluating {model_name} on {benchmark} ({n_samples} samples)")
    print(f"n_wait settings: {n_wait_list}, trigger: '{trigger}'\n")

    all_results = {
        "model": model_name,
        "benchmark": benchmark,
        "trigger": trigger,
        "n_samples": n_samples,
        "run_metadata": {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "load_in_4bit": load_in_4bit,
            "runtime": {
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
            },
            "benchmark_spec": {
                "hf_path": cfg.hf_path,
                "hf_name": cfg.hf_name,
                "hf_split": cfg.hf_split,
                "question_key": cfg.question_key,
                "answer_key": cfg.answer_key,
                "choices_key": cfg.choices_key,
                "n_total": cfg.n_total,
            },
        },
        "experiments": {}
    }

    for n_wait in n_wait_list:
        print(f"\n{'='*50}")
        print(f"n_wait = {n_wait} (trigger appended {n_wait}× max)")
        print(f"{'='*50}")

        correct = 0
        run_results = []

        for i, sample in enumerate(tqdm(samples, desc=f"n_wait={n_wait}")):
            question = format_question(sample, cfg)
            ground_truth = str(sample[cfg.answer_key])

            prompt = format_prompt(question, tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            t0 = time.time()
            try:
                output = decoder.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    n_wait=n_wait,
                    trigger=trigger,
                )
                elapsed = time.time() - t0

                predicted = extract_answer(output["answer_text"])
                is_correct = check_answer(predicted, ground_truth)
                correct += int(is_correct)

                run_results.append({
                    "idx": indices[i],
                    "question": question[:100] + "...",
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": is_correct,
                    "thinking_tokens": output["thinking_tokens"],
                    "n_waits_triggered": output["n_waits_triggered"],
                    "elapsed_sec": round(elapsed, 2),
                })

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                run_results.append({"idx": indices[i], "error": str(e), "correct": False})

        accuracy = correct / n_samples
        print(f"\nAccuracy (n_wait={n_wait}): {accuracy:.1%} ({correct}/{n_samples})")

        all_results["experiments"][f"n_wait_{n_wait}"] = {
            "n_wait": n_wait,
            "accuracy": accuracy,
            "correct": correct,
            "details": run_results,
        }

    # Compute scaling metrics
    compute_levels = list(n_wait_list)
    accuracies = [
        all_results["experiments"][f"n_wait_{nw}"]["accuracy"]
        for nw in n_wait_list
    ]
    metrics = compute_all_metrics(compute_levels, accuracies)
    all_results["metrics"] = {
        "control": metrics.control,
        "scaling": round(metrics.scaling, 2),
        "performance": round(metrics.performance, 4),
        "control_note": (
            "Control requires target_budgets and actual_tokens. "
            "Current run uses n_wait sweep only."
            if metrics.control is None
            else ""
        ),
    }
    print(f"\nFinal metrics: {metrics}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{model_name.replace('/', '_')}_{benchmark}_nwait{'_'.join(map(str, n_wait_list))}.json"
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {fpath}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_output_dir = str(Path(__file__).resolve().parents[1] / "results")

    parser = argparse.ArgumentParser(description="Budget Forcing Evaluation")
    parser.add_argument("--model", default="qwen2.5-7B", choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--benchmark", default="math500",
                        choices=list(BENCHMARK_REGISTRY.keys()))
    parser.add_argument("--n_wait", nargs="+", type=int, default=[0, 1, 2, 4],
                        help="List of n_wait values to test")
    parser.add_argument("--trigger", default="Wait",
                        help="Trigger phrase to append (default: 'Wait')")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_evaluation(
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
