"""
Download and inspect Vietnamese benchmarks.

Validates that datasets are accessible on HuggingFace and that
field names match what BENCHMARK_REGISTRY expects.

Usage:
    python experiments/data/download_vi_benchmarks.py
    python experiments/data/download_vi_benchmarks.py --benchmark vi_gsm8k
    python experiments/data/download_vi_benchmarks.py --benchmark vimmlu
    python experiments/data/download_vi_benchmarks.py --n_samples 5
"""

from __future__ import annotations

import argparse
from datasets import load_dataset

BENCHMARKS = {
    "vi_gsm8k": {
        "hf_path": "hllj/vi_gsm8k",
        "hf_name": None,
        "split": "test",
        "expected_fields": ["question", "answer", "index", "explanation"],
        "description": "Multilingual GSM8K — Vietnamese split (250 problems)",
    },
    "vimmlu": {
        "hf_path": "tridm/VMLU",
        "hf_name": None,
        "split": "test",
        "expected_fields": ["question", "answer", "choices", "id"],
        "description": "Vietnamese MMLU — multi-domain knowledge",
    },
    "vnhsge": {
        "hf_path": "roshansk23/Vietnam_HighSchool_Exam_Dataset",
        "hf_name": None,
        "split": "train",
        "expected_fields": ["question", "options", "answer"],
        "description": "VNHSGE - Vietnamese High School Exam Dataset",
    }
}


def check_benchmark(name: str, cfg: dict, n_samples: int = 3) -> bool:
    """Download and validate a single benchmark. Returns True if OK."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"  {cfg['description']}")
    print(f"  Source: {cfg['hf_path']}" + (f" / {cfg['hf_name']}" if cfg['hf_name'] else ""))
    print(f"{'='*60}")

    try:
        if cfg["hf_name"]:
            dataset = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
        else:
            dataset = load_dataset(cfg["hf_path"], split=cfg["split"])
    except Exception as e:
        print(f"[FAIL] Could not load dataset: {e}")
        return False

    print(f"[OK]  Loaded. Size: {len(dataset)} samples")
    print(f"      Fields: {list(dataset.features.keys())}")

    # Check expected fields
    missing = [f for f in cfg["expected_fields"] if f not in dataset.features]
    if missing:
        print(f"[WARN] Missing expected fields: {missing}")
        print(f"       Available: {list(dataset.features.keys())}")
    else:
        print(f"[OK]  All expected fields present: {cfg['expected_fields']}")

    # Print sample rows
    n = min(n_samples, len(dataset))
    print(f"\n--- First {n} samples ---")
    for i in range(n):
        row = dataset[i]
        print(f"\nSample {i+1}:")
        for key in list(row.keys())[:8]:  # show up to 8 fields
            val = row[key]
            if isinstance(val, str) and len(val) > 100:
                val = val[:100] + "..."
            print(f"  {key}: {val!r}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download and validate Vietnamese benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Which benchmark to check (default: all)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Number of sample rows to print per benchmark",
    )
    args = parser.parse_args()

    if args.benchmark == "all":
        targets = BENCHMARKS
    else:
        targets = {args.benchmark: BENCHMARKS[args.benchmark]}

    results = {}
    for name, cfg in targets.items():
        ok = check_benchmark(name, cfg, n_samples=args.n_samples)
        results[name] = "OK" if ok else "FAIL"

    print(f"\n{'='*60}")
    print("Summary:")
    for name, status in results.items():
        marker = "[OK]  " if status == "OK" else "[FAIL]"
        print(f"  {marker} {name}")

    if "FAIL" in results.values():
        print("\nSome benchmarks failed. Check HF access or field names.")
        raise SystemExit(1)
    else:
        print("\nAll benchmarks validated successfully.")


if __name__ == "__main__":
    main()
