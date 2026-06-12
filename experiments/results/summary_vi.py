"""
Aggregate Vietnamese Budget Forcing results into summary_vi.csv + summary_vi.md.

Reads all JSON files produced by run_eval_vi.py in a results directory,
computes per-model scaling metrics, and outputs two files:
  summary_vi.csv   machine-readable, matches column schema in taskboard
  summary_vi.md    human-readable markdown table for the report

Column schema:
  model, benchmark, language, n_wait,
  n_samples, accuracy, scaling, performance,
  avg_thinking_tokens,
  extraction_failures, cuda_available, mps_available,
  run_dir, timestamp_utc

Usage:
    # From timestamped run dir
    python experiments/results/summary_vi.py \
        --results_dir experiments/results/vi_20260611_120000

    # From all vi_* subdirs under experiments/results/
    python experiments/results/summary_vi.py \
        --results_dir experiments/results \
        --glob "vi_*"

    # Print only (no file write)
    python experiments/results/summary_vi.py \
        --results_dir experiments/results/vi_20260611_120000 \
        --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_scaling_slope(n_wait_values: List[int], accuracies: List[float]) -> Optional[float]:
    """
    Linear regression slope of accuracy vs n_wait.
    Positive = accuracy improves with more thinking. Returns None if < 2 points.
    """
    pairs = [(n, a) for n, a in zip(n_wait_values, accuracies) if a is not None]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denom = sum((x - x_mean) ** 2 for x in xs)
    return round(numer / denom, 4) if denom != 0 else 0.0


def compute_performance(accuracies: List[float]) -> Optional[float]:
    """Max accuracy across n_wait values."""
    valid = [a for a in accuracies if a is not None]
    return round(max(valid), 4) if valid else None


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json_results(results_dir: Path, glob_pattern: str = "vi_*") -> List[dict]:
    """Load all JSON result files from a results directory."""
    json_files = []

    # Check if results_dir itself contains JSONs
    direct_jsons = list(results_dir.glob("*.json"))
    if direct_jsons:
        json_files.extend(direct_jsons)
    else:
        # Walk subdirectories matching glob
        for subdir in sorted(results_dir.glob(glob_pattern)):
            if subdir.is_dir():
                json_files.extend(sorted(subdir.glob("*.json")))

    if not json_files:
        print(f"[WARN] No JSON files found in {results_dir}")
        return []

    records = []
    for jf in json_files:
        if jf.name == "meta.json":
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = str(jf)
            records.append(data)
        except Exception as e:
            print(f"[WARN] Could not parse {jf}: {e}")

    print(f"[summary_vi] Loaded {len(records)} result files from {results_dir}")
    return records


# ── Row building ──────────────────────────────────────────────────────────────

def build_rows(records: List[dict]) -> List[Dict[str, Any]]:
    """
    Convert raw JSON records into summary rows.
    Groups BF_only records by (model, benchmark, condition) to compute scaling slope.
    """
    rows = []
    # Group records by (model, benchmark, condition) for scaling computation
    from collections import defaultdict
    groups: dict = defaultdict(list)

    for rec in records:
        model = rec.get("model", "unknown")
        benchmark = rec.get("benchmark", "unknown")
        condition = rec.get("condition", "unknown")
        key = (model, benchmark, condition)
        groups[key].append(rec)

    for (model, benchmark, condition), group_recs in groups.items():
        # Sort by n_wait for scaling computation
        group_recs = sorted(group_recs, key=lambda r: r.get("n_wait", 0))
        nwaits = [r.get("n_wait", 0) for r in group_recs]
        accs = [r.get("accuracy") for r in group_recs]

        scaling = compute_scaling_slope(nwaits, [a for a in accs if a is not None])
        performance = compute_performance([a for a in accs if a is not None])

        for rec in group_recs:
            runtime = rec.get("runtime", {})
            rows.append({
                "model": model,
                "benchmark": benchmark,
                "language": rec.get("language", "vi"),
                "n_wait": rec.get("n_wait", 0),
                "n_samples": rec.get("n_samples", 0),
                "accuracy": rec.get("accuracy"),
                "scaling": scaling,
                "performance": performance,
                "avg_thinking_tokens": rec.get("avg_thinking_tokens", 0),
                "extraction_failures": rec.get("extraction_failures", 0),
                "cuda_available": runtime.get("cuda_available", False),
                "mps_available": runtime.get("mps_available", False),
                "run_dir": rec.get("run_dir", ""),
                "timestamp_utc": rec.get("timestamp_utc", ""),
            })

    return rows


# ── Output writers ────────────────────────────────────────────────────────────

COLUMNS = [
    "model", "benchmark", "language", "n_wait",
    "n_samples", "accuracy", "scaling", "performance",
    "avg_thinking_tokens",
    "extraction_failures", "cuda_available", "mps_available",
    "run_dir", "timestamp_utc",
]


def write_csv(rows: List[dict], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary_vi] CSV written: {out_path}")


def write_markdown(rows: List[dict], out_path: Path):
    """Write a human-readable markdown table."""
    # Group for display: one table per benchmark
    from collections import defaultdict
    by_benchmark: dict = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)

    lines = ["# Vietnamese Budget Forcing Results\n"]
    lines.append(f"Generated: {__import__('datetime').datetime.utcnow().isoformat(timespec='seconds')}Z\n")

    display_cols = ["model", "n_wait", "n_samples",
                    "accuracy", "scaling", "performance",
                    "avg_thinking_tokens", "extraction_failures"]

    for bench, bench_rows in sorted(by_benchmark.items()):
        lines.append(f"\n## {bench}\n")
        bench_rows = sorted(bench_rows, key=lambda r: (r["model"], r["n_wait"]))

        # Header
        header = "| " + " | ".join(display_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        lines.append(header)
        lines.append(sep)

        for row in bench_rows:
            acc = row.get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "--"
            scaling = row.get("scaling")
            scaling_str = f"{scaling:+.4f}" if scaling is not None else "--"
            perf = row.get("performance")
            perf_str = f"{perf:.1%}" if perf is not None else "--"

            cells = [
                str(row.get("model", "")),
                str(row.get("n_wait", "")),
                str(row.get("n_samples", "")),
                acc_str,
                scaling_str,
                perf_str,
                str(row.get("avg_thinking_tokens", "")),
                str(row.get("extraction_failures", "")),
            ]
            lines.append("| " + " | ".join(cells) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary_vi] Markdown written: {out_path}")


def print_summary(rows: List[dict]):
    """Print a quick human-readable summary to stdout."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    from collections import defaultdict
    by_bench: dict = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)

    for bench, bench_rows in sorted(by_bench.items()):
        print(f"\n{bench}:")
        bench_rows = sorted(bench_rows, key=lambda r: (r["model"], r["n_wait"]))
        for row in bench_rows:
            acc = row.get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "--"
            print(
                f"  {row['model']:20s} | "
                f"n_wait={row['n_wait']} | acc={acc_str} | "
                f"thinks={row.get('avg_thinking_tokens', 0):.0f} tok | "
                f"fails={row['extraction_failures']}"
            )
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate Vietnamese Budget Forcing results")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing JSON result files (or parent of vi_* subdirs)",
    )
    parser.add_argument(
        "--glob",
        default="vi_*",
        help="Glob pattern for subdirectory matching (default: vi_*)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save summary_vi.csv and summary_vi.md (default: same as results_dir)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summary without writing files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] results_dir not found: {results_dir}")
        raise SystemExit(1)

    records = load_json_results(results_dir, glob_pattern=args.glob)
    if not records:
        print("[ERROR] No result files found. Check --results_dir.")
        raise SystemExit(1)

    rows = build_rows(records)
    print_summary(rows)

    if not args.dry_run:
        out_dir = Path(args.output_dir) if args.output_dir else results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out_dir / "summary_vi.csv")
        write_markdown(rows, out_dir / "summary_vi.md")


if __name__ == "__main__":
    main()
