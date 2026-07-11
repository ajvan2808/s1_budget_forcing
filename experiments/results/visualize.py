"""
visualize.py — Generate publication figures for the Vietnamese Budget Forcing study.

Reads all JSON result files under experiments/results/vi_*/
and produces three figures + summary_vi.csv.

Usage:
    python experiments/results/visualize.py
    python experiments/results/visualize.py --results_dir path/to/results --out_dir path/to/figs

Outputs (saved to --out_dir, default: experiments/results/figures/):
    fig1_accuracy_vs_nwait.png     — main result: accuracy by model × benchmark × n_wait
    fig2_thinking_tokens.png       — mechanism: thinking tokens by model × benchmark × n_wait
    fig3_bf_delta.png              — summary: BF effect (nwait=2 − nwait=0) per model × benchmark
    summary_vi.csv                 — flat table of all results (source of truth for report)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/Kaggle environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Aesthetics ────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "qwen2.5-3B",
    "r1-distill-7B",
    "seallm-7b",
    "vinallama-7b",
    "vistral-7b",
    "greenmind-14b-r1",
]

MODEL_SHORT = {
    "qwen2.5-3B":      "qwen-3B",
    "r1-distill-7B":   "r1-7B",
    "seallm-7b":       "seallm",
    "vinallama-7b":    "vinallama",
    "vistral-7b":      "vistral",
    "greenmind-14b-r1":"greenmind",
}

# Model family annotation for grouping
MODEL_FAMILY = {
    "qwen2.5-3B":      "multilingual",
    "r1-distill-7B":   "reasoning-EN",
    "seallm-7b":       "Vi-specialized",
    "vinallama-7b":    "Vi-specialized",
    "vistral-7b":      "Vi-specialized",
    "greenmind-14b-r1":"Vi-reasoning",
}

MODEL_COLOR = {
    "qwen2.5-3B":      "#378ADD",
    "r1-distill-7B":   "#7F77DD",
    "seallm-7b":       "#639922",
    "vinallama-7b":    "#BA7517",
    "vistral-7b":      "#D85A30",
    "greenmind-14b-r1":"#208080",
}

BENCH_ORDER = ["vi_gsm8k", "vimmlu", "vnhsge"]
BENCH_LABEL = {
    "vi_gsm8k": "vi_gsm8k\n(math)",
    "vimmlu":   "vimmlu\n(factual MC)",
    "vnhsge":   "vnhsge\n(HS exam MC)",
}
BENCH_COLOR = {
    "vi_gsm8k": "#185FA5",
    "vimmlu":   "#BA7517",
    "vnhsge":   "#0F6E56",
}

MARKER = ["o", "s", "D", "^", "v", "P"]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "grid.linestyle":  "--",
    "figure.dpi":      150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Scan all vi_*/model__benchmark__nwaitN.json files.
    When (model, benchmark, n_wait) appears in multiple run dirs, keep
    the file with the highest n_samples (most complete run).

    Returns a DataFrame with columns:
        model, benchmark, n_wait, n_samples, accuracy, avg_thinking_tokens,
        extraction_failures, run_dir, timestamp_utc
    """
    rows: dict[tuple, dict] = {}  # key = (model, benchmark, n_wait)

    for jf in sorted(results_dir.glob("vi_*/*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if d.get("accuracy") is None:
            continue  # skip failed runs

        key = (d["model"], d["benchmark"], d["n_wait"])
        existing = rows.get(key)
        if existing is None or d.get("n_samples", 0) > existing.get("n_samples", 0):
            rows[key] = {
                "model":               d["model"],
                "benchmark":           d["benchmark"],
                "n_wait":              d["n_wait"],
                "n_samples":           d.get("n_samples", 0),
                "accuracy":            round(d["accuracy"] * 100, 2),   # → %
                "avg_thinking_tokens": d.get("avg_thinking_tokens", 0),
                "extraction_failures": d.get("extraction_failures", 0),
                "run_dir":             d.get("run_dir", ""),
                "timestamp_utc":       d.get("timestamp_utc", ""),
            }

    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        raise RuntimeError(f"No valid JSON results found under {results_dir}")

    # Sort for consistent ordering
    df["_model_order"] = df["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    ).fillna(99)
    df["_bench_order"] = df["benchmark"].map(
        {b: i for i, b in enumerate(BENCH_ORDER)}
    ).fillna(99)
    df = df.sort_values(["_model_order", "_bench_order", "n_wait"]).drop(
        columns=["_model_order", "_bench_order"]
    ).reset_index(drop=True)

    return df


# ── Figure 1: Accuracy vs n_wait ──────────────────────────────────────────────

def plot_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    """
    3 subplots (one per benchmark), lines = models.
    Each model's line shows accuracy at n_wait ∈ {0, 1, 2}.
    """
    benches   = [b for b in BENCH_ORDER if b in df["benchmark"].unique()]
    models    = [m for m in MODEL_ORDER  if m in df["model"].unique()]
    n_benches = len(benches)

    fig, axes = plt.subplots(1, n_benches, figsize=(4.8 * n_benches, 4.0),
                             sharey=True)
    if n_benches == 1:
        axes = [axes]

    fig.suptitle(
        "Accuracy vs n_wait  (Budget Forcing sweep)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, bench in zip(axes, benches):
        sub = df[df["benchmark"] == bench]
        for mi, model in enumerate(models):
            mdf = sub[sub["model"] == model].sort_values("n_wait")
            if mdf.empty:
                continue
            xs = mdf["n_wait"].tolist()
            ys = mdf["accuracy"].tolist()
            ax.plot(
                xs, ys,
                color=MODEL_COLOR.get(model, "#888"),
                marker=MARKER[mi % len(MARKER)],
                linewidth=1.8,
                markersize=6,
                label=MODEL_SHORT.get(model, model),
            )
        ax.set_title(BENCH_LABEL.get(bench, bench), fontsize=10)
        ax.set_xlabel("n_wait", fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.0f}%"
        ))

    axes[0].set_ylabel("Accuracy (%)", fontsize=9)

    # Legend below subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect any extra handles from other subplots
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(len(labels), 6),
        fontsize=8.5, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    # Note on n_samples
    n_note = df.groupby("model")["n_samples"].first()
    note_parts = [f"{MODEL_SHORT.get(m,m)}=n{n}" for m, n in n_note.items()]
    fig.text(
        0.5, -0.13, "Sample sizes: " + ", ".join(note_parts),
        ha="center", fontsize=7.5, color="#555",
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Thinking tokens vs n_wait ───────────────────────────────────────

def plot_thinking_tokens(df: pd.DataFrame, out_path: Path) -> None:
    """
    3 subplots (one per benchmark), lines = models.
    Shows that BF mechanically increases token budget even when accuracy does not improve.
    """
    benches   = [b for b in BENCH_ORDER if b in df["benchmark"].unique()]
    models    = [m for m in MODEL_ORDER  if m in df["model"].unique()]
    n_benches = len(benches)

    fig, axes = plt.subplots(1, n_benches, figsize=(4.8 * n_benches, 4.0),
                             sharey=False)
    if n_benches == 1:
        axes = [axes]

    fig.suptitle(
        "Avg Thinking Tokens vs n_wait  (BF token budget control)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, bench in zip(axes, benches):
        sub = df[df["benchmark"] == bench]
        for mi, model in enumerate(models):
            mdf = sub[sub["model"] == model].sort_values("n_wait")
            if mdf.empty:
                continue
            xs = mdf["n_wait"].tolist()
            ys = mdf["avg_thinking_tokens"].tolist()
            ax.plot(
                xs, ys,
                color=MODEL_COLOR.get(model, "#888"),
                marker=MARKER[mi % len(MARKER)],
                linewidth=1.8,
                markersize=6,
                label=MODEL_SHORT.get(model, model),
            )
        ax.set_title(BENCH_LABEL.get(bench, bench), fontsize=10)
        ax.set_xlabel("n_wait", fontsize=9)
        ax.set_xticks([0, 1, 2])

    axes[0].set_ylabel("Avg thinking tokens", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(len(labels), 6),
        fontsize=8.5, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: BF delta bar chart ──────────────────────────────────────────────

def plot_bf_delta(df: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: x = models, hue = benchmark.
    Bar height = accuracy(nwait=2) − accuracy(nwait=0) in percentage points.
    Positive = BF helps, negative = BF hurts.
    """
    models    = [m for m in MODEL_ORDER  if m in df["model"].unique()]
    benches   = [b for b in BENCH_ORDER  if b in df["benchmark"].unique()]

    n_models  = len(models)
    n_benches = len(benches)
    width     = 0.22
    x         = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_models), 4.2))
    fig.suptitle(
        "BF Effect — Accuracy Delta  (nwait=2 − nwait=0)",
        fontsize=12, fontweight="bold",
    )

    for bi, bench in enumerate(benches):
        deltas = []
        for model in models:
            sub = df[(df["model"] == model) & (df["benchmark"] == bench)]
            nw0 = sub[sub["n_wait"] == 0]["accuracy"].values
            nw2 = sub[sub["n_wait"] == 2]["accuracy"].values
            if len(nw0) and len(nw2):
                deltas.append(float(nw2[0] - nw0[0]))
            else:
                deltas.append(float("nan"))

        offset = (bi - (n_benches - 1) / 2) * (width + 0.02)
        bars = ax.bar(
            x + offset, deltas,
            width=width,
            color=BENCH_COLOR.get(bench, "#888"),
            alpha=0.85,
            label=bench,
            edgecolor="white",
            linewidth=0.5,
        )

        # Value labels on bars (skip NaN)
        for bar, val in zip(bars, deltas):
            if not np.isnan(val):
                va = "bottom" if val >= 0 else "top"
                ay = bar.get_height() + (0.4 if val >= 0 else -0.4)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ay,
                    f"{val:+.0f}pp",
                    ha="center", va=va, fontsize=7, color="#333",
                )

    # Zero line
    ax.axhline(0, color="#333", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_SHORT.get(m, m) for m in models], fontsize=9
    )
    ax.set_ylabel("Accuracy delta (pp)", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:+.0f}pp"
    ))

    legend_patches = [
        mpatches.Patch(color=BENCH_COLOR[b], label=b) for b in benches
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, frameon=False,
              loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a clean summary CSV — this is the source of truth for the report.
    Columns: model, benchmark, n_wait, n_samples, accuracy, avg_thinking_tokens,
             extraction_failures, bf_delta_from_nw0
    """
    # Compute BF delta (vs nwait=0) for each (model, benchmark)
    pivot = df.pivot_table(
        index=["model", "benchmark"], columns="n_wait", values="accuracy"
    ).reset_index()
    pivot.columns.name = None
    # Rename numeric columns
    for nw in [0, 1, 2]:
        if nw in pivot.columns:
            pivot = pivot.rename(columns={nw: f"acc_nw{nw}"})

    merged = df.merge(pivot, on=["model", "benchmark"], how="left")

    if "acc_nw0" in merged.columns and "acc_nw2" in merged.columns:
        merged["bf_delta_pp"] = (merged["acc_nw2"] - merged["acc_nw0"]).round(2)
    else:
        merged["bf_delta_pp"] = float("nan")

    out_cols = [
        "model", "benchmark", "n_wait", "n_samples",
        "accuracy", "avg_thinking_tokens", "extraction_failures",
        "acc_nw0", "acc_nw2", "bf_delta_pp",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate BF experiment figures and summary CSV"
    )
    default_results = Path(__file__).parent
    default_out     = Path(__file__).parent / "figures"

    parser.add_argument(
        "--results_dir", type=Path, default=default_results,
        help="Directory containing vi_*/ subdirs with JSON result files",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=default_out,
        help="Output directory for figures and CSV",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)
    print(f"  {len(df)} result rows loaded ({df['model'].nunique()} models, "
          f"{df['benchmark'].nunique()} benchmarks)")

    # Print quick summary table
    print("\nSummary (accuracy %):")
    pt = df.pivot_table(
        index=["model", "benchmark"], columns="n_wait",
        values="accuracy", aggfunc="first"
    )
    print(pt.to_string())
    print()

    print("Generating figures...")
    plot_accuracy(df,          args.out_dir / "fig1_accuracy_vs_nwait.png")
    plot_thinking_tokens(df,   args.out_dir / "fig2_thinking_tokens.png")
    plot_bf_delta(df,          args.out_dir / "fig3_bf_delta.png")
    export_csv(df,             args.out_dir / "summary_vi.csv")

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
