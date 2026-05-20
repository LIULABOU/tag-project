"""
Plot R@K vs K curves for multiple eval runs (and optionally aggregate
across seeds with error bands).

Reads `ranks.npy` from each eval out_dir and computes R@K for K in a
configurable range. Useful for visualizing the inductive-bias trade-off:
methods that integrate evidence (multi_C) outperform single-cue methods
(pure_corr) more strongly at larger K.

Usage:
  # single seed
  python -m src.eval.plot_rk_curve \
      --runs outputs/seed_0/baseline_clip_v6 \
             outputs/seed_0/eval_run_pure_corr_v6 \
             outputs/seed_0/eval_run_dcp_v6 \
             outputs/seed_0/eval_run_multi_C_v6 \
      --names CLIP-only pure_corr DCP multi_C \
      --out outputs/viz/rk_curve.png

  # multi-seed (mean ± std band)
  python -m src.eval.plot_rk_curve \
      --runs "outputs/seed_*/baseline_clip_v6" \
             "outputs/seed_*/eval_run_pure_corr_v6" \
             "outputs/seed_*/eval_run_dcp_v6" \
             "outputs/seed_*/eval_run_multi_C_v6" \
      --names CLIP-only pure_corr DCP multi_C \
      --out outputs/viz/rk_curve_meanstd.png
"""

import argparse
import glob
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


PALETTE = {
    "CLIP-only":   "#777777",
    "clip_only":   "#777777",
    "fixed":       "#9467bd",
    "pure_corr":   "#9467bd",
    "DCP":         "#d62728",
    "DCP-B":       "#d62728",
    "dcp_B":       "#d62728",
    "multi_C":     "#1f77b4",
    "multi_C+floor": "#2ca02c",
    "random":      "#cccccc",
}


def color_for(name):
    return PALETTE.get(name, None)


def expand(pattern: str) -> List[str]:
    """If pattern contains a glob char, expand it; else return [pattern]."""
    if any(ch in pattern for ch in "*?["):
        out = sorted(glob.glob(pattern))
        if not out:
            print(f"[warn] pattern matched nothing: {pattern}")
        return out
    return [pattern]


def load_ranks(eval_dir: str) -> np.ndarray:
    p = os.path.join(eval_dir, "ranks.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing ranks.npy in {eval_dir}")
    return np.load(p).astype(np.float64)


def rk_curve(ranks: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """For each K in ks, fraction of queries with rank <= K."""
    return np.array([(ranks <= k).mean() for k in ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="eval out_dirs OR glob patterns (one per method).")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/viz/rk_curve.png")
    ap.add_argument("--k_max", type=int, default=50)
    ap.add_argument("--log_x", action="store_true",
                    help="use log-x axis (clearer at small K)")
    args = ap.parse_args()

    if len(args.names) != len(args.runs):
        raise SystemExit("--names and --runs must be same length")

    ks = np.arange(1, args.k_max + 1)

    fig, ax = plt.subplots(figsize=(6, 6))

    for run_pattern, name in zip(args.runs, args.names):
        dirs = expand(run_pattern)
        if not dirs:
            continue

        curves = []
        for d in dirs:
            try:
                ranks = load_ranks(d)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            curves.append(rk_curve(ranks, ks))

        if not curves:
            continue

        curves = np.stack(curves, axis=0)            # [n_seeds, K]
        mean = curves.mean(axis=0)
        n_seeds = curves.shape[0]

        c = color_for(name)
        if n_seeds >= 2:
            std = curves.std(axis=0, ddof=1)
            ax.fill_between(ks, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)
        ax.plot(ks, mean, label=f"{name} (n={n_seeds})", color=c,
                linewidth=2.0, marker="o", markersize=3.5)

    ax.set_xlabel(r"$K$", fontsize=13)
    ax.set_ylabel(r"R@$K$", fontsize=13)
    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlim(1, args.k_max)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
