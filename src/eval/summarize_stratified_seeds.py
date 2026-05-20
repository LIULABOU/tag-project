"""
Summarize stratified evaluation results across seeds and plot length buckets.

This script expects each seed directory to contain:

  stratified_v6/stratified_metrics.csv
  stratified_v6/dialogue_strata.csv
  stratified_v6/stratified_metrics.json

Typical usage from the project root:

  python -m src.eval.summarize_stratified_seeds \
      --seed_dirs outputs/seed_0 outputs/seed_1 outputs/seed_2 \
      --models fixed multi_C pure_corr

Outputs:
  - outputs/stratified_condition_table_meanstd.csv:
      Condition, Model, R@1, R@5, R@10, MRR
  - outputs/stratified_condition_drop_meanstd.csv:
      paired drops across seeds, e.g. short - long
  - outputs/stratified_condition_drop_pct_meanstd.csv:
      paired percentage drops across seeds, e.g. (short - long) / short
  - outputs/viz/condition_performance_change.png/pdf:
      how much each model drops from short to long and low to high repair
  - outputs/viz/condition_change_<metric>.png/pdf:
      bar chart with an overlaid percentage-drop line for one metric
  - outputs/viz/condition_absolute_metrics.png/pdf:
      absolute performance under each condition
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


TABLE_METRICS = ("R@1", "R@5", "R@10", "MRR")
METRICS = (*TABLE_METRICS, "median_rank")
DEFAULT_MODELS = ("fixed", "multi_C", "pure_corr")
DEFAULT_STRATIFIED_SUBDIR = "stratified_v6"
DEFAULT_SEED_DIRS = ("outputs/seed_0", "outputs/seed_1", "outputs/seed_2")
CONDITIONS = (
    ("short", "length", "short"),
    ("long", "length", "long"),
    ("low_repair", "repair_rate", "low_repair"),
    ("high_repair", "repair_rate", "high_repair"),
)
CONTRASTS = (
    ("short_to_long_drop", "length", "short", "long"),
    ("low_to_high_repair_drop", "repair_rate", "low_repair", "high_repair"),
)


def _metric_fmt(metric: str) -> str:
    if metric.endswith("_pct"):
        return "{:.2f}"
    return "{:.1f}" if metric == "median_rank" else "{:.4f}"


def format_mean_std(mean: float, std: float, n: int, metric: str) -> str:
    if n == 0 or np.isnan(mean):
        return "-"
    fmt = _metric_fmt(metric)
    if n == 1 or np.isnan(std):
        return fmt.format(float(mean))
    return f"{fmt.format(float(mean))} ± {fmt.format(float(std))}"


def format_pct_mean_std(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "-"
    if n == 1 or np.isnan(std):
        return f"{float(mean):.2f}%"
    return f"{float(mean):.2f}% ± {float(std):.2f}%"


def parse_float(value: str) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def seed_name(seed_dir: Path) -> str:
    return seed_dir.name


def read_seed_metrics(
    seed_dirs: Sequence[str],
    stratified_subdir: str,
) -> List[dict]:
    rows = []
    for seed_dir_raw in seed_dirs:
        seed_dir = Path(seed_dir_raw)
        csv_path = seed_dir / stratified_subdir / "stratified_metrics.csv"
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {
                    "seed": seed_name(seed_dir),
                    "seed_dir": str(seed_dir),
                    "run": row["run"],
                    "bucket_type": row["bucket_type"],
                    "bucket": row["bucket"],
                    "N_queries": parse_float(row.get("N_queries")),
                }
                for metric in METRICS:
                    parsed[metric] = parse_float(row.get(metric))
                rows.append(parsed)
    return rows


def summarize_rows(rows: Iterable[dict]) -> Dict[Tuple[str, str, str], dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)

    for row in rows:
        key = (row["run"], row["bucket_type"], row["bucket"])
        seeds[key].add(row["seed"])
        grouped[key]["N_queries"].append(row["N_queries"])
        for metric in METRICS:
            grouped[key][metric].append(row[metric])

    summary = {}
    for key, cols in grouped.items():
        out = {"n_seeds": len(seeds[key])}
        for col, values in cols.items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                out[f"{col}_mean"] = np.nan
                out[f"{col}_std"] = np.nan
            else:
                out[f"{col}_mean"] = float(arr.mean())
                out[f"{col}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
        summary[key] = out
    return summary


def sort_key(item: Tuple[Tuple[str, str, str], Mapping[str, float]]) -> Tuple[str, int, str]:
    (run, bucket_type, bucket), _ = item
    bucket_type_order = {"length": 0, "repair_rate": 1, "quadrant": 2}
    bucket_order = {
        "short": 0,
        "long": 1,
        "low_repair": 2,
        "high_repair": 3,
        "short+low_repair": 4,
        "short+high_repair": 5,
        "long+low_repair": 6,
        "long+high_repair": 7,
    }
    return (run, bucket_type_order.get(bucket_type, 99), bucket_order.get(bucket, bucket))


def write_numeric_csv(summary: Mapping[Tuple[str, str, str], dict], out_path: Path) -> None:
    fields = ["run", "bucket_type", "bucket", "n_seeds"]
    for col in ("N_queries", *METRICS):
        fields.extend([f"{col}_mean", f"{col}_std"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (run, bucket_type, bucket), stats in sorted(summary.items(), key=sort_key):
            row = {
                "run": run,
                "bucket_type": bucket_type,
                "bucket": bucket,
                "n_seeds": stats["n_seeds"],
            }
            for col in ("N_queries", *METRICS):
                row[f"{col}_mean"] = stats.get(f"{col}_mean", np.nan)
                row[f"{col}_std"] = stats.get(f"{col}_std", np.nan)
            writer.writerow(row)


def write_pretty_csv(summary: Mapping[Tuple[str, str, str], dict], out_path: Path) -> None:
    fields = ["run", "bucket_type", "bucket", "n_seeds", "N_queries", *METRICS]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for (run, bucket_type, bucket), stats in sorted(summary.items(), key=sort_key):
            row = [run, bucket_type, bucket, stats["n_seeds"]]
            row.append(format_mean_std(
                stats.get("N_queries_mean", np.nan),
                stats.get("N_queries_std", np.nan),
                stats["n_seeds"],
                "median_rank",
            ))
            for metric in METRICS:
                row.append(format_mean_std(
                    stats.get(f"{metric}_mean", np.nan),
                    stats.get(f"{metric}_std", np.nan),
                    stats["n_seeds"],
                    metric,
                ))
            writer.writerow(row)


def write_json(summary: Mapping[Tuple[str, str, str], dict], out_path: Path) -> None:
    payload = []
    for (run, bucket_type, bucket), stats in sorted(summary.items(), key=sort_key):
        payload.append({
            "run": run,
            "bucket_type": bucket_type,
            "bucket": bucket,
            **stats,
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_condition_table(
    summary: Mapping[Tuple[str, str, str], dict],
    models: Sequence[str],
    out_path: Path,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Model", *TABLE_METRICS])
        for condition, bucket_type, bucket in CONDITIONS:
            for model in models:
                stats = summary.get((model, bucket_type, bucket), {})
                row = [condition, model]
                for metric in TABLE_METRICS:
                    row.append(format_mean_std(
                        stats.get(f"{metric}_mean", np.nan),
                        stats.get(f"{metric}_std", np.nan),
                        stats.get("n_seeds", 0),
                        metric,
                    ))
                writer.writerow(row)


def summarize_condition_drops(
    rows: Iterable[dict],
    models: Sequence[str],
) -> Dict[Tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        indexed[(row["seed"], row["run"], row["bucket_type"], row["bucket"])] = row

    seeds = sorted({row["seed"] for row in rows})
    drops = {}
    for contrast, bucket_type, easy_bucket, hard_bucket in CONTRASTS:
        for model in models:
            stats = {}
            for metric in TABLE_METRICS:
                abs_vals = []
                pct_vals = []
                for seed in seeds:
                    easy = indexed.get((seed, model, bucket_type, easy_bucket))
                    hard = indexed.get((seed, model, bucket_type, hard_bucket))
                    if easy is None or hard is None:
                        continue
                    easy_value = easy.get(metric, np.nan)
                    hard_value = hard.get(metric, np.nan)
                    if np.isnan(easy_value) or np.isnan(hard_value):
                        continue
                    abs_drop = float(easy_value - hard_value)
                    abs_vals.append(abs_drop)
                    if easy_value != 0:
                        pct_vals.append(float(abs_drop / easy_value * 100.0))

                abs_arr = np.asarray(abs_vals, dtype=np.float64)
                pct_arr = np.asarray(pct_vals, dtype=np.float64)
                stats[f"{metric}_abs_mean"] = (
                    float(abs_arr.mean()) if len(abs_arr) else np.nan
                )
                stats[f"{metric}_abs_std"] = (
                    float(abs_arr.std(ddof=1)) if len(abs_arr) > 1 else np.nan
                )
                stats[f"{metric}_abs_n"] = int(len(abs_arr))
                stats[f"{metric}_pct_mean"] = (
                    float(pct_arr.mean()) if len(pct_arr) else np.nan
                )
                stats[f"{metric}_pct_std"] = (
                    float(pct_arr.std(ddof=1)) if len(pct_arr) > 1 else np.nan
                )
                stats[f"{metric}_pct_n"] = int(len(pct_arr))
            drops[(contrast, model)] = stats
    return drops


def write_condition_drop_table(
    drops: Mapping[Tuple[str, str], dict],
    out_path: Path,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Contrast", "Model", *TABLE_METRICS])
        for contrast, _, _, _ in CONTRASTS:
            for (this_contrast, model), stats in sorted(drops.items()):
                if this_contrast != contrast:
                    continue
                row = [contrast, model]
                for metric in TABLE_METRICS:
                    row.append(format_mean_std(
                        stats.get(f"{metric}_abs_mean", np.nan),
                        stats.get(f"{metric}_abs_std", np.nan),
                        stats.get(f"{metric}_abs_n", 0),
                        metric,
                    ))
                writer.writerow(row)


def write_condition_drop_pct_table(
    drops: Mapping[Tuple[str, str], dict],
    out_path: Path,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Contrast", "Model", *TABLE_METRICS])
        for contrast, _, _, _ in CONTRASTS:
            for (this_contrast, model), stats in sorted(drops.items()):
                if this_contrast != contrast:
                    continue
                row = [contrast, model]
                for metric in TABLE_METRICS:
                    row.append(format_pct_mean_std(
                        stats.get(f"{metric}_pct_mean", np.nan),
                        stats.get(f"{metric}_pct_std", np.nan),
                        stats.get(f"{metric}_pct_n", 0),
                    ))
                writer.writerow(row)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it or rerun with --no_plots."
        ) from exc
    return plt


def get_stat(
    summary: Mapping[Tuple[str, str, str], dict],
    run: str,
    bucket_type: str,
    bucket: str,
    metric: str,
) -> Tuple[float, float]:
    stats = summary.get((run, bucket_type, bucket), {})
    return (
        float(stats.get(f"{metric}_mean", np.nan)),
        float(stats.get(f"{metric}_std", np.nan)),
    )


def plot_condition_absolute_metrics(
    summary: Mapping[Tuple[str, str, str], dict],
    models: Sequence[str],
    metrics: Sequence[str],
    viz_dir: Path,
    title: str,
) -> None:
    plt = require_matplotlib()

    colors = {
        "short": "#4C78A8",
        "long": "#F58518",
    }
    x = np.arange(len(models), dtype=np.float64)
    width = 0.36

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.6 * len(metrics), 4.2),
        sharex=False,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for offset, bucket in ((-width / 2, "short"), (width / 2, "long")):
            means = [get_stat(summary, model, "length", bucket, metric)[0] for model in models]
            stds = [get_stat(summary, model, "length", bucket, metric)[1] for model in models]
            yerr = [0.0 if np.isnan(v) else v for v in stds]
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=yerr,
                capsize=4,
                label=bucket,
                color=colors[bucket],
                edgecolor="#222222",
                linewidth=0.7,
                alpha=0.92,
            )
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        if metric != "median_rank":
            ax.set_ylim(bottom=0)
        ax.set_ylabel("mean across seeds")

    axes[0].legend(title="dialogue length", frameon=False)
    fig.suptitle(title, fontsize=13, y=1.03)
    for ext in ("png", "pdf"):
        fig.savefig(viz_dir / f"condition_absolute_metrics.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_condition_performance_change(
    drops: Mapping[Tuple[str, str], dict],
    models: Sequence[str],
    metrics: Sequence[str],
    viz_dir: Path,
    title: str,
) -> None:
    plt = require_matplotlib()

    fig, axes = plt.subplots(
        1,
        len(CONTRASTS),
        figsize=(5.4 * len(CONTRASTS), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(CONTRASTS) == 1:
        axes = [axes]

    x = np.arange(len(models), dtype=np.float64)
    width = 0.78 / max(1, len(metrics))
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    titles = {
        "short_to_long_drop": "Drop: short - long",
        "low_to_high_repair_drop": "Drop: low repair - high repair",
    }

    for ax, (contrast, _, _, _) in zip(axes, CONTRASTS):
        for i, metric in enumerate(metrics):
            means = [
                drops.get((contrast, model), {}).get(f"{metric}_abs_mean", np.nan)
                for model in models
            ]
            stds = [
                drops.get((contrast, model), {}).get(f"{metric}_abs_std", np.nan)
                for model in models
            ]
            yerr = [0.0 if np.isnan(v) else v for v in stds]
            ax.bar(
                x + offsets[i],
                means,
                width=width * 0.9,
                yerr=yerr,
                capsize=4,
                label=metric,
                color=palette[i % len(palette)],
                edgecolor="#222222",
                linewidth=0.7,
                alpha=0.92,
            )
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(titles.get(contrast, contrast))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("performance drop, mean ± std across seeds")
    axes[-1].legend(frameon=False, title="metric")
    fig.suptitle(title, fontsize=13, y=1.03)

    for ext in ("png", "pdf"):
        fig.savefig(viz_dir / f"condition_performance_change.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def safe_metric_name(metric: str) -> str:
    return metric.replace("@", "").replace("/", "_").replace(" ", "_")


def plot_metric_combo_change(
    summary: Mapping[Tuple[str, str, str], dict],
    drops: Mapping[Tuple[str, str], dict],
    models: Sequence[str],
    metric: str,
    viz_dir: Path,
) -> None:
    plt = require_matplotlib()

    fig, axes = plt.subplots(
        1,
        len(CONTRASTS),
        figsize=(5.8 * len(CONTRASTS), 4.6),
        sharey=True,
        constrained_layout=True,
    )
    if len(CONTRASTS) == 1:
        axes = [axes]

    colors = {"easy": "#4C78A8", "hard": "#F58518", "line": "#222222"}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.34
    panel_titles = {
        "short_to_long_drop": "Short vs Long",
        "low_to_high_repair_drop": "Low vs High Repair",
    }
    easy_labels = {
        "short_to_long_drop": "short",
        "low_to_high_repair_drop": "low repair",
    }
    hard_labels = {
        "short_to_long_drop": "long",
        "low_to_high_repair_drop": "high repair",
    }

    legend_handles = None
    for ax, (contrast, bucket_type, easy_bucket, hard_bucket) in zip(axes, CONTRASTS):
        easy_means = [
            get_stat(summary, model, bucket_type, easy_bucket, metric)[0]
            for model in models
        ]
        easy_stds = [
            get_stat(summary, model, bucket_type, easy_bucket, metric)[1]
            for model in models
        ]
        hard_means = [
            get_stat(summary, model, bucket_type, hard_bucket, metric)[0]
            for model in models
        ]
        hard_stds = [
            get_stat(summary, model, bucket_type, hard_bucket, metric)[1]
            for model in models
        ]
        pct_means = [
            drops.get((contrast, model), {}).get(f"{metric}_pct_mean", np.nan)
            for model in models
        ]
        pct_stds = [
            drops.get((contrast, model), {}).get(f"{metric}_pct_std", np.nan)
            for model in models
        ]

        b1 = ax.bar(
            x - width / 2,
            easy_means,
            width=width,
            yerr=[0.0 if np.isnan(v) else v for v in easy_stds],
            capsize=3,
            color=colors["easy"],
            edgecolor="#222222",
            linewidth=0.7,
            label=easy_labels[contrast],
            alpha=0.92,
        )
        b2 = ax.bar(
            x + width / 2,
            hard_means,
            width=width,
            yerr=[0.0 if np.isnan(v) else v for v in hard_stds],
            capsize=3,
            color=colors["hard"],
            edgecolor="#222222",
            linewidth=0.7,
            label=hard_labels[contrast],
            alpha=0.92,
        )

        ax2 = ax.twinx()
        line = ax2.errorbar(
            x,
            pct_means,
            yerr=[0.0 if np.isnan(v) else v for v in pct_stds],
            color=colors["line"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            label="drop %",
        )
        ax2.set_ylabel("drop (%)")
        ax2.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(panel_titles.get(contrast, contrast))
        ax.set_ylabel("mean performance")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        if legend_handles is None:
            legend_handles = ([b1, b2, line], [
                easy_labels[contrast],
                hard_labels[contrast],
                "drop %",
            ])

    if legend_handles is not None:
        fig.legend(
            legend_handles[0],
            legend_handles[1],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=3,
            frameon=False,
        )
    fig.suptitle(f"{metric} performance and percentage drop", fontsize=13, y=1.16)

    stem = safe_metric_name(metric)
    for ext in ("png", "pdf"):
        fig.savefig(viz_dir / f"condition_change_{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_length_snapshot(
    summary: Mapping[Tuple[str, str, str], dict],
    models: Sequence[str],
    metrics: Sequence[str],
) -> None:
    print("\n========== LENGTH BUCKET MEAN ± STD ==========")
    for model in models:
        print(f"[{model}]")
        for bucket in ("short", "long"):
            cells = []
            for metric in metrics:
                stats = summary.get((model, "length", bucket), {})
                cells.append(
                    f"{metric}={format_mean_std(stats.get(f'{metric}_mean', np.nan), stats.get(f'{metric}_std', np.nan), stats.get('n_seeds', 0), metric)}"
                )
            print(f"  {bucket:5s}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", default=list(DEFAULT_SEED_DIRS))
    ap.add_argument("--stratified_subdir", default=DEFAULT_STRATIFIED_SUBDIR)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--plot_metrics", nargs="+", default=list(TABLE_METRICS),
                    choices=TABLE_METRICS)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--viz_dir", default="outputs/viz")
    ap.add_argument("--condition_table", default="stratified_condition_table_meanstd.csv")
    ap.add_argument("--drop_table", default="stratified_condition_drop_meanstd.csv")
    ap.add_argument("--drop_pct_table", default="stratified_condition_drop_pct_meanstd.csv")
    ap.add_argument("--title", default="Stratified retrieval performance change")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    viz_dir = Path(args.viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(viz_dir / ".matplotlib_cache"))

    rows = read_seed_metrics(args.seed_dirs, args.stratified_subdir)
    if not rows:
        raise SystemExit("no stratified_metrics.csv files found")

    summary = summarize_rows(rows)
    drops = summarize_condition_drops(rows, args.models)

    write_condition_table(summary, args.models, out_dir / args.condition_table)
    write_condition_drop_table(drops, out_dir / args.drop_table)
    write_condition_drop_pct_table(drops, out_dir / args.drop_pct_table)
    write_numeric_csv(summary, out_dir / "stratified_condition_numeric_meanstd.csv")
    write_json(summary, out_dir / "stratified_condition_numeric_meanstd.json")

    if not args.no_plots:
        plot_condition_absolute_metrics(
            summary,
            args.models,
            args.plot_metrics,
            viz_dir,
            "Absolute performance by dialogue length",
        )
        plot_condition_performance_change(
            drops,
            args.models,
            args.plot_metrics,
            viz_dir,
            args.title,
        )
        for metric in args.plot_metrics:
            plot_metric_combo_change(summary, drops, args.models, metric, viz_dir)

    print(f"saved tables to: {out_dir}")
    if not args.no_plots:
        print(f"saved figures to: {viz_dir}")
    print_length_snapshot(summary, args.models, args.plot_metrics)


if __name__ == "__main__":
    main()
