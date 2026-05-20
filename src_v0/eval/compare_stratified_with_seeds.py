"""
Aggregate post-hoc stratified eval CSVs across training seeds.

Run after src.eval.eval_stratified has produced one stratified_v6 directory per
seed.

Usage:
  python -m src.eval.compare_stratified_with_seeds \
      --seed_dirs outputs/seed_0 outputs/seed_1 outputs/seed_2 \
      --out_csv outputs/stratified_table_meanstd.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("R@1", "R@5", "R@10", "MRR", "median_rank")
DEFAULT_STRATIFIED_SUBDIR = "stratified_v6"


def fmt_meanstd(vals, metric):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "-"
    fmt = "{:.1f}" if metric == "median_rank" else "{:.4f}"
    if len(arr) == 1:
        return fmt.format(float(arr.mean()))
    return f"{fmt.format(float(arr.mean()))} ± {fmt.format(float(arr.std(ddof=1)))}"


def collect(seed_dirs, stratified_subdir):
    values = defaultdict(lambda: defaultdict(list))
    n_values = defaultdict(list)
    for seed_dir in seed_dirs:
        csv_path = Path(seed_dir) / stratified_subdir / "stratified_metrics.csv"
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["run"], row["bucket_type"], row["bucket"])
                try:
                    n_values[key].append(float(row["N_queries"]))
                except (TypeError, ValueError):
                    pass
                for metric in METRICS:
                    try:
                        values[key][metric].append(float(row[metric]))
                    except (TypeError, ValueError):
                        pass
    return values, n_values


def write_table(values, n_values, out_csv):
    rows = [["run", "bucket_type", "bucket", "N_queries", *METRICS]]
    for key in sorted(values.keys()):
        run, bucket_type, bucket = key
        n_cell = fmt_meanstd(n_values[key], "median_rank")
        rows.append([
            run,
            bucket_type,
            bucket,
            n_cell,
            *[fmt_meanstd(values[key][metric], metric) for metric in METRICS],
        ])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--stratified_subdir", default=DEFAULT_STRATIFIED_SUBDIR)
    ap.add_argument("--out_csv", default="outputs/stratified_table_meanstd.csv")
    args = ap.parse_args()

    values, n_values = collect(args.seed_dirs, args.stratified_subdir)
    if not values:
        raise SystemExit("no stratified_metrics.csv files found")
    write_table(values, n_values, args.out_csv)


if __name__ == "__main__":
    main()
