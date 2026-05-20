"""
Aggregate per-seed eval results into mean ± std tables.

Each --seed_dirs entry is an OUT_BASE produced by run_v6.sh, e.g.
outputs/seed_0/, outputs/seed_1/, outputs/seed_2/.  Inside each one
there should be the standard subdirectories:

  baseline_random_v6/        baseline_clip_v6/
  eval_run_fixed_v6/         eval_run_dcp_v6/
  eval_run_multi_C_v6/       eval_run_multi_C_floor_v6/
  eval_run_multi_C_floor_drop_v6/   eval_run_pure_corr_v6/
  ablate_v6/{full,img_zero,img_random,img_mean,txt_zero,txt_random,txt_mean}

Outputs:
  --out_csv        the main table (8 columns, mean ± std cells)
  --ablation_csv   the ablation table (7 columns, mean ± std cells)

Usage:
  python -m src.eval.compare_runs_with_seeds \
      --seed_dirs outputs/seed_0 outputs/seed_1 outputs/seed_2 \
      --out_csv outputs/main_table_meanstd.csv \
      --ablation_csv outputs/ablation_table_meanstd.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


# (display_name, subpath relative to seed_dir)
MAIN_VARIANTS = [
    ("random",             "baseline_random_v6"),
    ("clip_only",          "baseline_clip_v6"),
    ("fixed",              "eval_run_fixed_v6"),
    ("dcp_B",              "eval_run_dcp_v6"),
    ("multi_C",            "eval_run_multi_C_v6"),
    ("multi_C+floor",      "eval_run_multi_C_floor_v6"),
    ("multi_C+floor+drop", "eval_run_multi_C_floor_drop_v6"),
    ("pure_corr",          "eval_run_pure_corr_v6"),
]

ABLATION_VARIANTS = [
    ("full",       "ablate_v6/full"),
    ("img_zero",   "ablate_v6/img_zero"),
    ("img_random", "ablate_v6/img_random"),
    ("img_mean",   "ablate_v6/img_mean"),
    ("txt_zero",   "ablate_v6/txt_zero"),
    ("txt_random", "ablate_v6/txt_random"),
    ("txt_mean",   "ablate_v6/txt_mean"),
]

# (display_name, dotted_path, fmt, lower_is_better_or_None)
METRICS = [
    ("R@1",         "retrieval.R@1",                                "{:.4f}", False),
    ("R@5",         "retrieval.R@5",                                "{:.4f}", False),
    ("R@10",        "retrieval.R@10",                               "{:.4f}", False),
    ("MRR",         "retrieval.MRR",                                "{:.4f}", False),
    ("med_rank",    "retrieval.median_rank",                        "{:.1f}", True),
    ("stab_KL",     "stability_switching.stability_idx",            "{:.5f}", True),
    ("switch_KL",   "stability_switching.switching_idx",            "{:.5f}", False),
    ("rho_KL_corr", "stability_switching.rho_kl_pearson",           "{:.4f}", False),
    ("mean_rho",    "repair_sparsity.mean_rho",                     "{:.3f}", None),
]


def get_path(d, dotted):
    cur = d
    for k in dotted.split("."):
        if cur is None or not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def collect(seed_dirs, variants):
    """Returns dict[variant_name][metric_name] -> list of seed values."""
    out = {}
    for vname, vsub in variants:
        out[vname] = {m[0]: [] for m in METRICS}
        for sd in seed_dirs:
            mj = Path(sd) / vsub / "metrics.json"
            if not mj.exists():
                continue
            with open(mj) as f:
                d = json.load(f)
            for mname, dotted, _, _ in METRICS:
                v = get_path(d, dotted)
                if v is None:
                    continue
                try:
                    out[vname][mname].append(float(v))
                except (ValueError, TypeError):
                    pass
    return out


def fmt_meanstd(vals, fmt):
    """Returns 'mean ± std' as one string. If only 1 seed, std hidden."""
    if not vals:
        return "—"
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "—"
    mean = float(arr.mean())
    if len(arr) >= 2:
        std = float(arr.std(ddof=1))
        return f"{fmt.format(mean)} ± {fmt.format(std)}"
    return f"{fmt.format(mean)}"


def best_indices(cells_means, lower_is_better):
    """For colored-cell highlighting, returns set of variant indices with best mean."""
    valid = [(i, m) for i, m in enumerate(cells_means) if m is not None]
    if not valid:
        return set()
    if lower_is_better:
        target = min(m for _, m in valid)
    else:
        target = max(m for _, m in valid)
    return {i for i, m in valid if m == target}


def render(agg, variants, out_csv, n_seeds, no_color=False):
    headers = ["metric"] + [v[0] for v in variants]
    csv_rows = [headers]

    BOLD = "" if no_color else "\033[1m"
    RST = "" if no_color else "\033[0m"
    GREEN = "" if no_color else "\033[92m"

    name_w = max(len("metric"), max(len(m[0]) for m in METRICS)) + 2
    val_w = max(22, max(len(v[0]) for v in variants) + 2)

    print()
    print(BOLD + "metric".ljust(name_w) + RST, end="")
    for v in variants:
        print(BOLD + v[0].rjust(val_w) + RST, end="")
    print()
    print("-" * (name_w + val_w * len(variants)))

    for mname, dotted, fmt, lib in METRICS:
        cell_strs = []
        cell_means = []
        for vname, _ in variants:
            vals = agg[vname][mname]
            cell_strs.append(fmt_meanstd(vals, fmt))
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[~np.isnan(arr)] if len(arr) else arr
            cell_means.append(float(arr.mean()) if len(arr) else None)

        best = best_indices(cell_means, lib) if lib is not None else set()

        print(mname.ljust(name_w), end="")
        csv_row = [mname]
        for i, c in enumerate(cell_strs):
            csv_row.append(c)
            disp = (GREEN + c + RST) if i in best else c
            extra = (len(GREEN) + len(RST)) if i in best else 0
            print(disp.rjust(val_w + extra), end="")
        print()
        csv_rows.append(csv_row)
    print()

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            for r in csv_rows:
                w.writerow(r)
        print(f"saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True,
                    help="per-seed OUT_BASE dirs, e.g. outputs/seed_0 outputs/seed_1 ...")
    ap.add_argument("--out_csv", default=None,
                    help="CSV path for the main 8-column table")
    ap.add_argument("--ablation_csv", default=None,
                    help="CSV path for the 7-column ablation table")
    ap.add_argument("--no_color", action="store_true")
    args = ap.parse_args()

    n_seeds = len(args.seed_dirs)
    print(f"\n==== MAIN TABLE  (mean ± std over {n_seeds} seeds) ====")
    agg_main = collect(args.seed_dirs, MAIN_VARIANTS)
    render(agg_main, MAIN_VARIANTS, args.out_csv, n_seeds, no_color=args.no_color)

    print(f"\n==== ABLATION TABLE  (mean ± std over {n_seeds} seeds) ====")
    agg_ab = collect(args.seed_dirs, ABLATION_VARIANTS)
    render(agg_ab, ABLATION_VARIANTS, args.ablation_csv, n_seeds,
           no_color=args.no_color)


if __name__ == "__main__":
    main()
