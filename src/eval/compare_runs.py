"""
Compare multiple eval runs side-by-side.

Each --run is the path to an eval out_dir produced by eval_tom_model.py
(it must contain metrics.json). You can pass any number of runs.

Usage:
  python -m src.eval.compare_runs \
      --runs outputs/eval_fixed outputs/eval_dcp outputs/eval_supA \
      --names fixed dcp_B supervised_A \
      --out_csv outputs/compare.csv

If --names is omitted, the directory basename is used.
The pretty table is printed to stdout; the CSV is saved if --out_csv is given.
"""

import os
import json
import argparse
from pathlib import Path



COLS = [
    # general
    ("dialogues",   "n_dialogues",                                  "{:d}",   None),
    ("turns",       "n_turns_evaluated",                            "{:d}",   None),
    # retrieval (higher is better)
    ("R@1",         "retrieval.R@1",                                "{:.4f}", False),
    ("R@5",         "retrieval.R@5",                                "{:.4f}", False),
    ("R@10",        "retrieval.R@10",                               "{:.4f}", False),
    ("MRR",         "retrieval.MRR",                                "{:.4f}", False),
    ("med_rank",    "retrieval.median_rank",                        "{:.1f}", True),
    # stability / switching
    ("stab_KL",     "stability_switching.stability_idx",            "{:.5f}", True),
    ("switch_KL",   "stability_switching.switching_idx",            "{:.5f}", False),
    ("switch_ratio","stability_switching.switching_ratio",          "{:.3f}", False),
    ("rho_KL_corr", "stability_switching.rho_kl_pearson",           "{:.4f}", False),
    ("n_stable",    "stability_switching.n_stable_turns",           "{:d}",   None),
    ("n_repair",    "stability_switching.n_repair_turns",           "{:d}",   None),
    # repair sparsity
    ("mean_rho",    "repair_sparsity.mean_rho",                     "{:.4f}", None),
    ("P(rho>0.5)",  "repair_sparsity.p_rho_gt_05",                  "{:.4f}", None),
    ("P(rho>0.7)",  "repair_sparsity.p_rho_gt_07",                  "{:.4f}", None),
    ("rho_q50",     "repair_sparsity.rho_quantiles.q50",            "{:.3f}", None),
    # optional repair AUC if labels existed
    ("AUC",         "repair_label_auc",                             "{:.4f}", False),
]


def get_path(d: dict, dotted: str):
    cur = d
    for key in dotted.split("."):
        if cur is None:
            return None
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def fmt_val(v, fmt: str) -> str:
    if v is None:
        return "—"
    try:
        if fmt.endswith("d"):
            return fmt.format(int(v))
        return fmt.format(float(v))
    except (ValueError, TypeError):
        return str(v)


def best_idx(vals, lower_is_better):
    """Return set of indices that hold the best value, ignoring None."""
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return set()
    if lower_is_better:
        target = min(v for _, v in valid)
    else:
        target = max(v for _, v in valid)
    return {i for i, v in valid if v == target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="eval out_dirs (each must contain metrics.json)")
    ap.add_argument("--names", nargs="*", default=None,
                    help="display names; defaults to directory basenames")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--no_color", action="store_true")
    args = ap.parse_args()

    names = args.names if args.names else [Path(r).name for r in args.runs]
    if len(names) != len(args.runs):
        raise SystemExit("--names length must match --runs length")

    metrics_list = []
    for r in args.runs:
        path = Path(r) / "metrics.json"
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        with open(path) as f:
            metrics_list.append(json.load(f))


    variants = []
    for m in metrics_list:
        v = (
            m.get("gate_variant")                                       
            or get_path(m, "checkpoint_args.gate_variant")              
            or get_path(m, "args.gate_variant")                         
        )
        variants.append(v if v else "?")


    BOLD = "" if args.no_color else "\033[1m"
    RST = "" if args.no_color else "\033[0m"
    GREEN = "" if args.no_color else "\033[92m"

    name_col_width = max(len("metric"), max(len(c[0]) for c in COLS)) + 2
    val_col_widths = [max(len(n), 10) + 2 for n in names]


    print()
    print(BOLD + "metric".ljust(name_col_width) + RST, end="")
    for n, w in zip(names, val_col_widths):
        print(BOLD + n.rjust(w) + RST, end="")
    print()
    print("variant".ljust(name_col_width), end="")
    for v, w in zip(variants, val_col_widths):
        print(v.rjust(w), end="")
    print()
    print("-" * (name_col_width + sum(val_col_widths)))

    csv_rows = [["metric"] + names, ["variant"] + variants]

    for col_name, dotted, fmt, lower_is_better in COLS:
        raw_vals = [get_path(m, dotted) for m in metrics_list]

        try:
            num_vals = [float(v) if v is not None else None for v in raw_vals]
        except (ValueError, TypeError):
            num_vals = raw_vals
        best = best_idx(num_vals, lower_is_better) if lower_is_better is not None else set()

        print(col_name.ljust(name_col_width), end="")
        csv_row = [col_name]
        for i, (v, w) in enumerate(zip(raw_vals, val_col_widths)):
            cell = fmt_val(v, fmt)
            csv_row.append(cell)
            if i in best:
                cell = GREEN + cell + RST
            print(cell.rjust(w + (len(GREEN) + len(RST) if i in best else 0)), end="")
        print()
        csv_rows.append(csv_row)
    print()

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        import csv as _csv
        with open(args.out_csv, "w", newline="") as f:
            w = _csv.writer(f)
            for row in csv_rows:
                w.writerow(row)
        print(f"saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
