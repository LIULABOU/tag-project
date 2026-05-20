"""
Visualizations for the ToM evidence dynamics paper.

Three subcommands:
  hist     : overlay rho histograms across runs
              -> shows DCP bimodal vs multi_C continuous
  traj     : per-dialogue temporal trajectories of rho_t / a_t / d_t
              -> shows DCP spike-resets vs multi_C smooth tracking
  examples : auto-rank dialogues by interestingness criteria
              -> referential shift, grounding recovery, ambiguity

All plots use only matplotlib (no extra deps), saved at paper-quality DPI.
Reads per_turn.csv files produced by src.eval.eval_tom_model.

Usage:
  # 1. histogram comparison
  python -m src.eval.viz_dynamics hist \
      --runs outputs/eval_run_pure_corr_v4 outputs/eval_run_dcp_v4 outputs/eval_run_multi_C_v4 \
      --names pure_corr dcp_B multi_C \
      --out outputs/viz/rho_hist.png

  # 2. temporal trajectories (auto-pick first 6 dialogues)
  python -m src.eval.viz_dynamics traj \
      --runs outputs/eval_run_dcp_v4 outputs/eval_run_multi_C_v4 \
      --names dcp_B multi_C \
      --num_examples 6 \
      --out_dir outputs/viz/traj

  # 3. find interesting dialogues to feature
  python -m src.eval.viz_dynamics examples \
      --run outputs/eval_run_multi_C_v4 \
      --top_k 10
  # then feed the printed photo_ids back to traj via --photo_ids
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# Consistent style across plots. Lookup is case- and separator-insensitive
# (DCP-B / dcp_B / DCP_B all match the same entry).
STYLE = {
    "fixed":        {"color": "#777777", "marker": "s", "linestyle": "-"},
    "pure_corr":    {"color": "#9467bd", "marker": "D", "linestyle": "-"},
    "clip_only":    {"color": "#aaaaaa", "marker": "x", "linestyle": ":"},
    "dcp_b":        {"color": "#d62728", "marker": "^", "linestyle": "--"},
    "multi_c":      {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "multi_c+floor":{"color": "#2ca02c", "marker": "v", "linestyle": "-"},
    "supervised_a": {"color": "#ff7f0e", "marker": "P", "linestyle": "-."},
}


def _norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def style_for(name: str):
    """Return (color, marker, linestyle) for a method name. Case/-/_ insensitive."""
    s = STYLE.get(_norm(name))
    if s is None:
        return {"color": None, "marker": "o", "linestyle": "-"}
    return s


def color_for(name: str):
    return style_for(name)["color"]


def load_per_turn(csv_path: str):
    """photo_id -> list[dict] sorted by t."""
    by_dlg = defaultdict(list)
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            by_dlg[row["photo_id"]].append(row)
    for pid in by_dlg:
        by_dlg[pid].sort(key=lambda x: int(x["t"]))
    return by_dlg


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return np.array(out)


# ----------------------------------------------------------------------
# (1) histogram
# ----------------------------------------------------------------------
def cmd_hist(args):
    fig, ax = plt.subplots(figsize=(9, 5))

    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        by_dlg = load_per_turn(csv_path)
        rhos = np.array([float(r["rho"]) for dlg in by_dlg.values() for r in dlg])
        st = style_for(name)
        ax.hist(
            rhos, bins=40, alpha=0.55, density=False,
            label=f"{name}  (mean={rhos.mean():.3f}, std={rhos.std():.3f})",
            color=st["color"],
        )

    ax.set_xlabel(r"$\rho_t$  (repair gate)", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.set_title("Repair-gate distribution", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=11, loc="best", framealpha=0.95)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}")


# ----------------------------------------------------------------------
# (2) temporal trajectories
# ----------------------------------------------------------------------
def cmd_traj(args):
    runs_data = {}
    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        runs_data[name] = load_per_turn(csv_path)

    if not runs_data:
        print("no data loaded")
        return

    photo_ids = args.photo_ids
    if not photo_ids:
        first = next(iter(runs_data.values()))
        photo_ids = list(first.keys())[: args.num_examples]
    else:
        # Validate that requested photo_ids actually exist in at least one run
        all_pids = set()
        for by_dlg in runs_data.values():
            all_pids.update(by_dlg.keys())
        missing = [pid for pid in photo_ids if pid not in all_pids]
        if missing:
            print(f"[ERROR] {len(missing)} photo_id(s) not found in any run:")
            for m in missing[:5]:
                print(f"        {m!r}")
            print(f"[hint] run 'viz_dynamics examples --run <eval_dir>' to see valid IDs.")
            print(f"[hint] valid IDs look like 'train/<hash>_<num>', not '<PID1>'.")
            raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    signals = [
        ("rho", r"$\rho_t$  (repair gate)", (-0.05, 1.05)),
        ("a_t", r"$a_t$  (coherence)",      (-0.05, 1.05)),
        ("d_t", r"$d_t = \mathrm{KL}(e_{t-1}\,\|\,\hat e_t)$", None),
    ]

    for pid in photo_ids:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)

        for ax, (sig, ylab, ylim) in zip(axes, signals):
            for name, by_dlg in runs_data.items():
                if pid not in by_dlg:
                    continue
                dlg = by_dlg[pid]
                ts = [int(r["t"]) for r in dlg]
                vals = _floats(dlg, sig)
                st = style_for(name)
                ax.plot(
                    ts, vals, label=name,
                    color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                    linewidth=2.0, markersize=6, alpha=0.9,
                )
            ax.set_ylabel(ylab, fontsize=12)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("turn  t", fontsize=12)
        axes[0].set_title(f"Trajectory:  {pid}", fontsize=13)
        axes[0].legend(fontsize=10, loc="best", framealpha=0.95)
        plt.tight_layout()

        safe_pid = pid.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(args.out_dir, f"traj_{safe_pid}.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()

    print(f"saved {len(photo_ids)} trajectory plots to {args.out_dir}/")


# ----------------------------------------------------------------------
# (3) qualitative example finder
# ----------------------------------------------------------------------
def cmd_examples(args):
    csv_path = os.path.join(args.run, "per_turn.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing: {csv_path}")
    by_dlg = load_per_turn(csv_path)

    rankings = {
        "max_d_t (referential shift)":             [],
        "a_t drop+recovery (grounding recovery)":  [],
        "rho variance (modulation/ambiguity)":     [],
        "max abs(delta a_t) (alignment instability)": [],
    }

    for pid, dlg in by_dlg.items():
        if len(dlg) < 3:
            continue
        d_t = _floats(dlg, "d_t")
        a_t = _floats(dlg, "a_t")
        rho = _floats(dlg, "rho")

        if not np.isnan(d_t).all():
            rankings["max_d_t (referential shift)"].append((pid, float(np.nanmax(d_t))))

        if not np.isnan(a_t).all():
            a_clean = a_t[~np.isnan(a_t)]
            if len(a_clean) >= 3:
                peak_idx = int(a_clean.argmax())
                if peak_idx < len(a_clean) - 1:
                    after = a_clean[peak_idx:]
                    valley = float(after.min())
                    valley_idx = int(after.argmin())
                    after_valley = after[valley_idx:]
                    if len(after_valley) >= 1:
                        drop = float(a_clean[peak_idx]) - valley
                        recovery = float(after_valley.max()) - valley
                        if drop > 0.05 and recovery > 0.02:
                            rankings["a_t drop+recovery (grounding recovery)"].append(
                                (pid, drop + recovery)
                            )
            if len(a_clean) >= 2:
                deltas = np.abs(np.diff(a_clean))
                rankings["max abs(delta a_t) (alignment instability)"].append(
                    (pid, float(deltas.max()))
                )

        if not np.isnan(rho).all():
            rankings["rho variance (modulation/ambiguity)"].append(
                (pid, float(np.nanstd(rho)))
            )

    print(f"\nRanking dialogues from: {args.run}\n")
    for crit, lst in rankings.items():
        if not lst:
            continue
        lst.sort(key=lambda x: -x[1])
        print(f"=== {crit} ===")
        for pid, val in lst[: args.top_k]:
            print(f"  {val:7.3f}   {pid}")
        print()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hist", help="overlay rho histograms across runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out", default="outputs/viz/rho_hist.png")
    p.set_defaults(func=cmd_hist)

    p = sub.add_parser("traj", help="rho/a/d trajectories per dialogue")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--photo_ids", nargs="*", default=None,
                   help="dialogues to plot (default: auto first N)")
    p.add_argument("--num_examples", type=int, default=6)
    p.add_argument("--out_dir", default="outputs/viz/traj")
    p.set_defaults(func=cmd_traj)

    p = sub.add_parser("examples", help="auto-rank interesting dialogues")
    p.add_argument("--run", required=True, help="single eval out_dir")
    p.add_argument("--top_k", type=int, default=10)
    p.set_defaults(func=cmd_examples)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
