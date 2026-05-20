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
      --runs outputs/eval_run_pure_corr_v4 outputs/eval_run_dcp_v4 outputs/eval_run_multi_C_v6 \
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
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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
    fig, ax = plt.subplots(figsize=(6, 6))

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
            label=name,
            color=st["color"],
        )

    ax.set_xlabel(r"$\rho_t$  (repair gate)", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.set_xlim(0.0, 1.0)
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
# (4) find best exemplar for the regime panel
# ----------------------------------------------------------------------
def _score_exemplar(dcp_rows, mc_rows):
    """Score a dialogue for the regime panel.
    Rewards: DCP-B spike + a_t collapse, MULTI-C final coherence wins."""
    if len(dcp_rows) < 4 or len(mc_rows) < 4:
        return -1.0, {}
    dcp_rho = _floats(dcp_rows, "rho")
    dcp_at  = _floats(dcp_rows, "a_t")
    mc_at   = _floats(mc_rows,  "a_t")
    if np.isnan(dcp_rho).all() or np.isnan(dcp_at).all() or np.isnan(mc_at).all():
        return -1.0, {}

    mid_candidates = [(i, dcp_rho[i]) for i in range(1, len(dcp_rho)) if dcp_rho[i] > 0.5]
    peak_i = max(mid_candidates, key=lambda x: x[1])[0] if mid_candidates else int(np.nanargmax(dcp_rho))
    mid_bonus = 1.5 if peak_i >= 2 else 1.0

    dcp_max_rho    = float(dcp_rho[peak_i])
    at_before      = float(np.nanmean(dcp_at[:peak_i])) if peak_i > 0 else float(dcp_at[0])
    at_after       = float(np.nanmean(dcp_at[peak_i + 1:])) if peak_i < len(dcp_at) - 1 else float(dcp_at[-1])
    dcp_collapse   = max(0.0, at_before - at_after)
    mc_final       = float(np.nanmean(mc_at[-3:]))
    dcp_final      = float(np.nanmean(dcp_at[-3:]))
    mc_wins        = max(0.0, mc_final - dcp_final)

    score = dcp_max_rho * dcp_collapse * mc_wins * mid_bonus * mc_final
    stats = dict(
        dcp_max_rho=dcp_max_rho, dcp_collapse=dcp_collapse,
        mc_final=mc_final, dcp_final=dcp_final, mc_wins=mc_wins,
        spike_t=peak_i + 1, n_turns=len(dcp_rows),
    )
    return score, stats


def cmd_find_exemplar(args):
    runs_data = {}
    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        runs_data[_norm(name)] = load_per_turn(csv_path)

    if "dcp_b" not in runs_data or "multi_c" not in runs_data:
        print(f"[ERROR] Need dcp_b and multi_c in --names. Got: {list(runs_data.keys())}")
        return

    common = set.intersection(*[set(d.keys()) for d in runs_data.values()])
    print(f"Dialogues present in all {len(runs_data)} runs: {len(common)}")

    scored = []
    for pid in common:
        sc, stats = _score_exemplar(runs_data["dcp_b"][pid], runs_data["multi_c"][pid])
        if sc > 0:
            scored.append((sc, pid, stats))

    scored.sort(reverse=True)
    print(f"\nTop {args.top_k} candidates  (score = dcp_spike × collapse × multi_c_wins × final_at):\n")
    hdr = f"{'score':>8}  {'spike_t':>7}  {'dcp_rho':>7}  {'collapse':>8}  {'mc_final':>8}  {'mc_wins':>7}  photo_id"
    print(hdr)
    print("-" * len(hdr))
    for sc, pid, s in scored[: args.top_k]:
        print(
            f"{sc:8.5f}  {s['spike_t']:>7}  {s['dcp_max_rho']:7.3f}  "
            f"{s['dcp_collapse']:8.3f}  {s['mc_final']:8.3f}  {s['mc_wins']:7.3f}  {pid}"
        )
    if scored:
        print(f"\nRecommended photo_id: {scored[0][1]}")


# ----------------------------------------------------------------------
# (5) regime panel: 4 columns × (image + d_t + rho + a_t)
# ----------------------------------------------------------------------
def _load_image_map(jsonl_path):
    m = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
                m[e["photo_id"]] = e["image_path"]
            except Exception:
                pass
    return m


def _center_crop(img, target_ratio=0.75):
    """Crop image to target_ratio (w/h). Default 0.75 gives a tall-ish portrait."""
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img[:, x0: x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return img[y0: y0 + new_h, :]


def _load_image_array(pid, image_map_path, image_dir):
    """Return an HxWx3 uint8 array or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not image_map_path or not os.path.exists(image_map_path):
        return None
    m = _load_image_map(image_map_path)
    if pid not in m:
        return None
    fname = os.path.basename(m[pid])
    candidates = [
        os.path.join(image_dir, fname) if image_dir else None,
        os.path.join(os.path.dirname(image_map_path), "..", "images", fname),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return np.array(Image.open(p).convert("RGB"))
            except Exception:
                pass
    return None


def cmd_regime(args):
    runs_data = {}
    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        runs_data[name] = load_per_turn(csv_path)

    pid  = args.photo_id
    names = [n for n in args.names if n in runs_data]
    n_cols = len(names)
    if n_cols == 0:
        print("no data loaded"); return

    img = _load_image_array(pid, args.image_map, args.image_dir)
    has_img = img is not None


    fig = plt.figure(figsize=(4.8 * n_cols, 14 if has_img else 10))
    if has_img:
        gs = GridSpec(
            4, n_cols, figure=fig,
            height_ratios=[2.8, 1.1, 1.5, 1.5],
            hspace=0.50, wspace=0.28,
            left=0.07, right=0.97, top=0.96, bottom=0.09,
        )
        
        if n_cols >= 4:
            ax_img = fig.add_subplot(gs[0, 1:3])
            for c in [0, n_cols - 1]:
                fig.add_subplot(gs[0, c]).axis("off")
        elif n_cols >= 2:
            ax_img = fig.add_subplot(gs[0, :])
        else:
            ax_img = fig.add_subplot(gs[0, 0])

        ax_img.imshow(img, aspect="auto")
        ax_img.axis("off")
        ax_img.set_title(f"Shared photo  ·  dialogue: {pid}", fontsize=11, pad=5)
        metric_row_offset = 1
    else:
        gs = GridSpec(
            3, n_cols, figure=fig,
            hspace=0.48, wspace=0.28,
            left=0.08, right=0.97, top=0.93, bottom=0.09,
        )
        metric_row_offset = 0

    signals = [
        ("d_t",  r"$d_t$  (divergence signal)",        None),
        ("rho",  r"$\rho_t$  (repair gate)",            (-0.05, 1.05)),
        ("a_t",  r"$a_t$  (coherence with referent)",  (-0.05, 1.05)),
    ]

    # ---- plot metric rows ----
    axes = {}   
    for col, name in enumerate(names):
        if pid not in runs_data.get(name, {}):
            print(f"[warn] photo_id {pid!r} not found in run {name!r}")
            continue
        dlg = runs_data[name][pid]
        ts  = [int(r["t"]) for r in dlg]
        st  = style_for(name)

        for row, (sig, ylab, ylim) in enumerate(signals):
            ax = fig.add_subplot(gs[metric_row_offset + row, col])
            axes[(row, col)] = ax
            vals = _floats(dlg, sig)

            ax.plot(ts, vals,
                    color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"],
                    linewidth=2.2, markersize=5, alpha=0.9)
            if row == 0:
                label = name.upper().replace("_", "-")
                ax.set_title(label, fontsize=12, fontweight="bold",
                             color=st["color"] or "black", pad=4)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if row == len(signals) - 1:
                ax.set_xlabel("turn  t", fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)

    # ---- DCP-B annotations ----
    dcpb_col = next((i for i, n in enumerate(names) if _norm(n) == "dcp_b"), None)
    if dcpb_col is not None:
        dname = names[dcpb_col]
        if pid in runs_data.get(dname, {}):
            dlg      = runs_data[dname][pid]
            ts       = [int(r["t"]) for r in dlg]
            rho_vals = _floats(dlg, "rho")
            at_vals  = _floats(dlg, "a_t")

            mid_cands = [(i, rho_vals[i]) for i in range(1, len(rho_vals)) if rho_vals[i] > 0.5]
            peak_i = max(mid_cands, key=lambda x: x[1])[0] if mid_cands else int(np.nanargmax(rho_vals))

            # ρ_t: mark spike
            ax_rho = axes.get((1, dcpb_col))
            if ax_rho:
                ax_rho.axvspan(ts[peak_i] - 0.45, ts[peak_i] + 0.45, alpha=0.15, color="#d62728")
                yt = float(rho_vals[peak_i])
                ax_rho.annotate(
                    "overreact",
                    xy=(ts[peak_i], yt),
                    xytext=(ts[peak_i] + max(1, len(ts) * 0.1), max(0.05, yt - 0.22)),
                    fontsize=8.5, color="#d62728", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
                )

            # a_t: mark collapse after spike
            ax_at = axes.get((2, dcpb_col))
            if ax_at and peak_i < len(at_vals) - 1:
                post = at_vals[peak_i + 1:]
                collapse_i = peak_i + 1 + int(np.nanargmin(post))
                ax_at.axvspan(ts[peak_i] - 0.45, ts[collapse_i] + 0.45, alpha=0.10, color="#d62728")
                yc = float(at_vals[collapse_i])
                offset_y = min(yc + 0.18, 0.95)
                ax_at.annotate(
                    "collapses",
                    xy=(ts[collapse_i], yc),
                    xytext=(ts[collapse_i] + max(1, len(ts) * 0.1), offset_y),
                    fontsize=8.5, color="#d62728", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
                )

    # ---- MULTI-C annotations ----
    mc_col = next((i for i, n in enumerate(names) if _norm(n) == "multi_c"), None)
    if mc_col is not None:
        mname = names[mc_col]
        if pid in runs_data.get(mname, {}):
            dlg      = runs_data[mname][pid]
            ts       = [int(r["t"]) for r in dlg]
            rho_vals = _floats(dlg, "rho")
            at_vals  = _floats(dlg, "a_t")

            # ρ_t: "persists" bracket over the middle stretch
            ax_rho = axes.get((1, mc_col))
            if ax_rho and len(ts) >= 4:
                m0, m1 = ts[1], ts[-2]
                y_mid  = float(np.nanmean(rho_vals[1:-1]))
                ax_rho.annotate(
                    "", xy=(m1, y_mid), xytext=(m0, y_mid),
                    arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.6),
                )
                ax_rho.text(
                    (m0 + m1) / 2, y_mid + 0.07, "persists",
                    ha="center", va="bottom", fontsize=8.5,
                    color="#1f77b4", fontweight="bold",
                )

            # a_t: "converges" at final high value
            ax_at = axes.get((2, mc_col))
            if ax_at:
                fi = len(ts) - 1
                yf = float(at_vals[fi])
                ax_at.annotate(
                    "converges",
                    xy=(ts[fi], yf),
                    xytext=(ts[fi] - max(2, len(ts) // 3), max(0.05, yf - 0.2)),
                    fontsize=8.5, color="#1f77b4", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4),
                )

    # ---- narrative caption ----
    caption = (
        r"$\bf{DCP\!-\!B}$: treats local surprise as an immediate revision signal — "
        "abandons prior grounding.   "
        r"$\bf{MULTI\!-\!C}$: preserves accumulated belief under transient mismatch — "
        "revises selectively when ambiguity persists."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, style="italic",
             bbox=dict(facecolor="#f8f8f8", edgecolor="#cccccc", boxstyle="round,pad=0.5", alpha=0.9))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}")


# ----------------------------------------------------------------------
# (6) compact two-example paper figure  (figure*, 7" × ~2.6", no annotations)
# ----------------------------------------------------------------------
def _find_dialogue(photo_id, train_json_dir):
    """Return PhotoChat entry dict for photo_id, or None."""
    for path in sorted(glob.glob(os.path.join(train_json_dir, "*.json"))):
        try:
            for entry in json.load(open(path)):
                if entry.get("photo_id") == photo_id:
                    return entry
        except Exception:
            pass
    return None


def _select_bubble_turns(turns, max_t, max_show=9):
    """Select up to max_show turns to display, always including the photo turn.

    Strategy: keep first 2, last 1, photo turn ± 1, fill remaining from middle.
    Returns list of (original_index, turn_dict), with None entries for ellipsis gaps.
    """
    n = min(len(turns), max_t)
    if n <= max_show:
        return [(i, turns[i]) for i in range(n)]

    # Find photo turn index
    photo_idx = next((i for i in range(n) if turns[i]["share_photo"]), n - 1)

    # Mandatory indices
    must = {0, 1, max(0, photo_idx - 1), photo_idx,
            min(n - 1, photo_idx + 1), n - 1}
    must = sorted(i for i in must if i < n)

    # Fill remaining slots from evenly spaced middle turns
    remaining = max_show - len(must)
    middle = [i for i in range(2, n - 1) if i not in must]
    step = max(1, len(middle) // (remaining + 1))
    extra = [middle[j] for j in range(step, len(middle), step)][:remaining]
    selected = sorted(set(must) | set(extra))

    # Build result with ellipsis markers (None) for gaps
    result = []
    for k, idx in enumerate(selected):
        if k > 0 and idx > selected[k - 1] + 1:
            result.append(None)           # ellipsis
        result.append((idx, turns[idx]))
    return result


def _draw_bubbles(ax, turns, max_t, fontsize=5.8):
    """Draw compact speech bubbles top-to-bottom, intelligently selecting turns."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    items = _select_bubble_turns(turns, max_t, max_show=9)
    n = len(items)
    row_h = 1.0 / n

    for row_i, item in enumerate(items):
        y = 1.0 - (row_i + 0.5) * row_h

        if item is None:
            ax.text(0.20, y, "⋮", ha="left", va="center",
                    fontsize=fontsize + 1, color="#999999",
                    transform=ax.transAxes)
            continue

        orig_i, turn = item
        msg = turn["message"]
        is_photo = turn["share_photo"]

        if is_photo:
            msg = "▶ [photo]"
            fc, ec, lw = "#FFF3CD", "#E8A000", 1.0
        else:
            if len(msg) > 22:
                cut = msg[:23].rfind(" ")
                cut = cut if cut > 8 else 22
                msg = msg[:cut] + "…"
            fc  = "#E8F4FD" if turn["user_id"] == 0 else "#F5F5F5"
            ec, lw = "#BBBBBB", 0.4

        ax.text(
            0.04, y, f"{orig_i + 1}  {msg}",
            ha="left", va="center", fontsize=fontsize,
            transform=ax.transAxes, clip_on=False,
            bbox=dict(boxstyle="square,pad=0.10", facecolor=fc,
                      edgecolor=ec, linewidth=lw, alpha=0.95),
        )


def _build_example(pid, runs_data, names, train_json_dir, image_map, image_dir):
    """Load all data for one example into a dict."""
    dlg_entry = _find_dialogue(pid, train_json_dir) if train_json_dir else None
    img       = _load_image_array(pid, image_map, image_dir)
    max_t     = max(
        (max(int(r["t"]) for r in runs_data[nm][pid])
         for nm in names if nm in runs_data and pid in runs_data[nm]),
        default=10,
    )
    photo_t = None
    if dlg_entry:
        for i, t in enumerate(dlg_entry["dialogue"]):
            if t["share_photo"]:
                photo_t = i + 1
                break
    return dict(pid=pid, dlg=dlg_entry, img=img, max_t=max_t, photo_t=photo_t)


def _draw_example(fig, subplot_spec, ex, runs_data, names, signals,
                  left_w, metric_w, fontsize, lw, ms):
    """Draw one example (photo + context + metrics) into subplot_spec."""
    n_methods = len(names)
    gs = GridSpecFromSubplotSpec(
        2, 1 + n_methods,
        subplot_spec=subplot_spec,
        width_ratios=[left_w] + [metric_w] * n_methods,
        height_ratios=[1.15, 1.0],
        wspace=0.22, hspace=0.28,
    )

    pid     = ex["pid"]
    img     = ex["img"]
    dlg     = ex["dlg"]
    max_t   = ex["max_t"]
    photo_t = ex["photo_t"]

    # photo — top-left, center-cropped
    ax_photo = fig.add_subplot(gs[0, 0])
    if img is not None:
        ax_photo.imshow(_center_crop(img, target_ratio=0.85), aspect="auto")
    ax_photo.axis("off")

    # context — bottom-left
    ax_bub = fig.add_subplot(gs[1, 0])
    if dlg:
        _draw_bubbles(ax_bub, dlg["dialogue"], max_t, fontsize=fontsize)
    else:
        ax_bub.axis("off")

    # metrics: row 0 = ρ_t, row 1 = a_t
    for m_idx, name in enumerate(names):
        for row, (sig, ylab, _ylim) in enumerate(signals):
            ax = fig.add_subplot(gs[row, 1 + m_idx])
            st = style_for(name)

            if name in runs_data and pid in runs_data[name]:
                rows = [r for r in runs_data[name][pid] if int(r["t"]) >= 1]
                ts   = [int(r["t"]) for r in rows]
                vals = _floats(rows, sig)
                ax.plot(ts, vals,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["linestyle"],
                        linewidth=lw, markersize=ms, alpha=0.9)

            if photo_t is not None:
                ax.axvline(photo_t, color="#E8A000", linestyle="--",
                           linewidth=0.8, alpha=0.75)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.18, linewidth=0.35)
            ax.tick_params(labelsize=fontsize - 0.5, length=1.5, pad=1)
            ax.set_xticks(range(1, max_t + 1, max(1, max_t // 5)))

            if row == 0:
                ax.set_title(name.upper().replace("_", "-"),
                             fontsize=fontsize, fontweight="bold",
                             color=st["color"] or "black", pad=2)
            if m_idx == 0:
                ax.set_ylabel(ylab, fontsize=fontsize, labelpad=2)
            else:
                ax.set_yticklabels([])
            if row == len(signals) - 1:
                ax.set_xlabel("t", fontsize=fontsize - 0.5, labelpad=1)


def cmd_paper(args):
    """Compact two-example figure for EMNLP paper (figure*, full page width)."""
    runs_data = {}
    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        runs_data[name] = load_per_turn(csv_path)

    signals = [
        ("rho", r"$\rho_t$", (-0.05, 1.05)),
        ("a_t", r"$a_t$",    (-0.05, 1.05)),
    ]
    examples = [
        _build_example(pid, runs_data, args.names,
                       args.train_json_dir, args.image_map, args.image_dir)
        for pid in args.photo_ids
    ]

    fig = plt.figure(figsize=(7.2, 2.85))
    gs_main = GridSpec(1, 2, figure=fig,
                       left=0.03, right=0.99, top=0.97, bottom=0.08,
                       wspace=0.06)

    for ex_idx, ex in enumerate(examples):
        _draw_example(fig, gs_main[ex_idx], ex, runs_data, args.names, signals,
                      left_w=3.2, metric_w=1.4, fontsize=5.5, lw=1.4, ms=2.0)
        if ex_idx == 0:
            fig.add_artist(plt.Line2D([0.502, 0.502], [0.0, 1.0],
                                      transform=fig.transFigure,
                                      color="#CCCCCC", linewidth=0.6))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}")


def cmd_single(args):
    """Save three files per example: metrics figure, photo, and context txt."""
    runs_data = {}
    for run_dir, name in zip(args.runs, args.names):
        csv_path = os.path.join(run_dir, "per_turn.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing: {csv_path}")
            continue
        runs_data[name] = load_per_turn(csv_path)

    ex = _build_example(args.photo_id, runs_data, args.names,
                        args.train_json_dir, args.image_map, args.image_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. metrics-only figure (ρ_t and a_t, 4 columns) ----
    signals   = [("rho", r"$\rho_t$"), ("a_t", r"$a_t$")]
    n_methods = len(args.names)
    pid, max_t, photo_t = ex["pid"], ex["max_t"], ex["photo_t"]

    fig, axes = plt.subplots(
        2, n_methods,
        figsize=(2.2 * n_methods, 2.8),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12,
                        wspace=0.12, hspace=0.25)

    # pre-reveal cutoff: only plot turns strictly before photo sharing
    pre_max_t = (photo_t - 1) if photo_t is not None else max_t

    for m_idx, name in enumerate(args.names):
        st = style_for(name)
        for row, (sig, ylab) in enumerate(signals):
            ax = axes[row, m_idx]
            if name in runs_data and pid in runs_data[name]:
                rows_data = [r for r in runs_data[name][pid]
                             if 1 <= int(r["t"]) <= pre_max_t]
                if rows_data:
                    ts   = [int(r["t"]) for r in rows_data]
                    vals = _floats(rows_data, sig)
                    ax.plot(ts, vals,
                            color=st["color"], marker=st["marker"],
                            linestyle=st["linestyle"],
                            linewidth=1.6, markersize=2.5, alpha=0.9)

            # orange boundary line at the reveal turn (right edge)
            if photo_t is not None:
                ax.axvline(photo_t - 0.5, color="#E8A000", linestyle="--",
                           linewidth=0.9, alpha=0.8)

            ax.set_xlim(0.5, pre_max_t + 0.6)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.20, linewidth=0.4)
            ax.tick_params(labelsize=6.5, length=2, pad=1)
            ax.set_xticks(range(1, pre_max_t + 1, max(1, pre_max_t // 5)))

            if row == 0:
                ax.set_title(name.upper().replace("_", "-"),
                             fontsize=7.5, fontweight="bold",
                             color=st["color"] or "black", pad=3)
                # mark the reveal boundary on the top row only
                if photo_t is not None and m_idx == 0:
                    ax.text(photo_t - 0.4, 1.02, "▶ photo",
                            fontsize=5.5, color="#E8A000", va="bottom",
                            transform=ax.get_xaxis_transform())
            if m_idx == 0:
                ax.set_ylabel(ylab, fontsize=7.5, labelpad=2)
            else:
                ax.set_yticklabels([])
            if row == len(signals) - 1:
                ax.set_xlabel("t  (pre-reveal)", fontsize=7.0, labelpad=1)

    out_fig = os.path.join(args.out_dir, "figure.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_fig}")

    # ---- 2. photo ----
    if ex["img"] is not None:
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(_center_crop(ex["img"], target_ratio=0.85)).save(
                os.path.join(args.out_dir, "photo.jpg")
            )
            print(f"saved: {os.path.join(args.out_dir, 'photo.jpg')}")
        except Exception as e:
            print(f"[warn] could not save photo: {e}")

    # ---- 3. context.txt ----
    if ex["dlg"]:
        lines = [f"photo_id: {pid}",
                 f"description: {ex['dlg'].get('photo_description', '')}",
                 ""]
        for i, turn in enumerate(ex["dlg"]["dialogue"]):
            speaker = "A" if turn["user_id"] == 0 else "B"
            mark    = "  ◀ PHOTO SHARED" if turn["share_photo"] else ""
            msg     = turn["message"] if turn["message"] else "[photo]"
            lines.append(f"t{i+1:02d} [{speaker}]: {msg}{mark}")
        out_txt = os.path.join(args.out_dir, "context.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved: {out_txt}")


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

    p = sub.add_parser("find_exemplar", help="rank photo_ids for the regime panel")
    p.add_argument("--runs",  nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.set_defaults(func=cmd_find_exemplar)

    p = sub.add_parser("single", help="standalone figure for one example (photo + context + metrics)")
    p.add_argument("--runs",          nargs="+", required=True)
    p.add_argument("--names",         nargs="+", required=True)
    p.add_argument("--photo_id",      required=True)
    p.add_argument("--image_map",     default="data/photochat/train_image_photo_desc.jsonl")
    p.add_argument("--image_dir",     default=None)
    p.add_argument("--train_json_dir",default="data/photochat/train_json")
    p.add_argument("--out_dir",       required=True)
    p.set_defaults(func=cmd_single)

    p = sub.add_parser("paper", help="compact two-example figure for EMNLP (figure*, full width)")
    p.add_argument("--runs",          nargs="+", required=True)
    p.add_argument("--names",         nargs="+", required=True)
    p.add_argument("--photo_ids",     nargs=2,   required=True)
    p.add_argument("--image_map",     default="data/photochat/train_image_photo_desc.jsonl")
    p.add_argument("--image_dir",     default=None)
    p.add_argument("--train_json_dir",default="data/photochat/train_json")
    p.add_argument("--out",           default="outputs/viz/paper_figure.png")
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("regime", help="4-column narrative panel (image + d_t + rho + a_t)")
    p.add_argument("--runs",      nargs="+", required=True)
    p.add_argument("--names",     nargs="+", required=True)
    p.add_argument("--photo_id",  required=True)
    p.add_argument("--image_map", default="data/photochat/train_image_photo_desc.jsonl",
                   help="JSONL mapping photo_id -> image path")
    p.add_argument("--image_dir", default=None,
                   help="directory containing image files (auto-detected if omitted)")
    p.add_argument("--out",       default="outputs/viz/regime_panel.png")
    p.set_defaults(func=cmd_regime)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
