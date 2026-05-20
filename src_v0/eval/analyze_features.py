"""
Feature contribution analysis for multi_C gates.

Loads a multi_C checkpoint, runs forward on the test split, captures the
five input signals  (d_t, a_{t-1}, Delta_a_hat_t, H(e_{t-1}), H(e_hat_t))
together with rho_t for every turn, and reports:

  1. Raw learned weights W_p[0..4] of the gate's Linear(5, 1)
  2. Standard deviation of each input signal on the test data
  3. Scaled importance: |W_p[i]| * std(signal_i)
  4. Pearson correlation: corr(signal_i, rho_t)

Outputs:
  <out_dir>/feature_contribution.csv
  <out_dir>/feature_contribution.png

Usage:
  python -m src.eval.analyze_features \
      --checkpoint outputs/seed_0/run_multi_C_v6/model.pt \
      --split_manifest outputs/split_v1.json \
      --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
      --out_dir outputs/feature_analysis_seed_0

For a multi-seed analysis, point --checkpoint at each seed's model.pt
in turn and the printed table can be averaged across seeds.
"""

import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from src.models.clip_model import CLIPBackbone
from src.models.alignment import AlignmentHead
from src.models.mvp_aligner import MSCPGate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8

SIGNAL_NAMES = ["d_t", "a_{t-1}", "Δâ_t", "H(e_{t-1})", "H(ê_t)"]


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def load_image_map(path: str):
    m = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            m[rec["photo_id"]] = rec["image_path"]
    return m


def load_test_dialogues(manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)
    out = []
    for s in manifest["test_shards"]:
        with open(os.path.join(manifest["shards_dir"], s)) as f:
            out.extend(json.load(f))
    return out


# ---------------------------------------------------------------------------
# core: run the multi_C path and capture signals
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_signals(
    clip: CLIPBackbone,
    align: AlignmentHead,
    gate_module: MSCPGate,
    image_map,
    dialogues,
    max_dialogues=None,
):
    """Returns ndarray of shape [N_turns, 6]: 5 signals + rho_t."""
    rows = []
    n_processed = 0

    for ex in dialogues:
        if max_dialogues is not None and n_processed >= max_dialogues:
            break
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue", [])
        if not photo_id or not dialogue or photo_id not in image_map:
            continue
        image_path = image_map[photo_id]
        if not os.path.exists(image_path):
            continue
        turns = [t.get("message", "") for t in dialogue]
        if not turns:
            continue
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        pixel_values = clip.processor.image_processor(
            images=[pil_img], return_tensors="pt"
        )["pixel_values"].to(DEVICE)
        text_inputs = clip.processor.tokenizer(
            turns, padding=True, truncation=True, return_tensors="pt", max_length=77
        ).to(DEVICE)

        v_cls_b, v_patches_b = clip.encode_vision_tokens(pixel_values)
        h_all = clip.encode_text_pooler(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        v_cls = v_cls_b[0]
        v_patches = v_patches_b[0]

        T = h_all.shape[0]
        N = v_patches.shape[0]
        q_seq = align.project_query(h_all)
        e_hat_seq = align.raw_evidence(q_seq, v_patches)

        e_prev = torch.full((N,), 1.0 / N, device=DEVICE)
        a_prev = torch.full((), 0.5, device=DEVICE)

        for t in range(T):
            e_hat_t = e_hat_seq[t]
            d_t = (e_prev * (torch.log(e_prev + EPS) - torch.log(e_hat_t + EPS))).sum()

            c_hat_t = e_hat_t @ v_patches
            a_hat_t = align.alignment_score(
                q_seq[t : t + 1], c_hat_t.unsqueeze(0), v_cls
            ).squeeze()
            delta_a_hat = a_hat_t - a_prev
            h_e_prev = -(e_prev * torch.log(e_prev + EPS)).sum()
            h_e_hat = -(e_hat_t * torch.log(e_hat_t + EPS)).sum()

            signals = torch.stack(
                [d_t, a_prev, delta_a_hat, h_e_prev, h_e_hat], dim=0
            )
            rho_t = gate_module(signals)

            rows.append([
                float(d_t.item()),
                float(a_prev.item()),
                float(delta_a_hat.item()),
                float(h_e_prev.item()),
                float(h_e_hat.item()),
                float(rho_t.item()),
            ])

            # advance the dynamics for the next turn
            e_t_unnorm = (1.0 - rho_t) * e_prev + rho_t * e_hat_t
            e_t = e_t_unnorm / (e_t_unnorm.sum() + 1e-12)
            c_t = e_t @ v_patches
            a_tilde_t = align.alignment_score(
                q_seq[t : t + 1], c_t.unsqueeze(0), v_cls
            ).squeeze()
            lam_t = 1.0 - rho_t
            a_t = (1.0 - lam_t) * a_prev + lam_t * a_tilde_t

            e_prev = e_t
            a_prev = a_t

        n_processed += 1

    return np.array(rows, dtype=np.float64), n_processed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split_manifest", type=str, required=True)
    ap.add_argument("--image_map_jsonl", type=str,
                    default="data/photochat/train_image_photo_desc.jsonl")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_dialogues", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load model ----
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    if ckpt.get("gate_variant") != "multi_C":
        raise SystemExit(
            f"checkpoint gate_variant={ckpt.get('gate_variant')!r}, expected 'multi_C'"
        )

    clip = CLIPBackbone(device=DEVICE)
    align = AlignmentHead(
        h_dim=clip.text_dim, v_dim=clip.vision_dim, q_dim=clip.vision_dim
    ).to(DEVICE)
    align.load_state_dict(ckpt["alignment_head"])
    align.eval()

    gate_module = MSCPGate().to(DEVICE)
    gate_module.load_state_dict(ckpt["gate_module"])
    gate_module.eval()

    # ---- load data ----
    dialogues = load_test_dialogues(args.split_manifest)
    image_map = load_image_map(args.image_map_jsonl)
    print(f"[analyze] {len(dialogues)} test dialogues "
          f"(max={args.max_dialogues})")

    # ---- collect signals ----
    rows, n_dlg = collect_signals(
        clip, align, gate_module, image_map, dialogues, args.max_dialogues
    )
    print(f"[analyze] collected {len(rows)} turns from {n_dlg} dialogues")

    if len(rows) == 0:
        raise SystemExit("[analyze] no turns collected")

    signals = rows[:, :5]   # [N, 5]
    rhos = rows[:, 5]       # [N]

    W = gate_module.head.weight.detach().cpu().numpy().squeeze()  # [5]
    b = float(gate_module.head.bias.detach().cpu().numpy().item())

    means = signals.mean(axis=0)
    stds = signals.std(axis=0)

    # Scaled importance: |W_i| * std(x_i)  -- accounts for different signal scales
    scaled_importance = np.abs(W) * stds

    # Pearson correlations between each signal and rho
    corrs = np.array([
        np.corrcoef(signals[:, i], rhos)[0, 1] if stds[i] > 1e-12 else 0.0
        for i in range(5)
    ])

    # ---- print table ----
    print()
    print(f"{'signal':<12} {'mean':>10} {'std':>10} {'W_p':>10} "
          f"{'|W|·std':>11} {'corr(·,ρ)':>11}")
    print("-" * 72)
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"{name:<12} {means[i]:>10.4f} {stds[i]:>10.4f} "
              f"{W[i]:>10.4f} {scaled_importance[i]:>11.4f} {corrs[i]:>11.4f}")
    print(f"{'(bias)':<12} {'':>10} {'':>10} {b:>10.4f}")
    print()
    print(f"final rho stats: mean={rhos.mean():.4f}, "
          f"std={rhos.std():.4f}, median={np.median(rhos):.4f}")
    print()

    # ---- save CSV ----
    csv_path = os.path.join(args.out_dir, "feature_contribution.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "mean", "std", "W_p", "|W|*std", "corr_with_rho"])
        for i, name in enumerate(SIGNAL_NAMES):
            w.writerow([
                name, f"{means[i]:.6f}", f"{stds[i]:.6f}",
                f"{W[i]:.6f}", f"{scaled_importance[i]:.6f}",
                f"{corrs[i]:.6f}",
            ])
        w.writerow(["(bias)", "", "", f"{b:.6f}", "", ""])
    print(f"saved: {csv_path}")

    # ---- plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(5)

    # Panel 1: scaled importance, colored by W sign
    bar_colors = ["#1f77b4" if w > 0 else "#d62728" for w in W]
    axes[0].bar(x, scaled_importance, color=bar_colors, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SIGNAL_NAMES, rotation=10)
    axes[0].set_ylabel(r"$|W_i|\cdot\mathrm{std}(x_i)$", fontsize=12)
    axes[0].set_title("Scaled feature importance", fontsize=13)
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(scaled_importance):
        sgn = "+" if W[i] > 0 else "−"
        axes[0].text(i, v, f"{sgn}{v:.3f}",
                     ha="center", va="bottom", fontsize=9)
    # legend hint for sign
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color="#1f77b4", label="positive W (encourages revision)"),
        Patch(color="#d62728", label="negative W (encourages persistence)"),
    ], fontsize=9, loc="best")

    # Panel 2: Pearson correlation between each signal and rho
    corr_colors = ["#2ca02c" if c > 0 else "#d62728" for c in corrs]
    axes[1].bar(x, corrs, color=corr_colors, edgecolor="black", linewidth=0.6)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SIGNAL_NAMES, rotation=10)
    axes[1].set_ylabel(r"Pearson $\mathrm{corr}(x_i, \rho_t)$", fontsize=12)
    axes[1].set_title(r"Per-signal correlation with $\rho_t$", fontsize=13)
    axes[1].set_ylim(-1, 1)
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(corrs):
        axes[1].text(i, v, f"{v:.2f}",
                     ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=9)
    axes[1].legend(handles=[
        Patch(color="#2ca02c", label=r"positive corr ($\uparrow x \Rightarrow \uparrow \rho$)"),
        Patch(color="#d62728", label=r"negative corr ($\uparrow x \Rightarrow \downarrow \rho$)"),
    ], fontsize=9, loc="best")

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "feature_contribution.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
