"""
Evaluation for the binary-A ToM coherence model (after the four-point refactor).

Metrics:
  1) Image retrieval (Recall@1/5/10, MRR) -- dialogue-level
        Aggregate per-turn q_t -> dialogue vector, score against gallery v^cls.

  2) Evidence stability & switching (intrinsic; no ground truth needed)
        KL_t = KL(e_{t-1} || e_t), bucketed by rho_t.
        - stability_index = mean KL on stable turns (rho_t < 0.5)
        - switching_index = mean KL on repair turns (rho_t >= 0.5)
        - switching_ratio = switching / stability
        - corr(rho_t, KL_t)  (also reported in (3))

  3) Repair sparsity & shift coupling
        - mean rho, P(rho > 0.5)
        - histogram bins
        - Pearson corr(rho_t, KL_t)
        - (optional) AUC vs binary repair label if any labels exist

Outputs:
  <out_dir>/metrics.json
  <out_dir>/rho_hist.png
  <out_dir>/kl_vs_rho.png
  <out_dir>/per_turn.csv

Usage:
  python -m src.eval.eval_tom_model \
      --checkpoint outputs/mvp_turnlevel_semisup/mvp_turnlevel_semisup.pt \
      --eval_shard data/photochat/train_json/train_02.json \
      --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
      --out_dir outputs/eval_run1
"""

import os
import json
import argparse
import csv
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.clip_model import CLIPBackbone
from src.models.alignment import AlignmentHead
from src.models.mvp_aligner import MVPAligner, DCPGate, MSCPGate
from src.train.train_mvp_wa_wh import run_dialogue
from src.dataloaders.photochat import (
    TURN_WINDOW_CHOICES,
    TURN_WINDOW_FULL,
    select_dialogue_window,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def load_image_map(path: str) -> Dict[str, str]:
    """photo_id -> image_path"""
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            m[rec["photo_id"]] = rec["image_path"]
    return m


def load_eval_shard(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_soft_target(label):
    if label == 0:
        return 1.0
    elif label == 1:
        return 0.4
    elif label == 2:
        return 0.0
    return None



@torch.no_grad()
def encode_dialogue(
    clip: CLIPBackbone,
    align: AlignmentHead,
    gate_module,
    gate_variant: str,
    image_path: str,
    turns: List[str],
    fixed_rho: float | None = None,
    ablate_image: str = "none",   # {none, zero, random, mean}
    ablate_text: str = "none",    # {none, zero, random, mean}
    mean_v_cls: torch.Tensor | None = None,
    mean_v_patches: torch.Tensor | None = None,
    mean_h: torch.Tensor | None = None,
):
    pil_img = Image.open(image_path).convert("RGB")
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

    # ---- modality ablation (applied AFTER CLIP encoding, BEFORE the model) ----
    if ablate_image == "zero":
        v_cls = torch.zeros_like(v_cls)
        v_patches = torch.zeros_like(v_patches)
    elif ablate_image == "random":
        v_cls = torch.randn_like(v_cls)
        v_patches = torch.randn_like(v_patches)
    elif ablate_image == "mean":
        assert mean_v_cls is not None and mean_v_patches is not None
        v_cls = mean_v_cls.to(v_cls.device)
        v_patches = mean_v_patches.to(v_patches.device)

    if ablate_text == "zero":
        h_all = torch.zeros_like(h_all)
    elif ablate_text == "random":
        h_all = torch.randn_like(h_all)
    elif ablate_text == "mean":
        assert mean_h is not None
        h_all = mean_h.to(h_all.device).unsqueeze(0).expand(h_all.shape[0], -1).contiguous()

    out = run_dialogue(
        h_all=h_all,
        v_cls=v_cls,
        v_patches=v_patches,
        align=align,
        gate_module=gate_module,
        gate_variant=gate_variant,
        fixed_rho=fixed_rho,
    )
    out["v_cls"] = v_cls
    return out


@torch.no_grad()
def precompute_modality_means(
    clip: CLIPBackbone,
    image_map: Dict[str, str],
    shard_dialogues,
    turn_window: str = TURN_WINDOW_FULL,
    max_samples: int = 200,
):
    """Compute mean v_cls / v_patches / h across (up to) max_samples dialogues."""
    v_cls_list, v_patches_list, h_list = [], [], []
    n = 0
    for ex in shard_dialogues:
        if n >= max_samples:
            break
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue", [])
        if not photo_id or not dialogue:
            continue
        dialogue, _ = select_dialogue_window(dialogue, turn_window)
        if not dialogue:
            continue
        if photo_id not in image_map:
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
        v_cls_list.append(v_cls_b[0].cpu())
        v_patches_list.append(v_patches_b[0].cpu())
        h_list.append(h_all.mean(dim=0).cpu())  # average across turns of this dialogue
        n += 1
    if n == 0:
        return None, None, None
    return (
        torch.stack(v_cls_list).mean(dim=0),
        torch.stack(v_patches_list).mean(dim=0),
        torch.stack(h_list).mean(dim=0),
    )


# ----------------------------------------------------------------------
# Metric (1): retrieval
# ----------------------------------------------------------------------
def dialogue_query(q_seq: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """q_seq: [T, D] -> [D]"""
    if mode == "mean":
        return q_seq.mean(dim=0)
    if mode == "last":
        return q_seq[-1]
    if mode == "max":
        return q_seq.max(dim=0).values
    raise ValueError(mode)


def retrieval_metrics(
    queries: torch.Tensor,        # [N, D]
    gallery: torch.Tensor,        # [G, D]
    target_idx: torch.Tensor,     # [N] long, position of correct gallery item per query
    ks=(1, 5, 10),
    return_ranks: bool = False,
):
    """
    Tie-aware retrieval ranking using the standard MID-RANK convention:
        rank(target) = (#strictly_higher) + (#tied_including_self + 1) / 2
    """
    q = F.normalize(queries, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.t()                                                   # [N, G]
    target_scores = sim.gather(1, target_idx.unsqueeze(1)).squeeze(1) # [N]

    strictly_higher = (sim > target_scores.unsqueeze(1)).float().sum(dim=1)   # [N]
    tied_incl_self  = (sim == target_scores.unsqueeze(1)).float().sum(dim=1)  # [N], >=1
    ranks = strictly_higher + (tied_incl_self + 1.0) / 2.0            # mid-rank, 1-indexed

    out = {f"R@{k}": float((ranks <= k).float().mean().item()) for k in ks}
    out["MRR"] = float((1.0 / ranks.float()).mean().item())
    out["median_rank"] = float(ranks.float().median().item())
    out["N_queries"] = int(queries.shape[0])
    out["N_gallery"] = int(gallery.shape[0])
    if return_ranks:
        return out, ranks.detach().cpu().numpy()
    return out


# ----------------------------------------------------------------------
# Metric (2)+(3): per-turn intrinsic stats
# ----------------------------------------------------------------------
def per_turn_kl(e_seq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(e_{t-1} || e_t) for t>=1, returns [T-1]."""
    p = e_seq[:-1]
    q = e_seq[1:]
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auc_binary(scores: np.ndarray, labels_binary: np.ndarray) -> float:
    """Mann-Whitney U / pairwise AUC. Labels in {0,1}."""
    pos = scores[labels_binary == 1]
    neg = scores[labels_binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    sum_pos_ranks = ranks[all_labels == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------
# Plotting helpers (matplotlib only)
# ----------------------------------------------------------------------
def plot_rho_hist(rhos: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(rhos, bins=30)
    plt.xlabel("rho_t")
    plt.ylabel("count")
    plt.title(f"Repair-gate distribution (mean={rhos.mean():.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_kl_vs_rho(rhos: np.ndarray, kls: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(rhos, kls, s=8, alpha=0.5)
    plt.xlabel("rho_t")
    plt.ylabel("KL(e_{t-1} || e_t)")
    plt.title(f"Shift coupling (Pearson={pearson(rhos, kls):.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--eval_shard", type=str, default=None,
                    help="single JSON shard (legacy). Use --split_manifest instead "
                         "for the proper held-out test split.")
    ap.add_argument("--split_manifest", type=str, default=None,
                    help="Path to split manifest. Eval will use ALL dialogues "
                         "from manifest['test_shards'] (concatenated).")
    ap.add_argument("--image_map_jsonl", type=str,
                    default="data/photochat/train_image_photo_desc.jsonl")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_dialogues", type=int, default=None)
    ap.add_argument("--query_mode", type=str, default="mean",
                    choices=["mean", "last", "max"])
    ap.add_argument("--turn_window", type=str, default=TURN_WINDOW_FULL,
                    choices=TURN_WINDOW_CHOICES,
                    help="Which dialogue turns to evaluate on. full keeps all turns; "
                         "preveal keeps only turns before the first share_photo=True "
                         "event (before the image is shown).")
    ap.add_argument("--rho_threshold", type=float, default=0.5,
                    help="rho >= threshold counts as 'repair' for bucketed metrics")

    # Modality ablation
    ap.add_argument("--ablate_image", choices=["none", "zero", "random", "mean"],
                    default="none")
    ap.add_argument("--ablate_text", choices=["none", "zero", "random", "mean"],
                    default="none")
    ap.add_argument("--mean_max_samples", type=int, default=200,
                    help="how many test dialogues to average for ablate=mean")

    args = ap.parse_args()
    if not args.eval_shard and not args.split_manifest:
        raise SystemExit("must provide --eval_shard or --split_manifest")

    os.makedirs(args.out_dir, exist_ok=True)

    # --------- load model ---------
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    clip = CLIPBackbone(device=DEVICE)
    align = AlignmentHead(
        h_dim=clip.text_dim, v_dim=clip.vision_dim, q_dim=clip.vision_dim
    ).to(DEVICE)
    align.load_state_dict(ckpt["alignment_head"])
    align.eval()

    # back-compat: old checkpoints used "repair_gate" key
    gate_variant = ckpt.get("gate_variant", "supervised_A")
    fixed_rho_eval = None
    if gate_variant == "supervised_A":
        gate_module = MVPAligner(text_dim=clip.text_dim, num_patches=49).to(DEVICE)
        sd = ckpt.get("gate_module", ckpt.get("repair_gate"))
        gate_module.load_state_dict(sd)
        gate_module.eval()
    elif gate_variant == "dcp_B":
        ck_args = ckpt.get("args", {})
        gate_module = DCPGate(
            init_a=ck_args.get("dcp_init_a", 5.0),
            init_mu=ck_args.get("dcp_init_mu", 0.5),
            ema_momentum=ck_args.get("dcp_ema", 0.1),
        ).to(DEVICE)
        gate_module.load_state_dict(ckpt["gate_module"])
        gate_module.eval()
    elif gate_variant == "multi_C":
        gate_module = MSCPGate().to(DEVICE)
        gate_module.load_state_dict(ckpt["gate_module"])
        gate_module.eval()
    elif gate_variant == "fixed":
        gate_module = None
        fixed_rho_eval = ckpt.get("args", {}).get("rho", 0.1)
    else:
        raise ValueError(f"unknown gate_variant in checkpoint: {gate_variant}")
    print(f"[eval] loaded gate_variant={gate_variant}")

    # --------- load data ---------
    image_map = load_image_map(args.image_map_jsonl)
    if args.split_manifest:
        with open(args.split_manifest) as _f:
            _manifest = json.load(_f)
        test_shard_paths = [
            os.path.join(_manifest["shards_dir"], s) for s in _manifest["test_shards"]
        ]
        print(f"[eval] using split manifest: {args.split_manifest}")
        print(f"[eval] test_shards: {[os.path.basename(p) for p in test_shard_paths]}")
        shard = []
        for sp in test_shard_paths:
            shard.extend(load_eval_shard(sp))
    else:
        shard = load_eval_shard(args.eval_shard)
    print(f"[eval] turn_window: {args.turn_window}")
    print(f"[eval] total dialogues in shard(s): {len(shard)}")

    # --------- precompute means if any ablation needs them ---------
    mean_v_cls = mean_v_patches = mean_h = None
    if args.ablate_image == "mean" or args.ablate_text == "mean":
        print(f"[eval] precomputing modality means from up to {args.mean_max_samples} dialogues...")
        mean_v_cls, mean_v_patches, mean_h = precompute_modality_means(
            clip, image_map, shard,
            turn_window=args.turn_window,
            max_samples=args.mean_max_samples,
        )
        if mean_v_cls is None:
            raise SystemExit("[eval] could not precompute means: no valid dialogues")
        print(f"[eval] mean_v_cls shape: {tuple(mean_v_cls.shape)}, "
              f"mean_h shape: {tuple(mean_h.shape)}")

    # --------- gather per-dialogue + per-turn data ---------
    dialogue_queries = []   # for retrieval, [N, D]
    dialogue_targets = []   # gallery index for each dialogue
    gallery_v_cls = {}      # photo_id -> v_cls tensor
    photo_id_order = []     # ordered list of unique photo_ids

    per_turn_records = []   # rho, kl, label (or -1)
    label_aucs_pool_rho = []
    label_aucs_pool_y = []

    n_processed = 0
    for ex in shard:
        if args.max_dialogues is not None and n_processed >= args.max_dialogues:
            break
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue", [])
        if not photo_id or not dialogue:
            continue
        dialogue, reveal_turn_idx = select_dialogue_window(dialogue, args.turn_window)
        if not dialogue:
            continue
        if photo_id not in image_map:
            continue
        image_path = image_map[photo_id]
        if not os.path.exists(image_path):
            continue

        turns = [t.get("message", "") for t in dialogue]
        if not turns:
            continue
        # original repair labels (if any) for AUC
        turn_labels = [t.get("label", None) for t in dialogue]
        turn_targets = [label_to_soft_target(l) for l in turn_labels]

        try:
            out = encode_dialogue(
                clip, align, gate_module, gate_variant,
                image_path, turns, fixed_rho=fixed_rho_eval,
                ablate_image=args.ablate_image,
                ablate_text=args.ablate_text,
                mean_v_cls=mean_v_cls,
                mean_v_patches=mean_v_patches,
                mean_h=mean_h,
            )
        except Exception as e:
            print(f"[skip] {photo_id}: {e}")
            continue

        q_seq = out["q_seq"]            # [T, D]
        e_seq = out["e_seq"]            # [T, N]
        rho_seq = out["rho_seq"]        # [T]
        a_seq = out["a_seq"]            # [T]
        a_tilde_seq = out["a_tilde_seq"]# [T]
        d_seq = out["d_seq"]            # [T]
        v_cls = out["v_cls"]            # [D]

        # ---- retrieval ----
        if photo_id not in gallery_v_cls:
            gallery_v_cls[photo_id] = v_cls.detach().cpu()
            photo_id_order.append(photo_id)

        q_dlg = dialogue_query(q_seq, mode=args.query_mode).detach().cpu()
        dialogue_queries.append(q_dlg)
        dialogue_targets.append(photo_id_order.index(photo_id))

        # ---- per-turn intrinsic ----
        kls = per_turn_kl(e_seq).detach().cpu().numpy()
        rho_pairs = rho_seq[1:].detach().cpu().numpy()  # align with kls (t>=1)
        rho_all = rho_seq.detach().cpu().numpy()
        a_all = a_seq.detach().cpu().numpy()
        a_tilde_all = a_tilde_seq.detach().cpu().numpy()
        d_all = d_seq.detach().cpu().numpy()

        for t, (r, k) in enumerate(zip(rho_pairs, kls), start=1):
            tgt = turn_targets[t] if t < len(turn_targets) else None
            per_turn_records.append({
                "photo_id": photo_id,
                "t": t,
                "turn_window": args.turn_window,
                "reveal_turn_idx": reveal_turn_idx,
                "rho": float(r),
                "kl": float(k),
                "a_t": float(a_all[t]),
                "a_tilde": float(a_tilde_all[t]),
                "d_t": float(d_all[t]),
                "soft_target": tgt,
            })
            if tgt is not None:
                # binary collapse: repair (1.0) vs others (0.0/0.4)
                label_aucs_pool_y.append(1 if tgt >= 0.9 else 0)
                label_aucs_pool_rho.append(float(r))

        # also include t=0 (no kl, but we still have rho/a/d/a_tilde at t=0)
        per_turn_records.append({
            "photo_id": photo_id,
            "t": 0,
            "turn_window": args.turn_window,
            "reveal_turn_idx": reveal_turn_idx,
            "rho": float(rho_all[0]),
            "kl": float("nan"),
            "a_t": float(a_all[0]),
            "a_tilde": float(a_tilde_all[0]),
            "d_t": float(d_all[0]),
            "soft_target": turn_targets[0] if turn_targets else None,
        })

        n_processed += 1

    if n_processed == 0:
        print("No dialogues evaluated.")
        return

    # --------- (1) retrieval ---------
    queries = torch.stack(dialogue_queries, dim=0)
    gallery = torch.stack([gallery_v_cls[p] for p in photo_id_order], dim=0)
    targets = torch.tensor(dialogue_targets, dtype=torch.long)
    ret, ranks_arr = retrieval_metrics(
        queries, gallery, targets, ks=(1, 5, 10), return_ranks=True
    )
    # Save per-dialogue ranks so plot_rk_curve.py can compute R@K for arbitrary K
    np.save(os.path.join(args.out_dir, "ranks.npy"), ranks_arr)

    # --------- (2) stability / switching ---------
    rho_arr = np.array([r["rho"] for r in per_turn_records if not np.isnan(r["kl"])])
    kl_arr = np.array([r["kl"] for r in per_turn_records if not np.isnan(r["kl"])])
    stable_mask = rho_arr < args.rho_threshold
    repair_mask = ~stable_mask
    stability_idx = float(kl_arr[stable_mask].mean()) if stable_mask.any() else float("nan")
    switching_idx = float(kl_arr[repair_mask].mean()) if repair_mask.any() else float("nan")
    switching_ratio = (
        float(switching_idx / stability_idx)
        if stable_mask.any() and repair_mask.any() and stability_idx > 1e-12
        else float("nan")
    )
    coupling = pearson(rho_arr, kl_arr)

    # --------- (3) sparsity ---------
    rho_all_arr = np.array([r["rho"] for r in per_turn_records])
    sparsity = {
        "mean_rho": float(rho_all_arr.mean()),
        "p_rho_gt_05": float((rho_all_arr > 0.5).mean()),
        "p_rho_gt_07": float((rho_all_arr > 0.7).mean()),
        "rho_quantiles": {
            "q25": float(np.quantile(rho_all_arr, 0.25)),
            "q50": float(np.quantile(rho_all_arr, 0.50)),
            "q75": float(np.quantile(rho_all_arr, 0.75)),
        },
    }

    # optional: AUC vs human repair label
    if label_aucs_pool_rho:
        rho_lab = np.array(label_aucs_pool_rho)
        y_lab = np.array(label_aucs_pool_y)
        repair_auc = auc_binary(rho_lab, y_lab)
        repair_pos_count = int((y_lab == 1).sum())
        repair_neg_count = int((y_lab == 0).sum())
    else:
        repair_auc = None
        repair_pos_count = 0
        repair_neg_count = 0

    metrics = {
        "args": vars(args),
        "gate_variant": gate_variant,
        "checkpoint_args": ckpt.get("args", {}),
        "turn_window": args.turn_window,
        "n_dialogues": int(n_processed),
        "n_turns_evaluated": int(len(per_turn_records)),
        "retrieval": ret,
        "stability_switching": {
            "stability_idx": stability_idx,
            "switching_idx": switching_idx,
            "switching_ratio": switching_ratio,
            "rho_kl_pearson": coupling,
            "rho_threshold": args.rho_threshold,
            "n_stable_turns": int(stable_mask.sum()),
            "n_repair_turns": int(repair_mask.sum()),
        },
        "repair_sparsity": sparsity,
        "repair_label_auc": repair_auc,
        "repair_label_n_pos": repair_pos_count,
        "repair_label_n_neg": repair_neg_count,
    }

    # --------- write outputs ---------
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out_dir, "per_turn.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "photo_id", "t", "turn_window", "reveal_turn_idx",
            "rho", "kl", "a_t", "a_tilde", "d_t", "soft_target",
        ])
        for r in per_turn_records:
            w.writerow([
                r["photo_id"], r["t"], r["turn_window"], r["reveal_turn_idx"],
                r["rho"], r["kl"],
                r["a_t"], r["a_tilde"], r["d_t"], r["soft_target"],
            ])

    plot_rho_hist(rho_all_arr, os.path.join(args.out_dir, "rho_hist.png"))
    plot_kl_vs_rho(rho_arr, kl_arr, os.path.join(args.out_dir, "kl_vs_rho.png"))

    # --------- print summary ---------
    print("\n========== SUMMARY ==========")
    print(f"dialogues:       {n_processed}")
    print(f"turns evaluated: {len(per_turn_records)}")
    print()
    print("[1] Image retrieval")
    print(f"    R@1 = {ret['R@1']:.4f}   R@5 = {ret['R@5']:.4f}   "
          f"R@10 = {ret['R@10']:.4f}   MRR = {ret['MRR']:.4f}   "
          f"median_rank = {ret['median_rank']:.1f}   |gallery| = {ret['N_gallery']}")
    print()
    print("[2] Evidence stability / switching")
    print(f"    stability_idx (KL on stable turns):   {stability_idx:.5f}")
    print(f"    switching_idx (KL on repair turns):   {switching_idx:.5f}")
    print(f"    switching_ratio (switching/stability): {switching_ratio:.3f}")
    print(f"    Pearson(rho, KL):                      {coupling:.4f}")
    print(f"    n_stable={int(stable_mask.sum())}  n_repair={int(repair_mask.sum())}")
    print()
    print("[3] Repair sparsity / coupling")
    print(f"    mean(rho)        = {sparsity['mean_rho']:.4f}")
    print(f"    P(rho > 0.5)     = {sparsity['p_rho_gt_05']:.4f}")
    print(f"    P(rho > 0.7)     = {sparsity['p_rho_gt_07']:.4f}")
    print(f"    rho q25/q50/q75  = "
          f"{sparsity['rho_quantiles']['q25']:.3f} / "
          f"{sparsity['rho_quantiles']['q50']:.3f} / "
          f"{sparsity['rho_quantiles']['q75']:.3f}")
    if repair_auc is not None:
        print(f"    AUC(rho vs human-label binary): {repair_auc:.4f}  "
              f"(pos={repair_pos_count}, neg={repair_neg_count})")
    print()
    print(f"All outputs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
