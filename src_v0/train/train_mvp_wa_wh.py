"""
Train script for the simplified ToM-as-Action visual-language coherence model.

Supports three repair-gate variants (set via --gate_variant):

  fixed         : rho_t = constant (--rho)                          baseline
  supervised_A  : rho_t = sigmoid(W_p [h_t; e_{t-1}; a_{t-1}])      report Sec. 1.2
                  trained with weighted MSE on human/predicted labels
  dcp_B         : rho_t = sigmoid(a * (d_t - mu))                   report Sec. 1.3
                  d_t = KL(e_{t-1} || e_hat_t), trained with sparsity prior L_rate

All variants share the same backbone & evidence dynamics:
  q_t = W_h h_t                                       AlignmentHead.Wa_text
  e_hat_t = softmax(q_t @ v_patches)
  e_t = (1-rho_t) e_{t-1} + rho_t e_hat_t             evidence dynamics
  c_t = e_t @ v_patches
  a_tilde_t = sigmoid(W_a [q_t; c_t; v_cls])          AlignmentHead.W_a
  a_t = (1-(1-rho_t)) a_{t-1} + (1-rho_t) a_tilde_t   coherence trajectory  (Eq. 11)

Loss:
  fixed         L_corr + alpha L_stab + beta L_switch
  supervised_A  L_corr + alpha L_stab + beta L_switch + gamma_repair L_repair
  dcp_B         L_corr + alpha L_stab + beta L_switch + gamma_rate   L_rate
"""

import os
import argparse
import random
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.dataloaders.photochat import PhotoChatDataset, collate_fn
from src.models.clip_model import CLIPBackbone
from src.models.alignment import AlignmentHead
from src.models.mvp_aligner import MVPAligner, DCPGate, MSCPGate
from src.models.losses import (
    masked_corr_loss,
    stab_switch_loss,
    weighted_repair_mse,
    rate_loss,
    coherence_floor_loss,
    coherence_drop_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 1e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_open_image(path):
    return Image.open(path).convert("RGB")


# ------------------------------------------------------------------
# Per-dialogue sequential differentiable forward
# ------------------------------------------------------------------
def kl_pq_scalar(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """KL(p || q) for two 1-D distributions, returns 0-d tensor."""
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum()


def run_dialogue(
    h_all: torch.Tensor,
    v_cls: torch.Tensor,
    v_patches: torch.Tensor,
    align: AlignmentHead,
    gate_module,                # MVPAligner | DCPGate | None
    gate_variant: str,
    fixed_rho: float | None = None,
):
    """
    Sequential per-turn forward. Everything differentiable.
    Returns dict with q_seq, e_seq, a_tilde_seq, a_seq, rho_seq, d_seq.
    """
    T = h_all.shape[0]
    N = v_patches.shape[0]
    device = h_all.device

    q_seq = align.project_query(h_all)               # [T, q_dim]
    e_hat_seq = align.raw_evidence(q_seq, v_patches) # [T, N]

    e_prev = torch.full((N,), 1.0 / N, device=device)
    a_prev = torch.full((), 0.5, device=device)

    e_list, a_tilde_list, a_list, rho_list, d_list = [], [], [], [], []

    for t in range(T):
        e_hat_t = e_hat_seq[t]

        # d_t = KL(e_{t-1} || e_hat_t)  -- always computed (used by DCP, also a useful diagnostic)
        d_t = kl_pq_scalar(e_prev, e_hat_t)
        d_list.append(d_t)

        # ---- repair gate ----
        if gate_variant == "fixed":
            rho_t = torch.tensor(float(fixed_rho), device=device)
        elif gate_variant == "supervised_A":
            rho_t = gate_module(h_all[t], e_prev, a_prev)
        elif gate_variant == "dcp_B":
            rho_t = gate_module(d_t)
        elif gate_variant == "multi_C":
            # forward-looking alignment if we fully trusted the new evidence
            # (uses e_hat_t, NOT e_t, so no circular dependency on rho_t)
            c_hat_t = e_hat_t @ v_patches
            a_hat_t = align.alignment_score(
                q_seq[t : t + 1], c_hat_t.unsqueeze(0), v_cls
            ).squeeze()
            delta_a_hat = a_hat_t - a_prev
            # entropies (uncertainty signals)
            h_e_prev = -(e_prev * torch.log(e_prev + EPS)).sum()
            h_e_hat = -(e_hat_t * torch.log(e_hat_t + EPS)).sum()
            signals = torch.stack(
                [d_t, a_prev, delta_a_hat, h_e_prev, h_e_hat], dim=0
            )
            rho_t = gate_module(signals)
        else:
            raise ValueError(f"unknown gate_variant: {gate_variant}")

        # ---- evidence dynamics ----
        e_t_unnorm = (1.0 - rho_t) * e_prev + rho_t * e_hat_t
        e_t = e_t_unnorm / (e_t_unnorm.sum() + 1e-12)

        # ---- alignment & coherence ----
        c_t = e_t @ v_patches
        a_tilde_t = align.alignment_score(
            q_seq[t : t + 1], c_t.unsqueeze(0), v_cls
        ).squeeze()
        lam_t = 1.0 - rho_t
        a_t = (1.0 - lam_t) * a_prev + lam_t * a_tilde_t

        rho_list.append(rho_t)
        e_list.append(e_t)
        a_tilde_list.append(a_tilde_t)
        a_list.append(a_t)

        e_prev = e_t
        a_prev = a_t

    return {
        "q_seq": q_seq,
        "e_seq": torch.stack(e_list, dim=0),
        "a_tilde_seq": torch.stack(a_tilde_list, dim=0),
        "a_seq": torch.stack(a_list, dim=0),
        "rho_seq": torch.stack(rho_list, dim=0),
        "d_seq": torch.stack(d_list, dim=0),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs/run_dcp")
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--gate_variant",
        type=str,
        default="dcp_B",
        choices=["fixed", "supervised_A", "dcp_B", "multi_C"],
        help="fixed: constant rho. supervised_A: state-aware learned, label-supervised. "
             "dcp_B: KL change-point gate, no labels. "
             "multi_C: multi-signal gate (d_t + state + delta_a + entropies), no labels.",
    )
    ap.add_argument("--rho", type=float, default=0.1, help="for gate_variant=fixed")

    # DCP-Gate specifics
    ap.add_argument("--dcp_init_a", type=float, default=5.0)
    ap.add_argument("--dcp_init_mu", type=float, default=0.5)
    ap.add_argument("--dcp_ema", type=float, default=0.1)
    ap.add_argument("--target_repair_rate", type=float, default=0.15,
                    help="r in L_rate = (mean(rho)-r)^2")

    # loss weights
    ap.add_argument("--alpha_stab", type=float, default=1.0)
    ap.add_argument("--beta_switch", type=float, default=1.0)
    ap.add_argument("--gamma_repair", type=float, default=1.0,
                    help="weight on weighted-MSE repair loss (used iff supervised_A)")
    ap.add_argument("--gamma_rate", type=float, default=1.0,
                    help="weight on L_rate (used iff dcp_B)")
    ap.add_argument("--switch_margin", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)

    # Optional coherence regularizers (default off)
    ap.add_argument("--gamma_coh_floor", type=float, default=0.0,
                    help="weight on L_coh_floor = mean(max(0, tau - a_t)^2)")
    ap.add_argument("--tau_coh_floor", type=float, default=0.15,
                    help="floor threshold for coherence trajectory")
    ap.add_argument("--gamma_coh_drop", type=float, default=0.0,
                    help="weight on L_coh_drop (asymmetric, penalize big drops)")
    ap.add_argument("--epsilon_coh_drop", type=float, default=0.05,
                    help="tolerance for natural coherence drops")

    # data flags (for variants that don't need labels we can skip predicted_train03)
    ap.add_argument("--use_predicted_labels", action="store_true",
                    help="Load predicted_train03.json. Default off; only useful for supervised_A.")
    ap.add_argument("--split_manifest", type=str, default=None,
                    help="Path to JSON manifest from src.dataloaders.make_split. "
                         "If given, training uses ALL dialogues from manifest['train_shards'] "
                         "(overrides legacy first-100-of-train_00/01 loading).")
    ap.add_argument("--image_map_jsonl", type=str,
                    default="data/photochat/train_image_photo_desc.jsonl")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Decide whether to use labels at all
    needs_labels = (args.gate_variant == "supervised_A")
    use_predicted = args.use_predicted_labels or needs_labels

    if args.split_manifest:
        # NEW: full-data path with explicit train shard list
        import json as _json
        with open(args.split_manifest) as _f:
            _manifest = _json.load(_f)
        train_shard_paths = [
            os.path.join(_manifest["shards_dir"], s) for s in _manifest["train_shards"]
        ]
        print(f"[train] using split manifest: {args.split_manifest}")
        print(f"[train] train_shards ({len(train_shard_paths)}): "
              f"{[os.path.basename(p) for p in train_shard_paths]}")
        ds = PhotoChatDataset(
            image_map_jsonl=args.image_map_jsonl,
            shard_files=train_shard_paths,
            max_items=args.max_items,
        )
    else:
        ds = PhotoChatDataset(
            shards_dir="data/photochat/train_json",
            predicted_json="data/photochat/predicted_train03.json",
            image_map_jsonl=args.image_map_jsonl,
            use_human=True,
            use_predicted=use_predicted,
            human_limit_per_file=100,
            max_items=args.max_items,
        )
    print(f"[train] dataset size: {len(ds)} dialogues")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Frozen CLIP ----
    clip = CLIPBackbone(device=DEVICE)
    img_processor = clip.processor.image_processor
    tokenizer = clip.processor.tokenizer

    # ---- AlignmentHead (always trainable) ----
    align = AlignmentHead(
        h_dim=clip.text_dim, v_dim=clip.vision_dim, q_dim=clip.vision_dim
    ).to(DEVICE)

    # ---- Gate module ----
    if args.gate_variant == "supervised_A":
        gate_module = MVPAligner(
            text_dim=clip.text_dim, num_patches=49
        ).to(DEVICE)
        gate_params = list(gate_module.parameters())
    elif args.gate_variant == "dcp_B":
        gate_module = DCPGate(
            init_a=args.dcp_init_a,
            init_mu=args.dcp_init_mu,
            ema_momentum=args.dcp_ema,
        ).to(DEVICE)
        gate_params = list(gate_module.parameters())  # only log_a
    elif args.gate_variant == "multi_C":
        gate_module = MSCPGate().to(DEVICE)
        gate_params = list(gate_module.parameters())
    else:
        gate_module = None
        gate_params = []

    params = list(align.parameters()) + gate_params
    opt = torch.optim.AdamW(params, lr=args.lr)

    # ---- logging ----
    log_keys = [
        "total", "corr", "stab", "switch", "repair", "rate",
        "coh_floor", "coh_drop", "mean_rho", "mean_a", "mu",
    ]
    history = {k: [] for k in log_keys}

    for ep in range(args.epochs):
        align.train()
        if gate_module is not None:
            gate_module.train()

        running = {k: 0.0 for k in log_keys}
        steps = 0

        for image_paths, turns_batch, metas in dl:
            dialogue_outputs = []

            for image_path, turns_i, meta_i in zip(image_paths, turns_batch, metas):
                if not turns_i:
                    continue

                try:
                    pil_img = safe_open_image(image_path)
                except Exception as e:
                    print(f"[WARN] Skipping image: {image_path} | {e}")
                    continue

                pixel_values = img_processor(
                    images=[pil_img], return_tensors="pt"
                )["pixel_values"].to(DEVICE)

                text_inputs = tokenizer(
                    turns_i, padding=True, truncation=True, return_tensors="pt", max_length=77
                ).to(DEVICE)

                with torch.no_grad():
                    v_cls_b, v_patches_b = clip.encode_vision_tokens(pixel_values)
                    h_all = clip.encode_text_pooler(
                        text_inputs["input_ids"], text_inputs["attention_mask"]
                    )

                v_cls = v_cls_b[0]
                v_patches = v_patches_b[0]

                fixed_rho = args.rho if args.gate_variant == "fixed" else None
                out = run_dialogue(
                    h_all=h_all,
                    v_cls=v_cls,
                    v_patches=v_patches,
                    align=align,
                    gate_module=gate_module,
                    gate_variant=args.gate_variant,
                    fixed_rho=fixed_rho,
                )
                out["v_cls"] = v_cls
                out["repair_targets"] = meta_i.get("repair_targets", [])
                out["repair_weights"] = meta_i.get("repair_weights", [])
                dialogue_outputs.append(out)

            if not dialogue_outputs:
                continue

            # ---------- losses ----------
            # L_corr: q_t vs v^cls cross-dialogue
            q_pool = torch.cat([d["q_seq"] for d in dialogue_outputs], dim=0)
            v_cls_pool = torch.stack([d["v_cls"] for d in dialogue_outputs], dim=0)
            dialogue_ids = []
            for d_idx, d in enumerate(dialogue_outputs):
                dialogue_ids.extend([d_idx] * d["q_seq"].shape[0])
            dialogue_ids = torch.tensor(dialogue_ids, dtype=torch.long, device=DEVICE)

            if v_cls_pool.shape[0] >= 2:
                corr_loss = masked_corr_loss(
                    q_pool=q_pool, v_cls_pool=v_cls_pool,
                    dialogue_ids=dialogue_ids, mask=None,
                    temperature=args.temperature,
                )
            else:
                corr_loss = torch.tensor(0.0, device=DEVICE)

            # L_stab + L_switch
            stab_total = torch.tensor(0.0, device=DEVICE)
            switch_total = torch.tensor(0.0, device=DEVICE)
            for d in dialogue_outputs:
                s, sw = stab_switch_loss(
                    d["e_seq"], d["rho_seq"], delta=args.switch_margin
                )
                stab_total = stab_total + s
                switch_total = switch_total + sw
            n_d = len(dialogue_outputs)
            stab_loss = stab_total / n_d
            switch_loss = switch_total / n_d

            # variant-specific loss
            repair_loss = torch.tensor(0.0, device=DEVICE)
            r_loss = torch.tensor(0.0, device=DEVICE)

            if args.gate_variant == "supervised_A":
                rep_total = torch.tensor(0.0, device=DEVICE)
                n_lab = 0
                for d in dialogue_outputs:
                    rl = weighted_repair_mse(
                        d["rho_seq"], d["repair_targets"], d["repair_weights"]
                    )
                    if rl is not None:
                        rep_total = rep_total + rl
                        n_lab += 1
                if n_lab > 0:
                    repair_loss = rep_total / n_lab

            elif args.gate_variant in ("dcp_B", "multi_C"):
                rho_pool = torch.cat([d["rho_seq"] for d in dialogue_outputs], dim=0)
                r_loss = rate_loss(rho_pool, target_rate=args.target_repair_rate)

            # Optional coherence regularizers
            coh_floor_loss_val = torch.tensor(0.0, device=DEVICE)
            coh_drop_loss_val = torch.tensor(0.0, device=DEVICE)
            if args.gamma_coh_floor > 0.0 or args.gamma_coh_drop > 0.0:
                cf_total = torch.tensor(0.0, device=DEVICE)
                cd_total = torch.tensor(0.0, device=DEVICE)
                for d in dialogue_outputs:
                    if args.gamma_coh_floor > 0.0:
                        cf_total = cf_total + coherence_floor_loss(
                            d["a_seq"], tau=args.tau_coh_floor
                        )
                    if args.gamma_coh_drop > 0.0:
                        cd_total = cd_total + coherence_drop_loss(
                            d["a_seq"], epsilon=args.epsilon_coh_drop
                        )
                if args.gamma_coh_floor > 0.0:
                    coh_floor_loss_val = cf_total / n_d
                if args.gamma_coh_drop > 0.0:
                    coh_drop_loss_val = cd_total / n_d

            total_loss = (
                corr_loss
                + args.alpha_stab * stab_loss
                + args.beta_switch * switch_loss
                + args.gamma_repair * repair_loss
                + args.gamma_rate * r_loss
                + args.gamma_coh_floor * coh_floor_loss_val
                + args.gamma_coh_drop * coh_drop_loss_val
            )

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()

            # post-step EMA update for DCP mu
            if args.gate_variant == "dcp_B":
                with torch.no_grad():
                    d_pool = torch.cat([d["d_seq"] for d in dialogue_outputs], dim=0)
                    gate_module.update_mu(d_pool)

            running["total"] += float(total_loss.item())
            running["corr"] += float(corr_loss.item())
            running["stab"] += float(stab_loss.item())
            running["switch"] += float(switch_loss.item())
            running["repair"] += float(repair_loss.item())
            running["rate"] += float(r_loss.item())
            running["coh_floor"] += float(coh_floor_loss_val.item())
            running["coh_drop"] += float(coh_drop_loss_val.item())
            running["mean_rho"] += float(
                torch.cat([d["rho_seq"] for d in dialogue_outputs]).mean().item()
            )
            running["mean_a"] += float(
                torch.cat([d["a_seq"] for d in dialogue_outputs]).mean().item()
            )
            running["mu"] += (
                float(gate_module.mu.item()) if args.gate_variant == "dcp_B" else 0.0
            )
            steps += 1

        avg = {k: running[k] / max(1, steps) for k in log_keys}
        for k in log_keys:
            history[k].append(avg[k])

        msg = (
            f"[Epoch {ep+1}/{args.epochs}] "
            f"total={avg['total']:.4f} | corr={avg['corr']:.4f} | "
            f"stab={avg['stab']:.5f} | switch={avg['switch']:.5f}"
        )
        if args.gate_variant == "supervised_A":
            msg += f" | repair={avg['repair']:.4f}"
        if args.gate_variant in ("dcp_B", "multi_C"):
            msg += (
                f" | rate={avg['rate']:.5f}"
                f" | mean_rho={avg['mean_rho']:.3f}"
                f" | mean_a={avg['mean_a']:.3f}"
            )
            if args.gate_variant == "dcp_B":
                msg += f" | mu={avg['mu']:.3f}"
        if args.gamma_coh_floor > 0.0:
            msg += f" | coh_floor={avg['coh_floor']:.5f}"
        if args.gamma_coh_drop > 0.0:
            msg += f" | coh_drop={avg['coh_drop']:.5f}"
        print(msg)

        # write CSV every epoch
        with open(os.path.join(args.out_dir, "loss.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + log_keys)
            for i in range(len(history["total"])):
                w.writerow([i + 1] + [history[k][i] for k in log_keys])

    # save checkpoint
    ckpt = {
        "alignment_head": align.state_dict(),
        "gate_variant": args.gate_variant,
        "args": vars(args),
    }
    if gate_module is not None:
        ckpt["gate_module"] = gate_module.state_dict()
    torch.save(ckpt, os.path.join(args.out_dir, "model.pt"))

    print("\nDONE")
    print(f"- outputs: {args.out_dir}")
    print(f"- model:   {os.path.join(args.out_dir, 'model.pt')}")


if __name__ == "__main__":
    main()
