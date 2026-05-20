"""
Baselines for the retrieval evaluation.

Two modes (--mode):
  clip    : raw CLIP projected features (no training).
            Question answered: "is our retrieval gain over CLIP-only?"

  random  : theoretical random ranking. Lower bound floor.

Both write metrics.json in the same format as src.eval.eval_tom_model so they
can be dropped into src.eval.compare_runs alongside the trained variants.

Usage:
  python -m src.eval.baselines --mode clip   --split_manifest data/photochat/split_v1.json --out_dir outputs/baseline_clip
  python -m src.eval.baselines --mode random --split_manifest data/photochat/split_v1.json --out_dir outputs/baseline_random
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataloaders.photochat import (
    TURN_WINDOW_CHOICES,
    TURN_WINDOW_FULL,
    select_dialogue_window,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_features(out):

    if torch.is_tensor(out):
        return out

    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds

    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0]
    
    if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
        return out[0]

    raise TypeError(f"Cannot extract tensor features from output type: {type(out)}")


def load_image_map(path: str) -> Dict[str, str]:
    m = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            m[rec["photo_id"]] = rec["image_path"]
    return m


def load_split_dialogues(manifest_path: str) -> List[dict]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    out = []
    for s in manifest["test_shards"]:
        with open(os.path.join(manifest["shards_dir"], s)) as f:
            out.extend(json.load(f))
    return out


def retrieval_metrics(queries, gallery, target_idx, ks=(1, 5, 10)):
    q = F.normalize(queries, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.t()
    target_scores = sim.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    higher = (sim > target_scores.unsqueeze(1)).sum(dim=1)
    ranks = higher + 1
    out = {f"R@{k}": float((ranks <= k).float().mean().item()) for k in ks}
    out["MRR"] = float((1.0 / ranks.float()).mean().item())
    out["median_rank"] = float(ranks.float().median().item())
    out["N_queries"] = int(queries.shape[0])
    out["N_gallery"] = int(gallery.shape[0])
    return out


def run_clip_only(
    test_dialogues: List[dict],
    image_map: Dict[str, str],
    query_mode: str = "mean",
    turn_window: str = TURN_WINDOW_FULL,
):
    from src.models.clip_model import CLIPBackbone

    clip = CLIPBackbone(device=DEVICE)
    img_proc = clip.processor.image_processor
    tok = clip.processor.tokenizer

    text_vecs = []
    image_vecs = {}        
    photo_id_order = []
    target_idx = []

    n = 0
    for ex in test_dialogues:
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue", [])
        if not photo_id or not dialogue or photo_id not in image_map:
            continue
        dialogue, _ = select_dialogue_window(dialogue, turn_window)
        if not dialogue:
            continue
        image_path = image_map[photo_id]
        if not os.path.exists(image_path):
            continue
        turns = [t.get("message", "").strip() for t in dialogue]
        turns = [x for x in turns if x]
        if not turns:
            continue
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        with torch.no_grad():

            pixel_values = img_proc(images=[pil_img], return_tensors="pt")[
                "pixel_values"
            ].to(DEVICE)

            text_inputs = tok(
                turns,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=77,
            ).to(DEVICE)

            v_out = clip.model.get_image_features(pixel_values=pixel_values)
            v_emb = extract_features(v_out)[0]
            t_out = clip.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            t_emb_all = extract_features(t_out)

            if query_mode == "mean":
                text_vec = t_emb_all.mean(dim=0)
            elif query_mode == "last":
                text_vec = t_emb_all[-1]
            elif query_mode == "max":
                text_vec = t_emb_all.max(dim=0).values
            else:
                raise ValueError(query_mode)

        if photo_id not in image_vecs:
            image_vecs[photo_id] = v_emb.cpu()
            photo_id_order.append(photo_id)
        text_vecs.append(text_vec.cpu())
        target_idx.append(photo_id_order.index(photo_id))
        n += 1

    if n == 0:
        return None, 0
    queries = torch.stack(text_vecs)
    gallery = torch.stack([image_vecs[p] for p in photo_id_order])
    targets = torch.tensor(target_idx, dtype=torch.long)
    return retrieval_metrics(queries, gallery, targets), n


def run_random(
    test_dialogues: List[dict],
    image_map: Dict[str, str],
    turn_window: str = TURN_WINDOW_FULL,
):
    n_queries = 0
    photos = set()
    for ex in test_dialogues:
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue", [])
        if not photo_id or not dialogue or photo_id not in image_map:
            continue
        dialogue, _ = select_dialogue_window(dialogue, turn_window)
        if not dialogue:
            continue
        if not os.path.exists(image_map[photo_id]):
            continue
        if not [t.get("message", "") for t in dialogue]:
            continue
        n_queries += 1
        photos.add(photo_id)

    n_gallery = len(photos)
    if n_gallery == 0:
        return None, 0

    out = {f"R@{k}": float(min(k, n_gallery) / n_gallery) for k in (1, 5, 10)}
    harmonic = sum(1.0 / i for i in range(1, n_gallery + 1))
    out["MRR"] = float(harmonic / n_gallery)
    out["median_rank"] = float((n_gallery + 1) / 2.0)
    out["N_queries"] = int(n_queries)
    out["N_gallery"] = int(n_gallery)
    return out, n_queries


def write_metrics(out_dir: str, gate_variant: str, retrieval: dict, n_dialogues: int, args_dict: dict):
    metrics = {
        "args": args_dict,
        "gate_variant": gate_variant,
        "n_dialogues": int(n_dialogues),
        "n_turns_evaluated": 0,
        "retrieval": retrieval,
        "stability_switching": {
            "stability_idx": None, "switching_idx": None, "switching_ratio": None,
            "rho_kl_pearson": None, "rho_threshold": None,
            "n_stable_turns": 0, "n_repair_turns": 0,
        },
        "repair_sparsity": {
            "mean_rho": None, "p_rho_gt_05": None, "p_rho_gt_07": None,
            "rho_quantiles": {"q25": None, "q50": None, "q75": None},
        },
        "repair_label_auc": None,
        "repair_label_n_pos": 0,
        "repair_label_n_neg": 0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clip", "random"], required=True)
    ap.add_argument("--split_manifest", type=str, required=True)
    ap.add_argument("--image_map_jsonl", type=str,
                    default="data/photochat/train_image_photo_desc.jsonl")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--query_mode", default="mean", choices=["mean", "last", "max"])
    ap.add_argument("--turn_window", type=str, default=TURN_WINDOW_FULL,
                    choices=TURN_WINDOW_CHOICES,
                    help="Which dialogue turns to evaluate on. full keeps all turns; "
                         "preveal keeps only turns before the first share_photo=True "
                         "event (before the image is shown).")
    args = ap.parse_args()

    test_dialogues = load_split_dialogues(args.split_manifest)
    image_map = load_image_map(args.image_map_jsonl)

    if args.mode == "clip":
        ret, n = run_clip_only(
            test_dialogues, image_map,
            query_mode=args.query_mode,
            turn_window=args.turn_window,
        )
        gate_variant = "baseline_clip"
    else:
        ret, n = run_random(
            test_dialogues, image_map,
            turn_window=args.turn_window,
        )
        gate_variant = "baseline_random"

    if ret is None:
        raise SystemExit("[baselines] no valid dialogues evaluated")

    write_metrics(args.out_dir, gate_variant, ret, n, vars(args))

    print(f"\n[{gate_variant}]  turn_window={args.turn_window}  dialogues={n}, gallery={ret['N_gallery']}")
    print(f"  R@1 = {ret['R@1']:.4f}   R@5 = {ret['R@5']:.4f}   "
          f"R@10 = {ret['R@10']:.4f}   MRR = {ret['MRR']:.4f}   "
          f"median_rank = {ret['median_rank']:.1f}")
    print(f"  -> {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
