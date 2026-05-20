"""
Post-hoc stratified retrieval evaluation.

This keeps trained checkpoints fixed and reports R@1/R@5/R@10/MRR on the
same held-out test split, bucketed by dialogue length and repair rate.

Typical usage for one seed:

  python -m src.eval.eval_stratified \
      --seed_dir outputs/seed_0 \
      --split_manifest outputs/split_v1.json \
      --repair_bucket_run multi_C \
      --out_dir outputs/seed_0/stratified_v6

Or pass checkpoints explicitly:

  python -m src.eval.eval_stratified \
      --run fixed=outputs/seed_2/run_fixed_v6/model.pt \
      --run multi_C=outputs/seed_2/run_multi_C_v6/model.pt \
      --run pure_corr=outputs/seed_2/run_pure_corr_v6/model.pt \
      --split_manifest outputs/split_v1.json \
      --repair_bucket_run multi_C \
      --out_dir outputs/seed_2/stratified_v6
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from src.eval.eval_tom_model import (
    DEVICE,
    dialogue_query,
    encode_dialogue,
    load_eval_shard,
    load_image_map,
    retrieval_metrics,
)
from src.models.alignment import AlignmentHead
from src.models.clip_model import CLIPBackbone
from src.models.mvp_aligner import DCPGate, MSCPGate, MVPAligner


DEFAULT_SEED_RUNS = {
    "fixed": "run_fixed_v6/model.pt",
    "dcp_B": "run_dcp_v6/model.pt",
    "multi_C": "run_multi_C_v6/model.pt",
    "multi_C+floor": "run_multi_C_floor_v6/model.pt",
    "multi_C+floor+drop": "run_multi_C_floor_drop_v6/model.pt",
    "pure_corr": "run_pure_corr_v6/model.pt",
}

METRIC_KEYS = ("R@1", "R@5", "R@10", "MRR", "median_rank")


@dataclass
class LoadedModel:
    name: str
    checkpoint: str
    clip: CLIPBackbone
    align: AlignmentHead
    gate_module: object
    gate_variant: str
    fixed_rho: Optional[float]
    checkpoint_args: dict


def load_model(name: str, checkpoint: str) -> LoadedModel:
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    clip = CLIPBackbone(device=DEVICE)
    align = AlignmentHead(
        h_dim=clip.text_dim, v_dim=clip.vision_dim, q_dim=clip.vision_dim
    ).to(DEVICE)
    align.load_state_dict(ckpt["alignment_head"])
    align.eval()

    gate_variant = ckpt.get("gate_variant", "supervised_A")
    fixed_rho = None
    if gate_variant == "supervised_A":
        gate_module = MVPAligner(text_dim=clip.text_dim, num_patches=49).to(DEVICE)
        gate_module.load_state_dict(ckpt.get("gate_module", ckpt.get("repair_gate")))
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
        fixed_rho = ckpt.get("args", {}).get("rho", 0.1)
    else:
        raise ValueError(f"unknown gate_variant in {checkpoint}: {gate_variant}")

    return LoadedModel(
        name=name,
        checkpoint=checkpoint,
        clip=clip,
        align=align,
        gate_module=gate_module,
        gate_variant=gate_variant,
        fixed_rho=fixed_rho,
        checkpoint_args=ckpt.get("args", {}),
    )


def load_split_dialogues(manifest_path: str) -> List[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out = []
    for shard_name in manifest["test_shards"]:
        out.extend(load_eval_shard(os.path.join(manifest["shards_dir"], shard_name)))
    return out


def parse_runs(args) -> List[Tuple[str, str]]:
    runs = []
    if args.seed_dir:
        for name, rel_path in DEFAULT_SEED_RUNS.items():
            ckpt = os.path.join(args.seed_dir, rel_path)
            if os.path.exists(ckpt):
                runs.append((name, ckpt))
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must look like name=/path/to/model.pt, got: {item}")
        name, ckpt = item.split("=", 1)
        runs.append((name, ckpt))
    if not runs:
        raise SystemExit("provide --seed_dir or at least one --run name=checkpoint")

    seen = set()
    unique = []
    for name, ckpt in runs:
        if name in seen:
            raise SystemExit(f"duplicate run name: {name}")
        if not os.path.exists(ckpt):
            raise SystemExit(f"checkpoint not found for {name}: {ckpt}")
        seen.add(name)
        unique.append((name, ckpt))
    return unique


def dialogue_key(ex: dict, fallback_idx: int) -> str:
    if ex.get("dialogue_id"):
        return str(ex["dialogue_id"])
    if ex.get("photo_id"):
        return f"{ex['photo_id']}#{fallback_idx}"
    return f"idx:{fallback_idx}"


def label_repair_rate(dialogue: List[dict]) -> Optional[float]:
    labels = [t.get("label") for t in dialogue if t.get("label") is not None]
    if not labels:
        return None
    return float(sum(1 for x in labels if x == 0) / len(labels))


def bucket_metrics(ranks: Iterable[float], gallery_size: int) -> dict:
    ranks_arr = np.asarray(list(ranks), dtype=np.float64)
    if len(ranks_arr) == 0:
        return {
            "R@1": None,
            "R@5": None,
            "R@10": None,
            "MRR": None,
            "median_rank": None,
            "N_queries": 0,
            "N_gallery": int(gallery_size),
        }
    return {
        "R@1": float((ranks_arr <= 1).mean()),
        "R@5": float((ranks_arr <= 5).mean()),
        "R@10": float((ranks_arr <= 10).mean()),
        "MRR": float((1.0 / ranks_arr).mean()),
        "median_rank": float(np.median(ranks_arr)),
        "N_queries": int(len(ranks_arr)),
        "N_gallery": int(gallery_size),
    }


def median_split(values: Dict[str, float]) -> Tuple[float, Dict[str, str]]:
    arr = np.asarray(list(values.values()), dtype=np.float64)
    threshold = float(np.median(arr))
    groups = {
        key: ("top50" if value > threshold else "bottom50")
        for key, value in values.items()
    }

    # If many values equal the median, the <= / > split can be very uneven. Fall
    # back to rank halves while keeping deterministic ordering.
    n_top = sum(1 for x in groups.values() if x == "top50")
    if n_top == 0 or n_top == len(groups):
        ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
        cut = len(ordered) // 2
        groups = {
            key: ("bottom50" if i < cut else "top50")
            for i, (key, _) in enumerate(ordered)
        }
    return threshold, groups


@torch.no_grad()
def evaluate_run(
    model: LoadedModel,
    dialogues: List[dict],
    image_map: Dict[str, str],
    query_mode: str,
    max_dialogues: Optional[int],
):
    queries = []
    targets = []
    gallery_v_cls = {}
    photo_id_order = []
    records = {}
    n_processed = 0

    for i, ex in enumerate(dialogues):
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
        turns = [x for x in turns if x]
        if not turns:
            continue

        out = encode_dialogue(
            model.clip,
            model.align,
            model.gate_module,
            model.gate_variant,
            image_path,
            turns,
            fixed_rho=model.fixed_rho,
        )

        if photo_id not in gallery_v_cls:
            gallery_v_cls[photo_id] = out["v_cls"].detach().cpu()
            photo_id_order.append(photo_id)

        key = dialogue_key(ex, i)
        q_dlg = dialogue_query(out["q_seq"], mode=query_mode).detach().cpu()
        queries.append(q_dlg)
        targets.append(photo_id_order.index(photo_id))
        records[key] = {
            "photo_id": photo_id,
            "dialogue_id": ex.get("dialogue_id"),
            "length": int(len(turns)),
            "label_repair_rate": label_repair_rate(dialogue),
            "rho_repair_rate": float(out["rho_seq"].detach().float().mean().cpu().item()),
            "rank": None,
        }
        n_processed += 1

    if not records:
        raise RuntimeError(f"no dialogues evaluated for run {model.name}")

    ret, ranks = retrieval_metrics(
        torch.stack(queries, dim=0),
        torch.stack([gallery_v_cls[p] for p in photo_id_order], dim=0),
        torch.tensor(targets, dtype=torch.long),
        ks=(1, 5, 10),
        return_ranks=True,
    )

    for key, rank in zip(records.keys(), ranks):
        records[key]["rank"] = float(rank)

    return records, ret, len(photo_id_order)


def build_strata(
    run_records: Dict[str, Dict[str, dict]],
    repair_source: str,
    repair_bucket_run: Optional[str],
):
    common_keys = set.intersection(*(set(v.keys()) for v in run_records.values()))
    if not common_keys:
        raise RuntimeError("no common evaluated dialogues across runs")

    first_run = next(iter(run_records))
    length_values = {
        key: float(run_records[first_run][key]["length"])
        for key in common_keys
    }
    length_threshold, length_groups = median_split(length_values)

    label_rates = {
        key: run_records[first_run][key]["label_repair_rate"]
        for key in common_keys
    }
    has_label_rates = all(v is not None for v in label_rates.values())

    if repair_source == "auto":
        repair_source = "labels" if has_label_rates else "rho"
    if repair_source == "labels" and not has_label_rates:
        raise SystemExit(
            "repair_source=labels requested, but the evaluated test dialogues have no labels"
        )
    if repair_source == "rho":
        source_run = repair_bucket_run or first_run
        if source_run not in run_records:
            raise SystemExit(f"--repair_bucket_run not found among runs: {source_run}")
        repair_values = {
            key: float(run_records[source_run][key]["rho_repair_rate"])
            for key in common_keys
        }
    else:
        source_run = None
        repair_values = {key: float(label_rates[key]) for key in common_keys}

    repair_threshold, repair_groups = median_split(repair_values)
    strata = {}
    for key in common_keys:
        l = length_groups[key]
        r = repair_groups[key]
        strata[key] = {
            "length_bucket": "long" if l == "top50" else "short",
            "repair_bucket": "high_repair" if r == "top50" else "low_repair",
            "quadrant": f"{'long' if l == 'top50' else 'short'}+"
                        f"{'high_repair' if r == 'top50' else 'low_repair'}",
            "repair_rate": repair_values[key],
        }

    return {
        "common_keys": sorted(common_keys),
        "repair_source": repair_source,
        "repair_bucket_run": source_run,
        "length_threshold": length_threshold,
        "repair_threshold": repair_threshold,
        "strata": strata,
    }


def summarize(run_records, strata_info, gallery_sizes):
    common_keys = strata_info["common_keys"]
    strata = strata_info["strata"]
    summaries = {}

    bucket_specs = [
        ("length", "short"),
        ("length", "long"),
        ("repair_rate", "low_repair"),
        ("repair_rate", "high_repair"),
        ("quadrant", "short+low_repair"),
        ("quadrant", "short+high_repair"),
        ("quadrant", "long+low_repair"),
        ("quadrant", "long+high_repair"),
    ]

    for run_name, records in run_records.items():
        run_out = {}
        for bucket_type, bucket_name in bucket_specs:
            if bucket_type == "length":
                keys = [k for k in common_keys if strata[k]["length_bucket"] == bucket_name]
            elif bucket_type == "repair_rate":
                keys = [k for k in common_keys if strata[k]["repair_bucket"] == bucket_name]
            else:
                keys = [k for k in common_keys if strata[k]["quadrant"] == bucket_name]
            run_out[f"{bucket_type}:{bucket_name}"] = bucket_metrics(
                (records[k]["rank"] for k in keys),
                gallery_size=gallery_sizes[run_name],
            )
        summaries[run_name] = run_out
    return summaries


def write_outputs(out_dir, args, run_records, global_metrics, gallery_sizes, strata_info, summaries):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "args": vars(args),
        "global": global_metrics,
        "gallery_sizes": gallery_sizes,
        "strata": {
            "repair_source": strata_info["repair_source"],
            "repair_bucket_run": strata_info["repair_bucket_run"],
            "length_threshold": strata_info["length_threshold"],
            "repair_threshold": strata_info["repair_threshold"],
            "n_common_dialogues": len(strata_info["common_keys"]),
        },
        "buckets": summaries,
    }
    with open(os.path.join(out_dir, "stratified_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(out_dir, "stratified_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "bucket_type", "bucket", "N_queries", *METRIC_KEYS])
        for run_name, buckets in summaries.items():
            for bucket_id, metrics in buckets.items():
                bucket_type, bucket = bucket_id.split(":", 1)
                w.writerow([
                    run_name,
                    bucket_type,
                    bucket,
                    metrics["N_queries"],
                    *[metrics[k] for k in METRIC_KEYS],
                ])

    with open(os.path.join(out_dir, "dialogue_strata.csv"), "w", newline="", encoding="utf-8") as f:
        run_names = list(run_records.keys())
        w = csv.writer(f)
        w.writerow([
            "dialogue_key",
            "photo_id",
            "length",
            "length_bucket",
            "repair_rate",
            "repair_bucket",
            "quadrant",
            *[f"{name}_rank" for name in run_names],
            *[f"{name}_rho_rate" for name in run_names],
        ])
        first_run = run_names[0]
        for key in strata_info["common_keys"]:
            base = run_records[first_run][key]
            s = strata_info["strata"][key]
            w.writerow([
                key,
                base["photo_id"],
                base["length"],
                s["length_bucket"],
                s["repair_rate"],
                s["repair_bucket"],
                s["quadrant"],
                *[run_records[name][key]["rank"] for name in run_names],
                *[run_records[name][key]["rho_repair_rate"] for name in run_names],
            ])


def print_summary(global_metrics, strata_info, summaries):
    print("\n========== STRATIFIED SUMMARY ==========")
    print(f"common dialogues: {len(strata_info['common_keys'])}")
    print(
        "length split: "
        f"short <= median({strata_info['length_threshold']:.3f}), "
        "long = top half"
    )
    repair_msg = (
        f"repair split source: {strata_info['repair_source']}"
        f" ({strata_info['repair_bucket_run']})"
        if strata_info["repair_bucket_run"]
        else f"repair split source: {strata_info['repair_source']}"
    )
    print(f"{repair_msg}; threshold={strata_info['repair_threshold']:.4f}")
    print()

    interesting = [
        "length:short",
        "length:long",
        "repair_rate:low_repair",
        "repair_rate:high_repair",
        "quadrant:short+low_repair",
        "quadrant:short+high_repair",
        "quadrant:long+low_repair",
        "quadrant:long+high_repair",
    ]
    for run_name, buckets in summaries.items():
        g = global_metrics[run_name]
        print(f"[{run_name}] global R@1={g['R@1']:.4f} R@5={g['R@5']:.4f} "
              f"R@10={g['R@10']:.4f} MRR={g['MRR']:.4f}")
        for bucket_id in interesting:
            m = buckets[bucket_id]
            if m["N_queries"] == 0:
                continue
            _, bucket = bucket_id.split(":", 1)
            print(f"  {bucket:18s} N={m['N_queries']:4d} "
                  f"R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
                  f"R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dir", default=None,
                    help="outputs/seed_X directory; uses the standard v6 checkpoint names")
    ap.add_argument("--run", action="append", default=[],
                    help="explicit run in the form name=/path/to/model.pt; can repeat")
    ap.add_argument("--split_manifest", default="outputs/split_v1.json")
    ap.add_argument("--image_map_jsonl", default="data/photochat/train_image_photo_desc.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--query_mode", default="mean", choices=["mean", "last", "max"])
    ap.add_argument("--max_dialogues", type=int, default=None)
    ap.add_argument("--repair_source", choices=["auto", "labels", "rho"], default="auto",
                    help="auto uses labels if present, otherwise mean rho from --repair_bucket_run")
    ap.add_argument("--repair_bucket_run", default=None,
                    help="run name whose mean rho defines low/high repair buckets when labels are absent")
    args = ap.parse_args()

    run_specs = parse_runs(args)
    dialogues = load_split_dialogues(args.split_manifest)
    image_map = load_image_map(args.image_map_jsonl)

    run_records = {}
    global_metrics = {}
    gallery_sizes = {}
    for name, ckpt in run_specs:
        print(f"[eval-stratified] {name}: {ckpt}")
        model = load_model(name, ckpt)
        records, ret, n_gallery = evaluate_run(
            model=model,
            dialogues=dialogues,
            image_map=image_map,
            query_mode=args.query_mode,
            max_dialogues=args.max_dialogues,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_records[name] = records
        global_metrics[name] = ret
        gallery_sizes[name] = n_gallery

    strata_info = build_strata(
        run_records=run_records,
        repair_source=args.repair_source,
        repair_bucket_run=args.repair_bucket_run,
    )
    summaries = summarize(run_records, strata_info, gallery_sizes)
    write_outputs(
        args.out_dir,
        args,
        run_records,
        global_metrics,
        gallery_sizes,
        strata_info,
        summaries,
    )
    print_summary(global_metrics, strata_info, summaries)
    print(f"saved: {args.out_dir}/stratified_metrics.json")
    print(f"saved: {args.out_dir}/stratified_metrics.csv")
    print(f"saved: {args.out_dir}/dialogue_strata.csv")


if __name__ == "__main__":
    main()
