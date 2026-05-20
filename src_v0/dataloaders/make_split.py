"""
Create a train/test split manifest from all train_*.json shards.

Defaults to taking ALL train_NN.json files in --shards_dir, randomly
shuffling, then taking --n_test for test, the rest for train.

Output: a small JSON manifest recording the split (so it's reproducible
and easy to share). Both src.train.train_mvp_wa_wh and
src.eval.eval_tom_model accept --split_manifest pointing at this file.

Usage:
  python -m src.dataloaders.make_split \
      --shards_dir data/photochat/train_json \
      --n_test 2 \
      --seed 42 \
      --out data/photochat/split_v1.json
"""

import argparse
import glob
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", default="data/photochat/train_json",
                    help="directory containing train_*.json files")
    ap.add_argument("--pattern", default="train_*.json",
                    help="glob pattern for shard files")
    ap.add_argument("--n_test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/photochat/split_v1.json")
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shards_dir, args.pattern)))
    if len(shard_paths) == 0:
        raise SystemExit(f"no shards matched: {args.shards_dir}/{args.pattern}")
    if args.n_test >= len(shard_paths):
        raise SystemExit(
            f"--n_test={args.n_test} but only {len(shard_paths)} shards available"
        )

    shard_basenames = [os.path.basename(p) for p in shard_paths]

    rng = random.Random(args.seed)
    shuffled = shard_basenames[:]
    rng.shuffle(shuffled)
    test = sorted(shuffled[: args.n_test])
    train = sorted(shuffled[args.n_test:])

    manifest = {
        "shards_dir": args.shards_dir,
        "pattern": args.pattern,
        "seed": args.seed,
        "n_total_shards": len(shard_basenames),
        "n_train": len(train),
        "n_test": len(test),
        "train_shards": train,
        "test_shards": test,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved: {args.out}")
    print(f"  total shards: {len(shard_basenames)}")
    print(f"  train ({len(train)}): {train}")
    print(f"  test  ({len(test)}): {test}")


if __name__ == "__main__":
    main()
