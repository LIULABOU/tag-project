import json
import os
from typing import Dict, List, Optional

from torch.utils.data import Dataset


def label_to_soft_target(label):
    """
    Human/predicted label -> soft supervision target y_t

    repair -> 1.0
    clarification -> 0.4
    stable -> 0.0

    If unlabeled, return None.
    """
    if label == 0:
        return 1.0
    elif label == 1:
        return 0.4
    elif label == 2:
        return 0.0
    return None


class PhotoChatDataset(Dataset):
    """
    Semi-supervised dialogue dataset.

    Sources:
      - human-labeled:
          data/photochat/train_json/train_00.json   (first 100 labeled)
          data/photochat/train_json/train_01.json   (first 100 labeled)
      - predicted-labeled:
          data/photochat/predicted_train03.json
    """

    def __init__(
        self,
        shards_dir: str = "data/photochat/train_json",
        predicted_json: str = "data/photochat/predicted_train03.json",
        image_map_jsonl: str = "data/photochat/train_image_photo_desc.jsonl",
        split_filter: Optional[str] = "train",
        use_human: bool = True,
        use_predicted: bool = True,
        human_limit_per_file: int = 100,
        max_items: Optional[int] = None,
        # NEW: full-shard mode -- loads ALL dialogues from the given shard files,
        # ignoring the legacy first-100-of-train_00/01 + predicted_train03 logic.
        # Used by --split_manifest workflows.
        shard_files: Optional[List[str]] = None,
        per_shard_limit: Optional[int] = None,
    ):
        self.photo_map: Dict[str, Dict[str, str]] = {}
        with open(image_map_jsonl, "r") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["photo_id"]
                self.photo_map[pid] = {
                    "image_path": rec["image_path"],
                    "photo_description": rec["photo_description"],
                }

        self.items: List[Dict] = []

        # ----------------------------------------------------------
        # NEW PATH: explicit shard list (full data, no label filter)
        # ----------------------------------------------------------
        if shard_files is not None:
            for sp in shard_files:
                if not os.path.exists(sp):
                    print(f"[PhotoChatDataset] shard not found, skipping: {sp}")
                    continue
                with open(sp, "r") as f:
                    data = json.load(f)
                limit = per_shard_limit if per_shard_limit else len(data)
                for ex in data[:limit]:
                    # mark label_source as "human" if we know it has labels,
                    # otherwise "predicted" (gives weight 0.5 in supervised_A);
                    # for DCP/multi_C this is irrelevant.
                    src = "human" if any("label" in t for t in ex.get("dialogue", [])) else "predicted"
                    self._add_example(ex=ex, label_source=src, split_filter=split_filter)
                    if max_items is not None and len(self.items) >= max_items:
                        return
            return

        # ----------------------------------------------------------
        # LEGACY PATH: first-100 of train_00/01 (+ optional predicted)
        # ----------------------------------------------------------
        if use_human:
            human_files = [
                os.path.join(shards_dir, "train_00.json"),
                os.path.join(shards_dir, "train_01.json"),
            ]

            for hf in human_files:
                if not os.path.exists(hf):
                    continue

                with open(hf, "r") as f:
                    data = json.load(f)

                # only first 100 conversations are labeled, per your setup
                for ex in data[:human_limit_per_file]:
                    self._add_example(
                        ex=ex,
                        label_source="human",
                        split_filter=split_filter,
                    )

                    if max_items is not None and len(self.items) >= max_items:
                        return

        if use_predicted and os.path.exists(predicted_json):
            with open(predicted_json, "r") as f:
                data = json.load(f)

            for ex in data:
                self._add_example(
                    ex=ex,
                    label_source="predicted",
                    split_filter=split_filter,
                )

                if max_items is not None and len(self.items) >= max_items:
                    return

    def _add_example(self, ex: Dict, label_source: str, split_filter: Optional[str]):
        photo_id = ex.get("photo_id")
        dialogue = ex.get("dialogue")
        dialogue_id = ex.get("dialogue_id")

        if not photo_id or not dialogue:
            return

        if split_filter is not None and not photo_id.startswith(split_filter + "/"):
            return

        if photo_id not in self.photo_map:
            return

        image_path = self.photo_map[photo_id]["image_path"]
        if not os.path.exists(image_path):
            return

        turns = [t.get("message", "") for t in dialogue]
        turn_labels = [t.get("label", None) for t in dialogue]
        repair_targets = [label_to_soft_target(lbl) for lbl in turn_labels]

        # supervision weight per labeled turn
        # human labels trusted more than predicted labels
        repair_weights = []
        for y in repair_targets:
            if y is None:
                repair_weights.append(None)
            else:
                if label_source == "human":
                    repair_weights.append(1.0)
                else:
                    repair_weights.append(0.5)

        self.items.append({
            "photo_id": photo_id,
            "dialogue_id": dialogue_id,
            "turns": turns,
            "turn_labels": turn_labels,
            "repair_targets": repair_targets,
            "repair_weights": repair_weights,
            "label_source": label_source,
            "image_path": image_path,
            "photo_description": self.photo_map[photo_id]["photo_description"],
        })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]

        meta = {
            "photo_id": ex["photo_id"],
            "dialogue_id": ex["dialogue_id"],
            "image_path": ex["image_path"],
            "photo_description": ex["photo_description"],
            "turn_labels": ex["turn_labels"],
            "repair_targets": ex["repair_targets"],
            "repair_weights": ex["repair_weights"],
            "label_source": ex["label_source"],
        }

        return ex["image_path"], ex["turns"], meta


def collate_fn(batch):
    image_paths, turns, metas = zip(*batch)
    return list(image_paths), list(turns), list(metas)