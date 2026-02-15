import json
import os
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset


class PhotoChatDataset(Dataset):
    """
    Dialogue-level dataset.

    Returns:
      image_path: str
      turns: List[str]
      meta: dict {photo_id, dialogue_id, photo_description, image_path}
    """

    def __init__(
        self,
        shards_dir: str = "data/photochat/train_json",
        image_map_jsonl: str = "data/photochat/train_image_photo_desc.jsonl",
        split_filter: Optional[str] = "train",   # "train", "test", or None
        max_items: Optional[int] = None,
    ):
        # 1) photo_id -> image_path, photo_description
        self.photo_map: Dict[str, Dict[str, str]] = {}
        with open(image_map_jsonl, "r") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["photo_id"]
                self.photo_map[pid] = {
                    "image_path": rec["image_path"],
                    "photo_description": rec["photo_description"],
                }

        # 2) scan shard dialogues and keep only ones with valid photo_id + image exists
        self.items: List[Dict] = []
        shard_files = sorted(
            os.path.join(shards_dir, x) for x in os.listdir(shards_dir) if x.endswith(".json")
        )

        for sf in shard_files:
            with open(sf, "r") as f:
                data = json.load(f)

            for ex in data:
                photo_id = ex.get("photo_id")
                dialogue = ex.get("dialogue")
                dialogue_id = ex.get("dialogue_id")

                if not photo_id or not dialogue:
                    continue

                if split_filter is not None and not photo_id.startswith(split_filter + "/"):
                    continue

                if photo_id not in self.photo_map:
                    continue

                image_path = self.photo_map[photo_id]["image_path"]
                if not os.path.exists(image_path):
                    continue

                turns = [t.get("message", "") for t in dialogue]  # includes "" for share_photo turn

                self.items.append({
                    "photo_id": photo_id,
                    "dialogue_id": dialogue_id,
                    "turns": turns,
                    "image_path": image_path,
                    "photo_description": self.photo_map[photo_id]["photo_description"],
                })

                if max_items is not None and len(self.items) >= max_items:
                    break

            if max_items is not None and len(self.items) >= max_items:
                break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]

        meta = {
            "photo_id": ex["photo_id"],
            "dialogue_id": ex["dialogue_id"],
            "image_path": ex["image_path"],
            "photo_description": ex["photo_description"],
        }

        # IMPORTANT: return image_path (not PIL / tensor). Trainer will call CLIPProcessor.
        return ex["image_path"], ex["turns"], meta


def collate_fn(batch):
    """
    Works for batch_size=1 or more.
    We keep turns as lists (variable length), and image_paths as strings.
    """
    image_paths, turns, metas = zip(*batch)
    return list(image_paths), list(turns), list(metas)
