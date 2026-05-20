import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader
from src.dataloaders.photochat import PhotoChatDataset, collate_fn


def main():
    # Your dataset paths (current repo structure)
    shards_dir = REPO_ROOT / "data" / "photochat" / "train_json"
    image_map_jsonl = REPO_ROOT / "data" / "photochat" / "train_image_photo_desc.jsonl"

    # Load a small subset for sanity
    ds = PhotoChatDataset(
        shards_dir=str(shards_dir),
        image_map_jsonl=str(image_map_jsonl),
        max_items=50,
    )

    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    images, turns_batch, metas_batch = next(iter(dl))

    turns = turns_batch[0]
    meta = metas_batch[0]
    img = images[0]

    print("Dataset length (subset):", len(ds))
    print("photo_id:", meta.get("photo_id"))
    print("image_path:", meta.get("image_path"))
    print("meta keys:", sorted(list(meta.keys())))
    print("Num turns:", len(turns))
    print("First 8 turns:", turns[:8])
    print("Image type:", type(img))


if __name__ == "__main__":
    main()
