import os
import json
import glob

TRAIN_JSON_DIR = "data/photochat/train_json"
IMAGE_DIR = "data/photochat/images"
OUT_FILE = "data/photochat/train_image_photo_desc.jsonl"

def photo_id_to_path(photo_id: str) -> str:
    split, pid = photo_id.split("/")
    return os.path.join(IMAGE_DIR, f"{split}_{pid}.jpg")

def iter_dialogues_from_file(fp: str):
    with open(fp, "r") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ["data", "dialogues", "dialogs", "items", "examples"]:
            if key in obj and isinstance(obj[key], list):
                yield from obj[key]
                return
        yield obj

def main():
    files = sorted(glob.glob(os.path.join(TRAIN_JSON_DIR, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found in {TRAIN_JSON_DIR}")

    total = kept = skipped_missing = skipped_no_desc = 0
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(OUT_FILE, "w") as out:
        for fp in files:
            for d in iter_dialogues_from_file(fp):
                total += 1
                photo_id = d.get("photo_id")
                photo_description = d.get("photo_description")  # ✅ correct key

                if not photo_id or not photo_description:
                    skipped_no_desc += 1
                    continue

                image_path = photo_id_to_path(photo_id)
                if not os.path.exists(image_path):
                    skipped_missing += 1
                    continue

                out.write(json.dumps({
                    "photo_id": photo_id,
                    "image_path": image_path,
                    "photo_description": photo_description
                }) + "\n")
                kept += 1

    print("Done ✅")
    print(f"JSON files: {len(files)}")
    print(f"Total dialogues scanned: {total}")
    print(f"Kept (has image + photo_description): {kept}")
    print(f"Skipped (missing photo_description/photo_id): {skipped_no_desc}")
    print(f"Skipped (image not found on disk): {skipped_missing}")
    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()
