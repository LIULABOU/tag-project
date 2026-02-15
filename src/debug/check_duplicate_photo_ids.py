import json
from collections import Counter

DATA_FILE = "data/photochat/train_image_photo_desc.jsonl"

def main():
    c = Counter()

    with open(DATA_FILE, "r") as f:
        for line in f:
            rec = json.loads(line)
            c[rec["photo_id"]] += 1

    total_entries = sum(c.values())
    unique_photo_ids = len(c)
    multi_dialogue_photo_ids = sum(1 for v in c.values() if v > 1)
    repeated_entries = sum(v for v in c.values() if v > 1)
    extra_duplicates = total_entries - unique_photo_ids
    max_repeat = max(c.values())

    print("=== Duplicate photo_id analysis ===")
    print(f"Total JSONL entries (dialogue-level pairs): {total_entries}")
    print(f"Unique photo_id (unique images referenced): {unique_photo_ids}")
    print(f"photo_id referenced by multiple dialogues : {multi_dialogue_photo_ids}")
    print(f"Total entries that are repeats (v>1)      : {repeated_entries}")
    print(f"Extra duplicate rows (total - unique)     : {extra_duplicates}")
    print(f"Max repeats for a single photo_id         : {max_repeat}")

    print("\nTop 10 most repeated photo_id:")
    shown = 0
    for pid, cnt in c.most_common():
        if cnt <= 1:
            break
        print(f"  {pid}: {cnt}")
        shown += 1
        if shown >= 10:
            break

if __name__ == "__main__":
    main()
