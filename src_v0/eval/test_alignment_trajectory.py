import json
import os
import datetime
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.clip_model import CLIPBackbone
from src.models.projection import ProjectionHead
from src.models.losses import clip_contrastive_loss

DATA_FILE = "data/photochat/train_image_photo_desc.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 2
MAX_TEXT_LEN = 77  # CLIP standard


class PhotoDescDataset(Dataset):
    """
    For CLIP contrastive training:
      returns (image_path, text)
    """
    def __init__(self, jsonl_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if "image_path" in ex and "photo_description" in ex:
                    self.samples.append((ex["image_path"], ex["photo_description"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Loads PIL images here so CLIPProcessor can batch them.
    """
    image_paths, texts = zip(*batch)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, list(texts)


def make_run_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = os.path.join("outputs", "runs", f"{ts}_train_full_clip")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir: str, dataset_size: int):
    cfg_path = os.path.join(run_dir, "config.txt")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"DATA_FILE={DATA_FILE}\n")
        f.write(f"DATASET_SIZE={dataset_size}\n")
        f.write(f"DEVICE={DEVICE}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"LR={LR}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"MAX_TEXT_LEN={MAX_TEXT_LEN}\n")


def main():
    run_dir = make_run_dir()

    dataset = PhotoDescDataset(DATA_FILE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print("Loaded samples:", len(dataset))
    print("Run dir:", run_dir)
    save_run_config(run_dir, len(dataset))

    # ----- Frozen CLIP backbone -----
    clip = CLIPBackbone(device=DEVICE)
    processor = clip.processor

    # ----- Trainable heads -----
    dim = clip.projection_dim
    img_head = ProjectionHead(dim).to(DEVICE)
    txt_head = ProjectionHead(dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(img_head.parameters()) + list(txt_head.parameters()),
        lr=LR
    )

    losses = []
    step = 0

    for epoch in range(EPOCHS):
        img_head.train()
        txt_head.train()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for images, texts in tqdm(loader):
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LEN,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            img_feat = clip.encode_image(inputs["pixel_values"])
            txt_feat = clip.encode_text(inputs["input_ids"], inputs["attention_mask"])

            img_emb = img_head(img_feat)
            txt_emb = txt_head(txt_feat)

            loss = clip_contrastive_loss(img_emb, txt_emb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    # ----- Save loss curve -----
    loss_curve_path = os.path.join(run_dir, "loss_curve.png")
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (CLIP contrastive baseline)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # ----- Save raw losses -----
    loss_txt_path = os.path.join(run_dir, "loss_per_step.txt")
    with open(loss_txt_path, "w", encoding="utf-8") as f:
        for i, l in enumerate(losses):
            f.write(f"{i}\t{l}\n")

    # ----- Optional: save trained heads (useful later) -----
    torch.save(img_head.state_dict(), os.path.join(run_dir, "img_head.pt"))
    torch.save(txt_head.state_dict(), os.path.join(run_dir, "txt_head.pt"))

    print("\nSaved:", loss_curve_path)
    print("Saved:", loss_txt_path)
    print("Saved: img_head.pt, txt_head.pt")
    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    main()
