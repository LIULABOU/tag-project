import os, argparse, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
from src.dataloaders.photochat import PhotoChatDataset, collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProjectionHead(nn.Module):
    """Trainable head to align text+image into same space for contrastive loss."""
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, dim=-1)

def clip_contrastive_loss(t_emb, v_emb, temperature=0.07):
    """
    t_emb: (B, D) normalized
    v_emb: (B, D) normalized
    """
    logits = (t_emb @ v_emb.T) / temperature  # (B,B)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_t2i + loss_i2t)

def save_loss_plot(losses, out_path):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/train_full_clip")
    ap.add_argument("--max_items", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--turn_strategy", type=str, default="random", choices=["random", "last"])
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = PhotoChatDataset(
        shards_dir="data/photochat/train_json",
        image_map_jsonl="data/photochat/train_image_photo_desc.jsonl",
        max_items=args.max_items,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    clip = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip.eval()  # keep CLIP frozen for now

    img_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    # CLIP text pooled dim is 512, vision CLS dim is 768 in your earlier code.
    # We'll project both into a shared 256-d space with trainable heads.
    text_proj = ProjectionHead(in_dim=512, out_dim=256).to(DEVICE)
    img_proj  = ProjectionHead(in_dim=768, out_dim=256).to(DEVICE)

    params = list(text_proj.parameters()) + list(img_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    epoch_losses = []

    for ep in range(args.epochs):
        text_proj.train()
        img_proj.train()

        running = 0.0
        steps = 0

        for batch in dl:
            images, turns, metas = batch  # depends on your collate_fn; images may be None
            # metas contains image_path per item
            B = len(metas)

            # ---- choose one turn per dialogue ----
            chosen_turns = []
            for i in range(B):
                tlist = turns[i]
                if args.turn_strategy == "last":
                    chosen_turns.append(tlist[-1] if len(tlist) else "")
                else:
                    chosen_turns.append(random.choice(tlist) if len(tlist) else "")

            # ---- load images from paths ----
            pil_imgs = [Image.open(m["image_path"]).convert("RGB") for m in metas]
            pixel_values = img_processor(images=pil_imgs, return_tensors="pt")["pixel_values"].to(DEVICE)

            # ---- tokenize text ----
            text_inputs = tokenizer(
                chosen_turns,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=77,
            ).to(DEVICE)

            with torch.no_grad():
                # vision
                vision_out = clip.vision_model(pixel_values=pixel_values)
                last_hidden = vision_out.last_hidden_state  # (B, 50, 768)
                v_cls = last_hidden[:, 0, :]                # (B, 768)

                # text
                text_out = clip.text_model(**text_inputs)
                h = text_out.pooler_output                  # (B, 512)

            # ---- projections (trainable) ----
            t_emb = text_proj(h)       # (B, 256) normalized
            v_emb = img_proj(v_cls)    # (B, 256) normalized

            loss = clip_contrastive_loss(t_emb, v_emb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

        avg = running / max(1, steps)
        epoch_losses.append(avg)
        print(f"[Epoch {ep+1}/{args.epochs}] loss={avg:.4f}")

        # save CSV each epoch
        csv_path = os.path.join(args.out_dir, "loss.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss"])
            for i, L in enumerate(epoch_losses, start=1):
                w.writerow([i, L])

        # save plot
        save_loss_plot(epoch_losses, os.path.join(args.out_dir, "loss.png"))

    print("\nDONE")
    print(f"- outputs: {args.out_dir}")
    print(f"- loss csv: {os.path.join(args.out_dir, 'loss.csv')}")
    print(f"- loss plot: {os.path.join(args.out_dir, 'loss.png')}")

if __name__ == "__main__":
    main()
