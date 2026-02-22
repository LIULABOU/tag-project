import os, argparse, random, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from src.dataloaders.photochat import PhotoChatDataset, collate_fn
from src.models.clip_model import CLIPBackbone
from src.models.alignment import AlignmentHead
from src.models.state_tracker import FixedGateEvidenceGRU
from src.models.mvp_aligner import MVPAligner
from src.models.losses import clip_contrastive_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs/mvp_wa_wh")
    ap.add_argument("--max_items", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turn_strategy", type=str, default="last", choices=["last","random"])
    ap.add_argument("--proj_dim", type=int, default=256)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Dataset ----
    ds = PhotoChatDataset(
        shards_dir="data/photochat/train_json",
        image_map_jsonl="data/photochat/train_image_photo_desc.jsonl",
        max_items=args.max_items,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # ---- Frozen CLIP ----
    clip = CLIPBackbone(device=DEVICE)
    processor = clip.processor
    img_processor = processor.image_processor
    tokenizer = processor.tokenizer

    # ---- AlignmentHead (frozen) + Tracker (frozen) ----
    align = AlignmentHead(h_dim=clip.text_dim, v_dim=clip.vision_dim, q_dim=clip.vision_dim).to(DEVICE)
    align.eval()
    for p in align.parameters():
        p.requires_grad = False

    tracker = FixedGateEvidenceGRU(num_patches=49, rho=args.rho).to(DEVICE)  # num_patches will be checked per example
    tracker.eval()

    # ---- Trainable Wa/Wh only ----
    mvp = MVPAligner(text_dim=clip.text_dim, vision_dim=clip.vision_dim, proj_dim=args.proj_dim).to(DEVICE)
    opt = torch.optim.AdamW(mvp.parameters(), lr=args.lr)

    losses = []

    for ep in range(args.epochs):
        mvp.train()
        running = 0.0
        steps = 0

        for images, turns, metas in dl:
            B = len(metas)

            # ---- load images ----
            pil_imgs = [Image.open(m["image_path"]).convert("RGB") for m in metas]
            pixel_values = img_processor(images=pil_imgs, return_tensors="pt")["pixel_values"].to(DEVICE)

            # ---- choose one text turn per dialogue ----
            chosen_turns = []
            for i in range(B):
                tlist = turns[i]
                if not tlist:
                    chosen_turns.append("")
                elif args.turn_strategy == "random":
                    chosen_turns.append(random.choice(tlist))
                else:
                    chosen_turns.append(tlist[-1])

            text_inputs = tokenizer(
                chosen_turns, padding=True, truncation=True, return_tensors="pt", max_length=77
            ).to(DEVICE)

            # ---- frozen features ----
            with torch.no_grad():
                v_cls_b, v_patches_b = clip.encode_vision_tokens(pixel_values)  # [B,768], [B,N,768]
                h_b = clip.encode_text_pooler(text_inputs["input_ids"], text_inputs["attention_mask"])  # [B,512]

            # ---- build attended image vector g using stabilized evidence e_t ----
            g_list = []
            for i in range(B):
                v_cls = v_cls_b[i]
                v_patches = v_patches_b[i]          # [N,768]
                N = v_patches.shape[0]

                # recompute evidence for this one sample using the chosen single turn
                h_one = h_b[i].unsqueeze(0)         # [1,512]
                with torch.no_grad():
                    a_one, e_hat_one = align(h_one, v_cls, v_patches)     # [1], [1,N]
                    e_final_one = tracker(e_hat_one)                      # [1,N]

                # attended image vector g = sum_i e(i)*v_patch(i)
                g_one = torch.matmul(e_final_one, v_patches.unsqueeze(0)).squeeze(0)  # [1,768]
                g_list.append(g_one)

            g_b = torch.cat(g_list, dim=0)  # [B,768]

            # ---- train Wa/Wh ----
            t_emb, v_emb = mvp(h_b, g_b)
            loss = clip_contrastive_loss(v_emb, t_emb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

        avg = running / max(1, steps)
        losses.append(avg)
        print(f"[Epoch {ep+1}/{args.epochs}] loss={avg:.4f}")

        # save log
        with open(os.path.join(args.out_dir, "loss.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss"])
            for i, L in enumerate(losses, start=1):
                w.writerow([i, L])

    print("\nDONE")
    print(f"- outputs: {args.out_dir}")

if __name__ == "__main__":
    main()
