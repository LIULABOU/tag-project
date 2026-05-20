import os, random, argparse, csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
from src.dataloaders.photochat import PhotoChatDataset, collate_fn
from src.models.alignment import AlignmentHead, evidence_dynamics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"


def check_no_nan(x, name):
    x = np.asarray(x)
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError(f"[NaN/Inf detected] {name}")


def save_alignment_plot(a_t, out_path):
    plt.figure()
    plt.plot(list(range(len(a_t))), a_t, marker="o")
    plt.xlabel("Turn")
    plt.ylabel("alignment a_t")
    plt.title("Alignment trajectory over turns")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def overlay_heatmap(image_pil, heatmap_2d, out_path):
    img = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    hm = np.clip(heatmap_2d.astype(np.float32), 0, 1)

    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(
        (img.shape[1], img.shape[0]), resample=Image.NEAREST
    )
    hm_resized = np.array(hm_img).astype(np.float32) / 255.0

    plt.figure()
    plt.imshow(img)
    plt.imshow(hm_resized, alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def attention_stats(attn):
    attn = np.asarray(attn, dtype=np.float32)
    attn = attn / (attn.sum() + 1e-9)
    mx = float(attn.max())
    ent = float(-(attn * np.log(attn + 1e-9)).sum())
    return mx, ent


def infer_patch_grid(num_patches):
    g = int(np.sqrt(num_patches))
    if g * g != num_patches:
        raise ValueError(
            f"num_patches={num_patches} not a perfect square. "
            f"Expected 49->7x7 or 196->14x14 etc."
        )
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/sanity")
    ap.add_argument("--max_items", type=int, default=5000, help="cap dataset length for faster sampling")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Dataset + loader -----
    ds = PhotoChatDataset(
        shards_dir="data/photochat/train_json",
        image_map_jsonl="data/photochat/train_image_photo_desc.jsonl",
        max_items=args.max_items,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # ----- Load CLIP (frozen) -----
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip.eval()
    img_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    align = AlignmentHead(h_dim=512, v_dim=768, q_dim=768).to(DEVICE)
    align.eval()

    summary_rows = []
    num_done = 0

    for batch in dl:
        if num_done >= args.num_examples:
            break

        _, turns, metas = batch
        turns = turns[0]
        meta = metas[0]
        T = len(turns)

        ex_dir = os.path.join(args.out_dir, f"ex_{num_done:02d}_{meta['photo_id'].replace('/', '_')}")
        os.makedirs(ex_dir, exist_ok=True)

        # ----- Load image -----
        pil_img = Image.open(meta["image_path"]).convert("RGB")

        # ----- Vision forward -----
        pixel_values = img_processor(images=pil_img, return_tensors="pt")["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vision_out = clip.vision_model(pixel_values=pixel_values)
            last_hidden = vision_out.last_hidden_state          # [1, 50, 768]
            v_cls = last_hidden[:, 0, :].squeeze(0)             # [768]
            v_patches = last_hidden[:, 1:, :].squeeze(0)        # [49, 768]

        # ----- Text forward for all turns -----
        text_inputs = tokenizer(
            turns,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77,
        ).to(DEVICE)

        with torch.no_grad():
            text_out = clip.text_model(**text_inputs)
            h_all = text_out.pooler_output                      # [T, 512]

        # ----- Compute per-turn alignment + attention using AlignmentHead -----
        with torch.no_grad():
            a_t, e_hat = align(h_all, v_cls, v_patches)          # a_t:[T], e_hat:[T,N]
            e_final = evidence_dynamics(e_hat, rho=0.1)          # e_t dynamics, fixed rho=0.1

        a_arr = a_t.detach().cpu().numpy().astype(np.float32)
        check_no_nan(a_arr, "a_t")
        save_alignment_plot(a_arr, os.path.join(ex_dir, "alignment_trajectory.png"))
        e_hat_np = e_hat.detach().cpu().numpy().astype(np.float32)     # [T, N]
        e_fin_np = e_final.detach().cpu().numpy().astype(np.float32)   # [T, N]

        num_patches = e_hat_np.shape[1]
        grid = infer_patch_grid(num_patches)
        uniform_max = 1.0 / num_patches

        # ----- Heatmaps per turn (RAW + FINAL) -----
        for t in range(T):
            # ---------- RAW (utterance-only) evidence: e_hat ----------
            attn_hat = e_hat_np[t]
            attn_hat = attn_hat / (attn_hat.sum() + 1e-9)

            mx_hat, ent_hat = attention_stats(attn_hat)
            check_no_nan(attn_hat, f"e_hat turn {t}")

            hm_hat = attn_hat.reshape(grid, grid)
            hm_hat = (hm_hat - hm_hat.min()) / (hm_hat.max() - hm_hat.min() + 1e-9)
            overlay_heatmap(pil_img, hm_hat, os.path.join(ex_dir, f"turn_{t:02d}_heatmap_hat.png"))

            # ---------- FINAL evidence after dynamics: e_t ----------
            attn_fin = e_fin_np[t]
            attn_fin = attn_fin / (attn_fin.sum() + 1e-9)

            mx_fin, ent_fin = attention_stats(attn_fin)
            check_no_nan(attn_fin, f"e_final turn {t}")

            hm_fin = attn_fin.reshape(grid, grid)
            hm_fin = (hm_fin - hm_fin.min()) / (hm_fin.max() - hm_fin.min() + 1e-9)
            overlay_heatmap(pil_img, hm_fin, os.path.join(ex_dir, f"turn_{t:02d}_heatmap_final.png"))

            # Log both in summary
            summary_rows.append([
                meta["photo_id"], t, float(a_arr[t]),
                mx_hat, ent_hat,
                mx_fin, ent_fin,
                uniform_max
            ])

        # Save turns for reference
        with open(os.path.join(ex_dir, "turns.txt"), "w", encoding="utf-8") as f:
            for t, ut in enumerate(turns):
                f.write(f"[{t}] {ut}\n")

        # Save meta for reference
        with open(os.path.join(ex_dir, "meta.txt"), "w", encoding="utf-8") as f:
            for k, v in meta.items():
                f.write(f"{k}: {v}\n")

        print(f"[OK] Saved example {num_done+1}/{args.num_examples}: {ex_dir}")
        num_done += 1

    # ----- Summary CSV -----
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "photo_id", "turn", "a_t",
            "hat_max", "hat_entropy",
            "final_max", "final_entropy",
            "uniform_max"
        ])
        w.writerows(summary_rows)

    print("\nDONE")
    print(f"- outputs: {args.out_dir}")
    print(f"- summary: {csv_path}")


if __name__ == "__main__":
    main()
