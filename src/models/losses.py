import torch
import torch.nn.functional as F


def clip_contrastive_loss(img_emb, txt_emb, temperature: float = 0.07):
    """
    Symmetric CLIP-style contrastive loss (image->text + text->image).
    img_emb: (B, D)
    txt_emb: (B, D)
    """
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    logits = img_emb @ txt_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0
