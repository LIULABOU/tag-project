import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Legacy (kept for backward compat with old training script / eval)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# New: ToM-aligned losses matching the report
# ----------------------------------------------------------------------
def kl_div_distributions(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-row KL(p || q) for two probability distributions sharing the last dim.
    p, q : [..., N]   (each row sums to 1)
    returns: [...]    (one scalar per row)
    """
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def stab_switch_loss(
    e_seq: torch.Tensor,
    rho_seq: torch.Tensor,
    delta: float = 0.1,
    detach_rho_weight: bool = True,
):
    """
    Implements the report's:

        L_stab   = sum_{t>=2} (1 - rho_t) * KL(e_{t-1} || e_t)
        L_switch = sum_{t>=2}     rho_t   * max(0, delta - KL(e_{t-1} || e_t))

    IMPORTANT: by default rho_t is DETACHED when used as the (1-rho)/rho weight.
    This treats rho as a regulator/mask whose role is to "select which loss is
    active on this turn" -- it should NOT be trained by these losses themselves
    (rho is trained by L_repair for supervised_A, or by L_rate for DCP).

    Without detach, gradient would flow:
        L_stab -> (1-rho) -> rho -> d_t -> e_hat_t -> Wa_text
    creating a degenerate pathway where Wa_text is pushed to bend e_hat_t to
    minimize the stab loss by making rho large (instead of by making evidence
    actually stable). This was empirically hurting retrieval in DCP mode.

    Gradient still flows through e_t = (1-rho)*e_{t-1} + rho*e_hat_t (the
    rho values inside e_t are the actual non-detached values, so Wa_text
    learns evidence dynamics normally).

    Args:
      e_seq:             [T, N] evidence distributions per turn
      rho_seq:           [T]    repair gate per turn
      delta:             switching margin
      detach_rho_weight: if True (default), detach rho when used as a weight
                         in the loss expression; e_seq itself is unchanged.
    """
    T = e_seq.shape[0]
    if T < 2:
        zero = torch.tensor(0.0, device=e_seq.device)
        return zero, zero

    e_prev = e_seq[:-1]      # [T-1, N]
    e_curr = e_seq[1:]       # [T-1, N]
    rho_curr = rho_seq[1:]   # [T-1]
    rho_w = rho_curr.detach() if detach_rho_weight else rho_curr

    kl = kl_div_distributions(e_prev, e_curr)  # [T-1]

    stab = ((1.0 - rho_w) * kl).mean()
    switch = (rho_w * torch.clamp(delta - kl, min=0.0)).mean()
    return stab, switch


def masked_corr_loss(
    q_pool: torch.Tensor,
    v_cls_pool: torch.Tensor,
    dialogue_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Implements the report's L_corr (Eq. 13):

        L_corr = sum_t m_t * [ -log( exp(sim(q_t, v^cls_{d(t)}) / tau)
                                     / sum_{j=1..B} exp(sim(q_t, v^cls,j) / tau) ) ]

    where v^cls,j ranges over images of all dialogues in the batch.

    Args:
      q_pool:       [N_total, D]  text queries q_t for all turns in the batch
      v_cls_pool:   [B, D]        global image embedding per dialogue
      dialogue_ids: [N_total] long tensor mapping each turn to its dialogue idx
      mask:         [N_total] groundedness mask m_t (None -> all ones)
      temperature:  CLIP-style temperature

    NOTE: q_pool and v_cls_pool must share the same dim D (since the report
    uses q_t = W_h h_t with W_h projecting into the visual space).
    """
    q_n = F.normalize(q_pool, dim=-1)
    v_n = F.normalize(v_cls_pool, dim=-1)
    logits = q_n @ v_n.t() / temperature  # [N_total, B]
    loss_per_turn = F.cross_entropy(logits, dialogue_ids, reduction="none")
    if mask is None:
        return loss_per_turn.mean()
    denom = mask.sum().clamp_min(1e-12)
    return (loss_per_turn * mask).sum() / denom


def coherence_floor_loss(a_seq: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    """
    Penalize coherence trajectory dropping below floor tau.

        L_coh_floor = (1/T) sum_t max(0, tau - a_t)^2

    Asymmetric: a_t > tau pays nothing, a_t < tau is penalized quadratically.
    Encourages alignment head not to collapse a_tilde to ~0 on ambiguous turns.

    Args:
      a_seq: [T] coherence trajectory
      tau:   floor threshold (e.g. 0.15 or 0.20)
    """
    return torch.clamp(tau - a_seq, min=0.0).pow(2).mean()


def coherence_drop_loss(a_seq: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """
    Penalize sudden coherence drops larger than epsilon.

        L_coh_drop = (1/(T-1)) sum_t max(0, a_{t-1} - a_t - epsilon)^2

    Asymmetric (only drops, not rises) and tolerates small natural variations.
    NOTE: this can structurally fight legitimate ToM repair events (where a_t
    is expected to dip during repair); use small gamma and small epsilon.

    Args:
      a_seq:   [T] coherence trajectory
      epsilon: tolerance for normal drops (e.g. 0.05)
    """
    if a_seq.shape[0] < 2:
        return torch.tensor(0.0, device=a_seq.device)
    diff = a_seq[:-1] - a_seq[1:]  # positive when a is dropping
    return torch.clamp(diff - epsilon, min=0.0).pow(2).mean()


def rate_loss(rho_pool: torch.Tensor, target_rate: float) -> torch.Tensor:
    """
    Sparsity prior from report Eq. (24):
        L_rate = (mean(rho_t) - r)^2

    Args:
      rho_pool:    flat tensor of rho values across the batch
      target_rate: desired repair frequency r in [0,1]
    """
    return (rho_pool.mean() - target_rate) ** 2


def weighted_repair_mse(
    rho_pred: torch.Tensor,
    repair_targets: list,
    repair_weights: list,
):
    """
    Weighted MSE only on labeled turns. Targets are soft (e.g., 1.0/0.4/0.0).

    Args:
      rho_pred:        [T]
      repair_targets:  list of length T with floats or None
      repair_weights:  list of length T with floats or None

    Returns:
      scalar loss tensor, or None if no labels.
    """
    labeled = [
        i for i, (y, w) in enumerate(zip(repair_targets, repair_weights))
        if (y is not None and w is not None)
    ]
    if not labeled:
        return None

    device = rho_pred.device
    target = torch.tensor(
        [repair_targets[i] for i in labeled], dtype=torch.float32, device=device
    )
    weights = torch.tensor(
        [repair_weights[i] for i in labeled], dtype=torch.float32, device=device
    )
    pred = rho_pred[labeled]

    per = (pred - target) ** 2
    return (per * weights).sum() / weights.sum().clamp_min(1e-12)
