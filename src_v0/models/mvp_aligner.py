import torch
import torch.nn as nn


class MVPAligner(nn.Module):
    """
    State-aware ToM-style repair gate (binary version).

    rho_t = sigmoid( W_p [h_t ; e_{t-1} ; a_{t-1}] )

    This replaces the older MVPAligner which carried Wa / Wh / coherence_head.
    Per the report:
      - W_h (text->visual projection) lives in AlignmentHead.Wa_text and is shared
        across grounding / coherence head / contrastive loss.
      - The coherence-head ã_t is produced by AlignmentHead.alignment_score (Eq. 10).
      - This module ONLY produces the repair gate rho_t.

    Why the input is [h_t; e_{t-1}; a_{t-1}]:
      - h_t  : current utterance content
      - e_{t-1}: prior grounding state (where the dialogue was looking)
      - a_{t-1}: prior coherence (how well things had been holding together)

    This is the minimal "state tracker" that lets rho_t be ToM-style instead
    of just a per-turn text classifier.
    """

    def __init__(self, text_dim: int, num_patches: int):
        super().__init__()
        self.text_dim = text_dim
        self.num_patches = num_patches
        in_dim = text_dim + num_patches + 1
        self.head = nn.Linear(in_dim, 1)

    def forward(
        self,
        h_t: torch.Tensor,
        e_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_t    : [text_dim]            current text embedding (frozen CLIP pooler)
        e_prev : [num_patches]         previous evidence distribution
        a_prev : scalar tensor         previous coherence

        returns: scalar tensor in (0,1), the repair gate rho_t.
        """
        assert h_t.dim() == 1, f"h_t expected [text_dim], got {tuple(h_t.shape)}"
        assert e_prev.dim() == 1, f"e_prev expected [N], got {tuple(e_prev.shape)}"
        assert e_prev.shape[0] == self.num_patches

        a_view = a_prev.reshape(1)
        x = torch.cat([h_t, e_prev, a_view], dim=0)  # [text_dim + N + 1]
        return torch.sigmoid(self.head(x)).squeeze()


# Backward-compat alias: the new module is conceptually a RepairGate.
RepairGate = MVPAligner


class DCPGate(nn.Module):
    """
    Variant B from report Section 1.3 -- Differentiable Change-Point Gating.

        d_t   = KL(e_{t-1} || ê_t)             (Eq. 22)  -- computed in run_dialogue
        rho_t = sigmoid( a * (d_t - mu) )       (Eq. 23)

    where:
      - a  is a learnable sharpness scalar   (parameterized as exp(log_a) > 0)
      - mu is a running mean of d_t          (non-trainable buffer, EMA-updated)

    Training pairs this gate with a sparsity prior L_rate (Eq. 24):
        L_rate = (mean(rho_t) - r)^2
    so it does NOT need any human repair labels.
    """

    def __init__(
        self,
        init_a: float = 5.0,
        init_mu: float = 0.5,
        ema_momentum: float = 0.1,
    ):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_a)))))
        self.register_buffer("mu", torch.tensor(float(init_mu)))
        self.ema_momentum = float(ema_momentum)

    @property
    def a(self) -> torch.Tensor:
        return self.log_a.exp()

    def forward(self, d_t: torch.Tensor) -> torch.Tensor:
        """
        d_t: scalar tensor or [T] tensor
        returns rho_t with the same shape, in (0,1)
        """
        return torch.sigmoid(self.a * (d_t - self.mu))

    @torch.no_grad()
    def update_mu(self, d_values: torch.Tensor):
        """EMA update from a batch of d_t values (any shape)."""
        if d_values.numel() == 0:
            return
        batch_mean = d_values.detach().mean()
        self.mu.mul_(1.0 - self.ema_momentum).add_(self.ema_momentum * batch_mean)


class MSCPGate(nn.Module):
    """
    Variant C -- Multi-Signal Change-Point Gate (no labels needed).

        rho_t = sigmoid( W_p [d_t ; a_{t-1} ; delta_a_hat ; H(e_{t-1}) ; H(e_hat_t)] + b )

    Signals (all scalar per turn):
      d_t           = KL(e_{t-1} || e_hat_t)                  local evidence divergence
      a_{t-1}       = previous coherence                       prior grounding state
      delta_a_hat   = a_hat_t - a_{t-1}                        forward-looking alignment jump
                      where a_hat_t is computed from c_hat = e_hat_t @ v_patches
                      (no dependence on rho_t -- avoids circular dep)
      H(e_{t-1})    = entropy of prior evidence               prior grounding uncertainty
      H(e_hat_t)    = entropy of current raw evidence          new evidence uncertainty

    Trained with the same objective as DCP-Gate (L_corr + L_stab + L_switch + L_rate).
    Linear layer has 5*1 + 1 = 6 trainable parameters.

    The gate sees both the local change-point signal (d_t, like B) AND the
    dialogue state (a_{t-1}, delta_a_hat, entropies). It can learn that, e.g.,
    a high d_t in a confident, well-grounded context (high a_{t-1}, low H(e_{t-1}))
    is a real repair, whereas a high d_t in an early-uncertain context is just
    grounding establishment.
    """
    NUM_SIGNALS = 5
    SIGNAL_NAMES = ["d_t", "a_prev", "delta_a_hat", "H_e_prev", "H_e_hat"]

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(self.NUM_SIGNALS, 1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        signals: [5] tensor of scalar features in the order of SIGNAL_NAMES.
        returns scalar tensor in (0,1).
        """
        assert signals.dim() == 1 and signals.shape[0] == self.NUM_SIGNALS, (
            f"signals must be [5], got {tuple(signals.shape)}"
        )
        return torch.sigmoid(self.head(signals)).squeeze()
