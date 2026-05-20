import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """
    Implements the baseline grounding + alignment equations:

        q_t = W_h h_t                    (Eq. 2 / Eq. 6 in report; 'Wa_text' here = W_h)
        e_hat = softmax(v_patches @ q_t) (Eq. 7)
        c_t = sum_i e_hat(i) * v_patches(i) (Eq. 9)
        a_t = sigmoid(W_a [q_t; c_t; v_cls]) (Eq. 10)

    NOTE: Per the report, this single W_h is shared across grounding,
    coherence head, and contrastive loss. The training loop is responsible
    for calling project_query() once per turn and reusing q_t everywhere.

    Inputs:
      - h_all: [T, h_dim] (text embeddings per turn)
      - v_cls: [v_dim] (global image token)
      - v_patches: [N, v_dim] (patch tokens, e.g., N=49)
    """

    def __init__(self, h_dim=512, v_dim=768, q_dim=768, bias_a=True):
        super().__init__()
        self.Wa_text = nn.Linear(h_dim, q_dim, bias=False)  # text -> query (W_h in report)
        self.W_a = nn.Linear(q_dim + v_dim + v_dim, 1, bias=bias_a)  # [q_t; c_t; v_cls] -> 1

    # ------------------------------------------------------------------
    # Per-step helpers (used by the differentiable per-turn training loop)
    # ------------------------------------------------------------------
    def project_query(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., h_dim] -> q: [..., q_dim]
        Works for both [T, h_dim] and [h_dim] inputs.
        """
        return self.Wa_text(h)

    def raw_evidence(self, q: torch.Tensor, v_patches: torch.Tensor) -> torch.Tensor:
        """
        q: [T, q_dim]      v_patches: [N, v_dim]   (q_dim must equal v_dim)
        returns e_hat: [T, N], softmax over patches.
        """
        scores = torch.matmul(q, v_patches.t())  # [T, N]
        return F.softmax(scores, dim=-1)

    def alignment_score(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        v_cls: torch.Tensor,
    ) -> torch.Tensor:
        """
        q:     [T, q_dim]   projected text query
        c:     [T, v_dim]   evidence-aware visual summary (post evidence dynamics)
        v_cls: [v_dim]      global image token
        returns a_tilde: [T] in (0,1)
        """
        v_cls_rep = v_cls.unsqueeze(0).expand(q.size(0), -1)
        x = torch.cat([q, c, v_cls_rep], dim=-1)
        return torch.sigmoid(self.W_a(x)).squeeze(-1)

    # ------------------------------------------------------------------
    # Convenience: full forward producing raw e_hat (no evidence dynamics)
    # ------------------------------------------------------------------
    def forward(self, h_all: torch.Tensor, v_cls: torch.Tensor, v_patches: torch.Tensor):
        q_all = self.project_query(h_all)
        e_hat = self.raw_evidence(q_all, v_patches)
        c_all = torch.matmul(e_hat, v_patches)
        a = self.alignment_score(q_all, c_all, v_cls)
        return a, e_hat


@torch.no_grad()
def evidence_dynamics(e_hat: torch.Tensor, rho: float = 0.1, e0: torch.Tensor | None = None) -> torch.Tensor:
    """
    Implements: e_t = (1-rho) e_{t-1} + rho * e_hat_t
    NOTE: kept for eval/sanity-viz scripts. The training loop now does this
    inline (and differentiably), so this helper is not used during training.
    """
    assert e_hat.dim() == 2, f"Expected e_hat [T,N], got {tuple(e_hat.shape)}"
    T, N = e_hat.shape
    device = e_hat.device

    rho_t = torch.tensor(rho, device=device).clamp(0.0, 1.0)

    if e0 is None:
        e_prev = torch.full((N,), 1.0 / N, device=device)
    else:
        e_prev = e0.to(device)

    e_out = []
    for t in range(T):
        x = torch.clamp(e_hat[t], min=0.0)
        x = x / (x.sum() + 1e-12)

        e_t = (1.0 - rho_t) * e_prev + rho_t * x
        e_t = e_t / (e_t.sum() + 1e-12)

        e_out.append(e_t)
        e_prev = e_t

    return torch.stack(e_out, dim=0)  # [T, N]
