import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """
    Implements the baseline grounding + alignment equations:

        q_t = W_h h_t
        e_hat = softmax(v_patches @ q_t)
        c_t = sum_i e_hat(i) * v_patches(i)
        a_t = sigmoid(W_a [q_t; c_t; v_cls])

    Inputs:
      - h_all: [T, 512] (text embeddings per turn)
      - v_cls: [768] (global image token)
      - v_patches: [N, 768] (patch tokens, e.g., N=49)
    Outputs:
      - a: [T] alignment score per turn
      - e: [T, N] attention distribution over patches per turn
    """

    def __init__(self, h_dim=512, v_dim=768, q_dim=768, bias_a=True):
        super().__init__()
        self.W_h = nn.Linear(h_dim, q_dim, bias=False)
        self.W_a = nn.Linear(q_dim + v_dim + v_dim, 1, bias=bias_a)  # [q_t; c_t; v_cls] -> 1

    def forward(self, h_all: torch.Tensor, v_cls: torch.Tensor, v_patches: torch.Tensor):
        """
        h_all: [T, 512]
        v_cls: [768]
        v_patches: [N, 768]
        returns:
          a: [T]
          e: [T, N]
        """
        # q_all: [T, 768]
        q_all = self.W_h(h_all)

        # scores: [T, N] where scores[t,i] = v_patches[i] dot q_all[t]
        scores = torch.matmul(q_all, v_patches.t())  # [T, N]

        # e_hat: [T, N]
        e_hat = F.softmax(scores, dim=-1)

        # c_all: [T, 768]
        c_all = torch.matmul(e_hat, v_patches)  # [T, 768]

        # expand v_cls to [T, 768]
        v_cls_rep = v_cls.unsqueeze(0).expand(h_all.size(0), -1)

        # x: [T, 2304]
        x = torch.cat([q_all, c_all, v_cls_rep], dim=-1)

        # a: [T]
        a = torch.sigmoid(self.W_a(x)).squeeze(-1)

        return a, e_hat

@torch.no_grad()
def evidence_dynamics(e_hat: torch.Tensor, rho: float = 0.1, e0: torch.Tensor | None = None) -> torch.Tensor:
    """
    Implements: e_t = (1-rho) e_{t-1} + rho * e_hat_t
    Args:
      e_hat: [T, N] raw evidence distributions (each row ~ sums to 1)
      rho: fixed repair gate (prof said 0.1)
      e0: optional initial evidence distribution [N]. If None, uniform prior.
    Returns:
      e: [T, N] final evidence distribution per turn
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
        # safety: keep it a proper distribution
        x = torch.clamp(e_hat[t], min=0.0)
        x = x / (x.sum() + 1e-12)

        e_t = (1.0 - rho_t) * e_prev + rho_t * x
        e_t = e_t / (e_t.sum() + 1e-12)

        e_out.append(e_t)
        e_prev = e_t

    return torch.stack(e_out, dim=0)  # [T, N]
