import torch
import torch.nn as nn


class FixedGateEvidenceGRU(nn.Module):
    """
    Evidence tracker with fixed gate rho:
        e_t = (1 - rho) * e_{t-1} + rho * e_hat_t
    """
    def __init__(self, num_patches: int, rho: float = 0.1):
        super().__init__()
        self.num_patches = num_patches
        self.rho = float(rho)

    @torch.no_grad()
    def forward(self, e_hat: torch.Tensor, e0: torch.Tensor | None = None) -> torch.Tensor:
        assert e_hat.dim() == 2, f"Expected e_hat [T,N], got {tuple(e_hat.shape)}"
        T, N = e_hat.shape
        assert N == self.num_patches, f"Mismatch: tracker={self.num_patches}, input={N}"

        device = e_hat.device
        rho = torch.tensor(self.rho, device=device).clamp(0.0, 1.0)

        if e0 is None:
            e_prev = torch.full((N,), 1.0 / N, device=device)
        else:
            e_prev = e0.to(device)

        e_out = []
        for t in range(T):
            x = torch.clamp(e_hat[t], min=0.0)
            x = x / (x.sum() + 1e-12)

            e_t = (1.0 - rho) * e_prev + rho * x
            e_t = e_t / (e_t.sum() + 1e-12)

            e_out.append(e_t)
            e_prev = e_t

        return torch.stack(e_out, dim=0)


class SoftGateEvidenceTracker(nn.Module):
    """
    Evidence tracker with learned per-turn gate p_t:
        e_t = (1 - p_t) * e_{t-1} + p_t * e_hat_t

    NOTE: forward is now DIFFERENTIABLE (no @torch.no_grad). The training
    loop updates rho_t via this path so stab/switch losses on e_t can
    backprop into the repair head.
    """
    def __init__(self, num_patches: int):
        super().__init__()
        self.num_patches = num_patches

    def forward(
        self,
        e_hat: torch.Tensor,
        p_t: torch.Tensor,
        e0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert e_hat.dim() == 2, f"Expected e_hat [T,N], got {tuple(e_hat.shape)}"
        T, N = e_hat.shape
        assert N == self.num_patches, f"Mismatch: tracker={self.num_patches}, input={N}"
        assert p_t.dim() == 1, f"Expected p_t [T], got {tuple(p_t.shape)}"
        assert p_t.shape[0] == T, f"Length mismatch: p_t has {p_t.shape[0]} but e_hat has {T}"

        device = e_hat.device

        if e0 is None:
            e_prev = torch.full((N,), 1.0 / N, device=device)
        else:
            e_prev = e0.to(device)

        e_out = []
        for t in range(T):
            x = torch.clamp(e_hat[t], min=0.0)
            x = x / (x.sum() + 1e-12)

            rho = torch.clamp(p_t[t], 0.0, 1.0)
            e_t = (1.0 - rho) * e_prev + rho * x
            e_t = e_t / (e_t.sum() + 1e-12)

            e_out.append(e_t)
            e_prev = e_t

        return torch.stack(e_out, dim=0)