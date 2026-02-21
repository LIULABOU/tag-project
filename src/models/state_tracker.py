import torch
import torch.nn as nn


class FixedGateEvidenceGRU(nn.Module):
    """
    GRU-style evidence state tracker with fixed update gate rho.

    - Hidden state IS the evidence distribution e_t (shape N)
    - Input is raw evidence distribution e_hat_t (shape N)
    - No projections: hidden_dim = input_dim = N
    - Controlled simulation: rho is fixed (e.g., 0.1)

    Update:
        e_t = (1 - rho) * e_{t-1} + rho * e_hat_t
    """
    def __init__(self, num_patches: int, rho: float = 0.1):
        super().__init__()
        self.num_patches = num_patches
        self.rho = float(rho)

    @torch.no_grad()
    def forward(self, e_hat: torch.Tensor, e0: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            e_hat: [T, N] raw evidence distributions per turn (utterance-only)
            e0: optional initial evidence [N], if None uses uniform

        Returns:
            e_all: [T, N] final evidence distribution per turn (GRU state)
        """
        assert e_hat.dim() == 2, f"Expected e_hat [T,N], got {tuple(e_hat.shape)}"
        T, N = e_hat.shape
        assert N == self.num_patches, f"num_patches mismatch: tracker N={self.num_patches}, e_hat N={N}"

        device = e_hat.device
        rho = torch.tensor(self.rho, device=device).clamp(0.0, 1.0)

        # init e_{-1}
        if e0 is None:
            e_prev = torch.full((N,), 1.0 / N, device=device)
        else:
            e_prev = e0.to(device)

        e_out = []
        for t in range(T):
            x = torch.clamp(e_hat[t], min=0.0)
            x = x / (x.sum() + 1e-12)  # keep distribution

            e_t = (1.0 - rho) * e_prev + rho * x
            e_t = e_t / (e_t.sum() + 1e-12)

            e_out.append(e_t)
            e_prev = e_t

        return torch.stack(e_out, dim=0)  # [T, N]
