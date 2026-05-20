import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Small trainable projection head on top of frozen CLIP embeddings.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.proj(x)
