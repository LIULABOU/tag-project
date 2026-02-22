import torch
import torch.nn as nn
import torch.nn.functional as F

class MVPAligner(nn.Module):
    """
    MVP training module:
      text -> Wa
      image -> Wh
    """
    def __init__(self, text_dim=512, vision_dim=768, proj_dim=256):
        super().__init__()
        self.Wa = nn.Linear(text_dim, proj_dim, bias=False)   # TEXT -> Wa
        self.Wh = nn.Linear(vision_dim, proj_dim, bias=False) # IMAGE -> Wh

    def forward(self, h: torch.Tensor, g: torch.Tensor):
        """
        h: [B, 512] text pooled
        g: [B, 768] image vector (attended from patches)
        """
        t = F.normalize(self.Wa(h), dim=-1)
        v = F.normalize(self.Wh(g), dim=-1)
        return t, v
