import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPBackbone:
    """
    Single source of truth for CLIP loading + freezing + feature extraction.

    Provides:
      - encode_image / encode_text (projected embeddings)  -> contrastive training
      - encode_vision_tokens (v_cls + patch tokens)        -> alignment heatmaps
      - encode_text_pooler (pooler_output)                 -> per-turn h_t
      - text_dim / vision_dim properties                   -> alignment init
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze CLIP
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    # -------------------------
    # Contrastive baseline APIs
    # -------------------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Projected image embedding. Shape: [B, projection_dim]."""
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=pixel_values)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Projected text embedding. Shape: [B, projection_dim]."""
        with torch.no_grad():
            return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    # -------------------------
    # Alignment / grounding APIs
    # -------------------------
    def encode_vision_tokens(self, pixel_values: torch.Tensor):
        """
        Returns tokens from CLIP vision transformer (NOT projected):
          v_cls:     [B, vision_dim]
          v_patches: [B, N, vision_dim]
        """
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            last_hidden = vision_out.last_hidden_state  # [B, 1+N, vision_dim]
            v_cls = last_hidden[:, 0, :]                # [B, vision_dim]
            v_patches = last_hidden[:, 1:, :]           # [B, N, vision_dim]
        return v_cls, v_patches

    def encode_text_pooler(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns text pooler output (NOT projected):
          h: [B, text_dim]
        """
        with torch.no_grad():
            out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            return out.pooler_output  # [B, text_dim]

    # -------------------------
    # Helpful dimensions
    # -------------------------
    @property
    def projection_dim(self) -> int:
        return self.model.config.projection_dim  # usually 512

    @property
    def vision_dim(self) -> int:
        return self.model.vision_model.config.hidden_size  # usually 768

    @property
    def text_dim(self) -> int:
        return self.model.text_model.config.hidden_size  # usually 512
