"""
backbone.py — Multi-scale DINOv2 backbone.

Returns features from multiple intermediate transformer blocks (DPT-style) instead
of only the final layer. Earlier ViT blocks retain more local / spatial detail
before global self-attention has fully diffused it, which substantially improves
downstream dense-prediction quality.

The previous single-scale baseline is preserved under archive/v1/ for reference.
"""

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    def __init__(self, freeze: bool = True, layers=(2, 5, 8, 11)):
        """
        Args:
            freeze: keep ViT weights frozen (recommended; matches your current setup).
            layers: which transformer block outputs to extract. Default = 4 evenly
                    spaced layers across DINOv2 ViT-Small's 12 blocks.
        """
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.layers = list(layers)
        self.embed_dim = 384  # ViT-Small

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] with H,W divisible by 14.
        Returns:
            List[Tensor] of length len(self.layers), each [B, embed_dim, H/14, W/14].
        """
        B, C, H, W = x.shape
        assert H % 14 == 0 and W % 14 == 0, f"H,W must be /14. Got {H}x{W}"

        # get_intermediate_layers handles reshape into a spatial grid for us.
        feats = self.encoder.get_intermediate_layers(
            x,
            n=self.layers,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return list(feats)
