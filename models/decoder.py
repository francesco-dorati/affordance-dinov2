"""
decoder.py — Multi-scale fusion decoder with RGB skip connections.

  (a) Multi-scale ViT fusion: concatenates intermediate ViT layers at 32x32.
  (b) RGB skip connections: a small trainable CNN stem produces low-level
      features at 448 / 224 / 112 / 56 resolutions that are concatenated into
      the decoder at each upsampling stage (DPT-style).
  (c) LOGITS output for the mask head — pair with BCEWithLogitsLoss in
      losses.py for numerical stability.
  (d) Light refinement convs in the normal head so it has its own filters.

SECTIONS:
  1. ConvBlock       — basic Conv-BN-ReLU x2
  2. FusionUp        — bilinear upsample + concat skip + ConvBlock
  3. RGBStem         — trainable shallow CNN producing multi-res RGB features
  4. MultiTaskDecoder — assembles everything; outputs (mask_logits, normals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. ConvBlock
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =====================================================================
# 2. FusionUp
# =====================================================================
class FusionUp(nn.Module):
    """Bilinear-upsample x to skip's resolution, concat, then ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# =====================================================================
# 3. RGBStem
# =====================================================================
class RGBStem(nn.Module):
    """
    Shallow trainable CNN that produces high-frequency RGB features at multiple
    resolutions. These are the skip connections that the original decoder lacks.

    Cheap by design (32->64->96->128 channels). Adds ~0.3M params.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )  # -> 32 ch @ 448
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )  # -> 64 ch @ 224
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )  # -> 96 ch @ 112
        self.down3 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )  # -> 128 ch @ 56

    def forward(self, rgb):
        s0 = self.stem(rgb)    # [B, 32, 448, 448]
        s1 = self.down1(s0)    # [B, 64, 224, 224]
        s2 = self.down2(s1)    # [B, 96, 112, 112]
        s3 = self.down3(s2)    # [B,128,  56,  56]
        return s0, s1, s2, s3


# =====================================================================
# 4. MultiTaskDecoder
# =====================================================================
class MultiTaskDecoder(nn.Module):
    """
    Inputs:
        vit_feats: list of N tensors from DINOv2Backbone,
                   each [B, embed_dim, 32, 32].
        rgb:       [B, 3, 448, 448] (the SAME normalized RGB you fed the ViT)
    Outputs:
        mask_logits: [B, n_classes, 448, 448]  (raw — do NOT apply sigmoid here)
        normal_pred: [B, 3, 448, 448]          (unit vectors)

    The mask head is MULTI-LABEL: each of `n_classes` channels carries an
    independent sigmoid logit for one UMD affordance class. Pair with a
    per-channel BCE + Dice loss.
    """
    def __init__(self, embed_dim: int = 384, n_vit_scales: int = 4,
                 n_classes: int = 7):
        super().__init__()
        self.n_classes = n_classes
        self.rgb_stem = RGBStem()

        # --- Multi-scale ViT fusion at 32x32 ---
        self.vit_proj = nn.ModuleList([
            nn.Conv2d(embed_dim, 256, 1) for _ in range(n_vit_scales)
        ])
        self.vit_fuse = ConvBlock(256 * n_vit_scales, 256)

        # --- Progressive upsampling with RGB skips ---
        # 32 -> 56  (skip: 128 ch)
        self.up1 = FusionUp(in_ch=256, skip_ch=128, out_ch=192)
        # 56 -> 112 (skip: 96 ch)
        self.up2 = FusionUp(in_ch=192, skip_ch=96,  out_ch=128)
        # 112 -> 224 (skip: 64 ch)
        self.up3 = FusionUp(in_ch=128, skip_ch=64,  out_ch=64)
        # 224 -> 448 (skip: 32 ch)
        self.up4 = FusionUp(in_ch=64,  skip_ch=32,  out_ch=32)

        # --- Heads ---
        self.mask_head = nn.Conv2d(32, n_classes, 1)   # per-class logits
        self.normal_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, vit_feats, rgb):
        assert len(vit_feats) == len(self.vit_proj), \
            f"Expected {len(self.vit_proj)} ViT scales, got {len(vit_feats)}"

        # Multi-scale ViT fuse -> [B, 256, 32, 32]
        projected = [proj(f) for proj, f in zip(self.vit_proj, vit_feats)]
        x = self.vit_fuse(torch.cat(projected, dim=1))

        # RGB skips
        s0, s1, s2, s3 = self.rgb_stem(rgb)
        x = self.up1(x, s3)   # 56
        x = self.up2(x, s2)   # 112
        x = self.up3(x, s1)   # 224
        x = self.up4(x, s0)   # 448

        mask_logits = self.mask_head(x)                # [B, n_classes, 448, 448]
        normal_raw  = self.normal_head(x)              # [B, 3, 448, 448]
        normal_pred = F.normalize(normal_raw, p=2, dim=1)
        return mask_logits, normal_pred
