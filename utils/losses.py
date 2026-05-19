"""
losses.py — Loss functions and metrics for the v2 training pipeline.

WHY:
  - Your old code does sigmoid + BCELoss which is numerically unstable.
    DiceBCELoss here uses BCEWithLogitsLoss (works on RAW logits) + soft Dice.
    Dice is critical because affordance masks are class-imbalanced (most pixels
    are background).
  - masked_cosine_loss is your original normals loss, restated cleanly.
  - edge_aware_normal_smoothness penalizes normal jitter inside flat regions
    while still allowing breaks where the RGB image has edges. This is the
    standard self-supervised smoothness term (Godard et al., Monodepth).
  - angle_error_degrees gives you a human-readable metric per epoch — much more
    interpretable than 1 - cosine.

REVERT: delete this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Mask loss
# =====================================================================
class DiceBCELoss(nn.Module):
    """BCE-with-logits + soft Dice. Pass RAW logits, not probabilities."""
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1 - (2 * inter + 1) / (union + 1)
        return self.bce_w * bce + self.dice_w * dice.mean()


# =====================================================================
# 2. Normal losses
# =====================================================================
def masked_cosine_loss(pred_normals, gt_normals, gt_mask):
    """Cosine-distance loss, averaged over GT mask pixels only."""
    pred = F.normalize(pred_normals, p=2, dim=1)
    gt   = F.normalize(gt_normals,   p=2, dim=1)
    sim  = F.cosine_similarity(pred, gt, dim=1)        # [B, H, W]
    loss_map = 1 - sim
    active = (gt_mask > 0).squeeze(1)
    if active.sum() == 0:
        return torch.tensor(0.0, device=pred_normals.device, requires_grad=True)
    return loss_map[active].mean()


def edge_aware_normal_smoothness(normals, rgb, edge_sharpness: float = 10.0):
    """
    Encourage normals to be smooth across the image, but allow discontinuities
    where the RGB image itself has strong edges.

    Args:
        normals: [B, 3, H, W] (after L2-normalization is fine)
        rgb:     [B, 3, H, W] normalized RGB (network input)
    """
    dn_dx = (normals[:, :, :, 1:] - normals[:, :, :, :-1]).abs().sum(dim=1, keepdim=True)
    dn_dy = (normals[:, :, 1:, :] - normals[:, :, :-1, :]).abs().sum(dim=1, keepdim=True)

    di_dx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    di_dy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

    wx = torch.exp(-edge_sharpness * di_dx)
    wy = torch.exp(-edge_sharpness * di_dy)
    return (dn_dx * wx).mean() + (dn_dy * wy).mean()


# =====================================================================
# 3. Metric
# =====================================================================
def angle_error_degrees(pred_normals, gt_normals, gt_mask):
    """Mean angular error in degrees over GT mask pixels."""
    pred = F.normalize(pred_normals, p=2, dim=1)
    gt   = F.normalize(gt_normals,   p=2, dim=1)
    cos = (pred * gt).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    deg = torch.acos(cos) * (180.0 / 3.141592653589793)
    active = (gt_mask > 0).squeeze(1)
    if active.sum() == 0:
        return torch.tensor(float('nan'))
    return deg[active].mean()


def iou(logits, target, thresh: float = 0.5):
    """Binary IoU (predicted sigmoid > thresh vs binary target)."""
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-6)).item()
