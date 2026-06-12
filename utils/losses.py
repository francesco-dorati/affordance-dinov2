"""
losses.py — Loss functions and metrics for the multi-class affordance pipeline.

WHY:
  - DiceBCELoss uses BCEWithLogitsLoss (works on RAW logits) + per-channel
    soft Dice. Multi-label, not multi-class softmax: each of the C affordance
    channels is an independent binary problem. Per-channel Dice means that
    channels with very few positive pixels (rare classes like `pound`) still
    contribute meaningfully to the loss instead of being washed out by the
    dominant `grasp` channel.
  - masked_cosine_loss / angle_error_degrees evaluate normals over the UNION
    of all affordance channels — wherever any annotated affordance pixel sits,
    we want the predicted normal to be correct.
  - edge_aware_normal_smoothness penalizes normal jitter inside flat regions
    while still allowing breaks where the RGB image has edges (Godard et al.,
    Monodepth).
  - `iou` returns the per-class binary IoU averaged across channels (mean-IoU,
    the multi-label headline metric). `iou_per_class` returns the full vector
    for diagnostics.

REVERT: delete this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Helpers
# =====================================================================
def _active_from_multihot(gt_mask: torch.Tensor) -> torch.Tensor:
    """Reduce a multi-label mask [B, C, H, W] to a per-pixel activity map
    [B, H, W] that is True wherever ANY affordance channel is active."""
    return gt_mask.sum(dim=1) > 0


# =====================================================================
# 1. Mask loss
# =====================================================================
class DiceBCELoss(nn.Module):
    """BCE-with-logits + per-channel soft Dice. Pass RAW logits.

    Inputs are [B, C, H, W] with C = number of affordance classes.
    BCE is element-wise so each channel contributes equally per pixel.
    Dice is computed per (sample, channel) on the spatial dims, then averaged
    over both batch and channels so each affordance class carries equal weight.
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        # BCEWithLogitsLoss broadcasts pos_weight right-aligned with `target`.
        # Our targets are [B, C, H, W]; a 1-D pos_weight of length C would
        # therefore try to match the W axis. Reshape to [C, 1, 1] so it
        # right-aligns as [1, C, 1, 1] and lines up with the channel dim.
        if pos_weight is not None and pos_weight.dim() == 1:
            pos_weight = pos_weight.view(-1, 1, 1)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)  # spatial only — keep channels separate
        inter = (probs * target).sum(dim=dims)            # [B, C]
        union = probs.sum(dim=dims) + target.sum(dim=dims)  # [B, C]
        dice = 1 - (2 * inter + 1) / (union + 1)            # [B, C]
        return self.bce_w * bce + self.dice_w * dice.mean()


# =====================================================================
# 2. Normal losses
# =====================================================================
def masked_cosine_loss(pred_normals, gt_normals, gt_mask):
    """Cosine-distance loss, averaged over the UNION of all affordance pixels."""
    pred = F.normalize(pred_normals, p=2, dim=1)
    gt   = F.normalize(gt_normals,   p=2, dim=1)
    sim  = F.cosine_similarity(pred, gt, dim=1)        # [B, H, W]
    loss_map = 1 - sim
    active = _active_from_multihot(gt_mask)
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
# 3. Metrics
# =====================================================================
def angle_error_degrees(pred_normals, gt_normals, gt_mask):
    """Mean angular error in degrees over the union of GT affordance pixels."""
    pred = F.normalize(pred_normals, p=2, dim=1)
    gt   = F.normalize(gt_normals,   p=2, dim=1)
    cos = (pred * gt).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    deg = torch.acos(cos) * (180.0 / 3.141592653589793)
    active = _active_from_multihot(gt_mask)
    if active.sum() == 0:
        return torch.tensor(float('nan'))
    return deg[active].mean()


def iou(logits, target, thresh: float = 0.5):
    """Mean-IoU: per-class binary IoU averaged across channels.

    Logits and target are [B, C, H, W]. Returns a single scalar — the
    headline mask metric for training-loop logging.
    """
    per_class = iou_per_class(logits, target, thresh=thresh)
    # Drop classes with no positive pixels anywhere in the batch (denominator
    # would be 0). This avoids spuriously perfect IoU=0 on absent classes.
    valid = [v for v in per_class if v is not None]
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


def iou_accumulate(logits, target, thresh: float = 0.5):
    """Per-class intersection and union pixel sums for one batch.

    Building block for DATASET-LEVEL IoU: sum these over all batches of a
    split, then divide once with `iou_from_accumulated`. Unlike averaging
    per-batch IoUs, the result is invariant to batch size and sample order,
    and batches where a class covers few pixels are weighted by their actual
    area instead of counting as much as area-rich batches.

    Returns:
        inter: [C] tensor — per-class intersection pixel counts.
        union: [C] tensor — per-class union pixel counts.
    """
    pred = (torch.sigmoid(logits) > thresh).float()
    dims = (0, 2, 3)  # batch + spatial — keep channels separate
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    return inter, union


def iou_from_accumulated(inter_sum, union_sum):
    """Final dataset-level IoU from accumulated per-class sums.

    Returns:
        mean_iou:  float — mean over classes with non-zero union (or 0.0).
        per_class: list of float-or-None (None = class absent in the split).
    """
    per_class = []
    for i, u in zip(inter_sum.tolist(), union_sum.tolist()):
        per_class.append((i / u) if u > 0 else None)
    valid = [v for v in per_class if v is not None]
    mean_iou = float(sum(valid) / len(valid)) if valid else 0.0
    return mean_iou, per_class


def iou_per_class(logits, target, thresh: float = 0.5):
    """Returns a list of per-class IoU floats (or None for absent classes).

    A class is "absent" when both prediction and target have zero positive
    pixels across the entire batch — IoU is undefined there.
    """
    pred = (torch.sigmoid(logits) > thresh).float()
    out = []
    C = pred.shape[1]
    for c in range(C):
        p = pred[:, c]
        t = target[:, c]
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        if union.item() == 0:
            out.append(None)
        else:
            out.append((inter / (union + 1e-6)).item())
    return out
