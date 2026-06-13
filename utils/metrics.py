"""
metrics.py — Benchmark-comparable evaluation metrics for affordance segmentation.

WHY:
  The UMD affordance literature (Myers et al. 2015; AffordanceNet, Do et al.
  ICRA 2018) does NOT report IoU. The standard metric is the **weighted
  F-measure** F_beta^omega of Margolin, Zelnik-Manor & Tal ("How to Evaluate
  Foreground Maps", CVPR 2014), evaluated per affordance class as a binary
  foreground map and averaged across classes. AffordanceNet's UMD Table II
  (average 0.799) is in exactly this metric. To place our number on that
  table we must report F_beta^omega, not mean-IoU.

  beta^2 = 0.3 (precision-weighted), matching Margolin's default and the
  affordance papers.

  This module operates on continuous foreground maps (the per-class sigmoid
  probability, NOT a thresholded mask) against a binary ground truth — the
  weighted F-measure is designed to consume soft maps directly.

REVERT: delete this file; remove the F_beta^omega block from evaluate.py.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def weighted_f_measure(pred: np.ndarray, gt: np.ndarray,
                       beta2: float = 0.3, eps: float = 1e-9):
    """Margolin et al. (CVPR 2014) weighted F-measure for one foreground map.

    Args:
        pred:  HxW float array in [0, 1] — continuous foreground (probability)
               map for ONE affordance class.
        gt:    HxW array, treated as boolean — ground-truth foreground for the
               same class.
        beta2: beta^2 weighting (0.3 = precision-weighted, the standard).

    Returns:
        F_beta^omega in [0, 1], or np.nan if the class has no GT foreground in
        this image (so it can be skipped by nan-aware aggregation).

    Port of the reference MATLAB `WFb.m`:
        E = |FG - GT|; errors are propagated from each background pixel to its
        nearest foreground pixel, Gaussian-smoothed (fspecial('gaussian',7,5)),
        weighted by a distance-decaying importance map B, then turned into
        weighted TP/FP and combined into the F-measure.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt).astype(bool)
    if gt.sum() == 0:
        return np.nan  # class absent in GT for this image — undefined

    dgt = gt.astype(np.float64)
    E = np.abs(pred - dgt)

    # Distance to the nearest GT-foreground pixel, and the index of that pixel.
    # distance_transform_edt(~gt) gives, for every pixel, distance to the
    # nearest zero of (~gt) == nearest True of gt; return_indices gives the
    # coordinates of that nearest foreground pixel (== MATLAB bwdist(gt)).
    dst, inds = distance_transform_edt(~gt, return_indices=True)

    # Propagate background error to the nearest foreground pixel's error.
    Et = E.copy()
    bg = ~gt
    Et[bg] = E[inds[0][bg], inds[1][bg]]

    # Gaussian of Et: fspecial('gaussian', 7, 5) -> sigma=5, 7x7 (radius 3),
    # zero ("constant") padding to match MATLAB imfilter's default.
    EA = gaussian_filter(Et, sigma=5.0, truncate=3.0 / 5.0, mode="constant")

    MIN_E_EA = E.copy()
    sel = gt & (EA < E)
    MIN_E_EA[sel] = EA[sel]

    # Pixel-importance weighting B: foreground = 1; background decays with
    # distance to the nearest foreground pixel.
    B = np.ones_like(dgt)
    B[bg] = 2.0 - np.exp(np.log(0.5) / 5.0 * dst[bg])

    Ew = MIN_E_EA * B
    TPw = dgt.sum() - Ew[gt].sum()
    FPw = Ew[bg].sum()

    R = 1.0 - Ew[gt].mean()              # weighted recall
    P = TPw / (eps + TPw + FPw)          # weighted precision
    Q = (1 + beta2) * P * R / (eps + beta2 * P + R)
    return float(Q)


def weighted_f_measure_per_class(pred_probs: np.ndarray, gt_masks: np.ndarray,
                                 beta2: float = 0.3):
    """Weighted F-measure for every channel of one sample.

    Args:
        pred_probs: (C, H, W) float in [0, 1] — per-class sigmoid maps.
        gt_masks:   (C, H, W) binary ground truth.

    Returns:
        np.ndarray of shape (C,) with F_beta^omega per class; np.nan where the
        class is absent from the GT of this sample.
    """
    C = pred_probs.shape[0]
    out = np.full(C, np.nan, dtype=np.float64)
    for c in range(C):
        out[c] = weighted_f_measure(pred_probs[c], gt_masks[c], beta2=beta2)
    return out


def _selftest():
    """Sanity checks: perfect map -> 1, empty/inverted -> ~0, monotonic."""
    rng = np.random.RandomState(0)
    gt = np.zeros((64, 64), np.uint8)
    gt[20:44, 20:44] = 1

    perfect = weighted_f_measure(gt.astype(float), gt)
    empty = weighted_f_measure(np.zeros_like(gt, float), gt)
    inverted = weighted_f_measure(1 - gt.astype(float), gt)
    # A slightly noisy-but-good map should score below perfect, above empty.
    noisy = np.clip(gt.astype(float) + rng.normal(0, 0.15, gt.shape), 0, 1)
    noisy_q = weighted_f_measure(noisy, gt)
    absent = weighted_f_measure(np.zeros_like(gt, float), np.zeros_like(gt))

    print(f"perfect  = {perfect:.4f}  (expect 1.000)")
    print(f"noisy    = {noisy_q:.4f}  (expect 0 < q < 1)")
    print(f"empty    = {empty:.4f}  (expect ~0)")
    print(f"inverted = {inverted:.4f}  (expect ~0)")
    print(f"absent   = {absent}  (expect nan)")
    ok = (abs(perfect - 1.0) < 1e-6 and empty < 0.05 and inverted < 0.05
          and 0.0 < noisy_q < 1.0 and np.isnan(absent))
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
