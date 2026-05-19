# Changelog — v2 Pipeline Upgrade

This document records every change introduced in the v2 refactor of the
Geometric-Semantic Affordance project. All v2 changes were designed to be
**fully additive and reversible**: the original code paths (`models/backbone.py`,
`models/decoder.py`, `utils/dataset.py`, `scripts/01_train.py`) are not
modified, so you can A/B test v1 against v2 without losing the baseline.

---

## 1. Motivation

Four problems were identified in the v1 pipeline:

1. **Spatial-recovery bottleneck.** The decoder upsampled a single 32×32 ViT
   grid to 448×448 with no skip connections back to the RGB image. High-
   frequency detail (sharp affordance boundaries, thin handles) had to be
   hallucinated from semantic tokens alone.
2. **Numerical instability.** The decoder applied `sigmoid()` and was paired
   with `nn.BCELoss`, which is the unstable form. The numerically stable form
   is `BCEWithLogitsLoss` operating on raw logits.
3. **Geometric inconsistency in the dataset.** Surface normals were computed
   with hardcoded Kinect-v1 intrinsics, and the principal point was not
   shifted after the center crop, biasing the back-projection.
4. **No augmentation.** UMD always centers its tools; the model never saw
   off-center, rotated, occluded, or photometrically varied inputs, so it
   generalized poorly to in-the-wild captures.

---

## 2. New files

Seven new files were added. None of them are imported by v1 code, so removing
them rolls back cleanly.

| File | Purpose |
|---|---|
| `models/backbone_v2.py` | Multi-scale DINOv2 backbone — returns features from 4 intermediate transformer blocks instead of only the last layer. |
| `models/decoder_v2.py` | Multi-scale fusion decoder with RGB skip connections; outputs raw logits for the mask head; gives the normal head its own refinement convs. |
| `utils/losses.py` | `DiceBCELoss` (BCE-with-logits + soft Dice), `masked_cosine_loss`, `edge_aware_normal_smoothness`, `angle_error_degrees`, `iou`. |
| `utils/augmentations.py` | `JointTrainTransform` — applies geometric and photometric augmentations consistently across RGB, mask, and normal vectors (with correct vector rotation and horizontal-flip sign flip). |
| `utils/dataset_v2.py` | Drop-in dataset with configurable intrinsics, principal-point correction after crop, optional augmentation, and optional depth tensor output. |
| `scripts/02_train_v2.py` | New training entry point assembling all of the above; standard `argparse`; logs IoU and mean angular error in degrees. |
| `docs/CHANGES_V2.md` | This file. |

---

## 3. Modified files

Only one file was modified, and only by appending. The original block is
untouched.

### `config.py`

Appended (delete this block to revert):

```python
# =====================================================================
# v2 ADDITIONS — used only by scripts/02_train_v2.py and dataset_v2.py.
# Safe to delete this block to revert to the original config.
# =====================================================================

CAMERA_INTRINSICS = {
    'kinect_v1':      dict(fx=525.0, fy=525.0, cx=320.0, cy=240.0),
    'realsense_d435': dict(fx=615.0, fy=615.0, cx=320.0, cy=240.0),
    'femto_bolt':     dict(fx=605.0, fy=605.0, cx=320.0, cy=240.0),
}
TRAIN_INTRINSICS     = CAMERA_INTRINSICS['kinect_v1']
INFERENCE_INTRINSICS = CAMERA_INTRINSICS['kinect_v1']
```

> The RealSense and Femto Bolt values are typical defaults. Replace them with
> calibrated values from your specific cameras before deployment.

---

## 4. Architectural changes — v1 vs v2

| Aspect | v1 | v2 |
|---|---|---|
| Backbone feature taps | last layer only (1 × 32×32×384) | layers 2, 5, 8, 11 (4 × 32×32×384) |
| RGB skip connections | none | trainable CNN stem with 32 / 64 / 96 / 128 ch at 448 / 224 / 112 / 56 |
| ViT feature fusion | direct upsample | 1×1 projection per layer + concat + ConvBlock at 32×32 |
| Mask head output | `sigmoid` applied in decoder | raw logits (sigmoid applied only at inference) |
| Mask loss | `nn.BCELoss` (unstable) | `BCEWithLogitsLoss` + soft Dice |
| Normal head | shared trunk only | shared trunk + dedicated Conv-BN-ReLU refinement |
| Normal regularization | none | edge-aware smoothness (`exp(-grad RGB)` weighted) |
| Augmentation | none | rotation, scale, h-flip, color jitter, Gaussian noise, random erasing |
| Normal-vector consistency under aug | n/a | rotated and flipped so they remain physically correct |
| Camera intrinsics | hardcoded in `geometry.py` | configurable per-sensor in `config.py` |
| Principal point under crop | not shifted (slight bias) | shifted to the cropped frame |
| Logged val metrics | aggregate loss | aggregate loss + IoU + mean angular error in degrees |
| Optimizer | Adam, no weight decay | AdamW, `weight_decay=1e-4` |
| CLI parsing | `parser.add_y_argument = parser.add_argument` (typo-style indirection) | standard `add_argument` |

---

## 5. API changes

### `decoder.forward(...)` (v1)

```python
mask_pred, normal_pred = decoder(features)
# mask_pred is probabilities in [0,1] (sigmoid already applied)
loss = nn.BCELoss()(mask_pred, gt_mask)
```

### `decoder_v2.forward(...)` (v2)

```python
mask_logits, normal_pred = decoder(vit_feats, rgb)
# mask_logits are raw — DO NOT apply sigmoid here
loss = DiceBCELoss()(mask_logits, gt_mask)         # for training
prob = torch.sigmoid(mask_logits)                  # for visualization / inference
```

Two breaking differences for v2-side code only (v1 unaffected):

1. The decoder accepts a **list** of ViT feature maps and the **RGB tensor**,
   rather than a single feature tensor.
2. The mask output is **logits**, not probabilities. Anything reading the v2
   decoder must apply `sigmoid()` itself when it wants a probability.

---

## 6. Running v2

```bash
# Train v2
python scripts/02_train_v2.py --epochs 25 --batch_size 8

# Train v2 without augmentation (useful for isolated A/B)
python scripts/02_train_v2.py --epochs 25 --batch_size 8 --no_augment

# Resume from checkpoint
python scripts/02_train_v2.py --resume --epochs 50
```

Checkpoints land in `checkpoints_v2/` (or
`/content/drive/MyDrive/robotic_affordance_project/checkpoints_v2/` if
`--use_drive` is passed). The v1 `checkpoints/` directory is untouched.

---

## 7. Suggested A/B protocol

To measure the impact of v2 on your dataset:

1. Train `scripts/01_train.py` to convergence with default hyperparameters.
   Record best val loss and (after extending v1 with an IoU/angular-error
   evaluator) best IoU and angular error in degrees.
2. Train `scripts/02_train_v2.py` with the same `--epochs` and `--batch_size`.
3. Compare. The v2 numbers reported every epoch already include IoU and
   angular error, so you only need to add the same metrics to the v1 loop for
   parity. Expected directional outcome: lower angular error and higher IoU
   from v2; the size of the gap is your empirical signal on whether the extra
   capacity is worth it.

For an ablation, you can also train v2 with `--no_augment` to isolate the
contribution of the augmentation pipeline from the architectural changes.

---

## 8. How to revert v2

```bash
rm models/backbone_v2.py
rm models/decoder_v2.py
rm utils/losses.py
rm utils/augmentations.py
rm utils/dataset_v2.py
rm scripts/02_train_v2.py
rm docs/CHANGES_V2.md
# Then in config.py, delete the block under "# v2 ADDITIONS".
```

Nothing in v1 imports anything from v2, so the v1 pipeline keeps working
unchanged.

---

## 9. Deliberately deferred

The following items were considered but not included in this batch. They are
sketched here so the next iteration can pick them up:

- **RGB-D variant.** `dataset_v2.py` already supports `return_depth=True`; the
  v2 decoder does not yet consume depth. The cleanest next step is a parallel
  shallow CNN encoder on the depth/normal map that fuses into the decoder at
  56×56 and 112×112, then an A/B against RGB-only.
- **Uncertainty head.** Either MC dropout or an evidential output for the
  mask, so the robot can gate execution on confidence. Important for the
  startup deployment, especially on adversarial materials.
- **Synthetic clutter augmentation.** Compositing two or three random UMD
  crops per scene with depth-aware blending. Addresses the cross-object
  occlusion edge case.
- **Multi-resolution test-time inference.** Run at 448 and 672 and average
  the mask, to recover thin sub-patch structures.
- **DINOv2 partial unfreeze.** Once the decoder converges, optionally
  fine-tune the last 2–4 ViT blocks at a 10× lower learning rate.
