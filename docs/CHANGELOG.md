# Changelog

This file records substantive architectural changes to the affordance pipeline.
Each entry describes the motivation, the change, the resulting file layout,
and any breaking API consequences.

The earlier baseline is preserved in `archive/v1/` so any quantitative result
attached to the original architecture remains reproducible. Nothing in the
current codebase imports from `archive/v1/`; the directory is a frozen
reference.

---

## Current architecture — multi-scale fusion + RGB skip connections

### Motivation

Four problems were identified in the original baseline:

1. **Spatial-recovery bottleneck.** The original decoder upsampled a single
   32 × 32 ViT grid to 448 × 448 with no skip connections back to the RGB
   image. High-frequency detail (sharp affordance boundaries, thin handles)
   had to be hallucinated from semantic tokens alone.
2. **Numerical instability in the mask loss.** The decoder applied
   `sigmoid()` inside the network and was paired with `nn.BCELoss`, which is
   the unstable form. The numerically stable form is `BCEWithLogitsLoss`
   operating on raw logits.
3. **Geometric inconsistency in the dataset.** Surface normals were computed
   with hardcoded Kinect-v1 intrinsics, and the principal point was not
   shifted after the center crop, biasing the back-projection.
4. **No augmentation.** UMD always centers its tools; the model never saw
   off-center, rotated, occluded, or photometrically varied inputs, so it
   generalized poorly to in-the-wild captures.

### Changes summary

| Aspect | Baseline (archive/v1) | Current |
|---|---|---|
| Backbone feature taps | last layer only (1 × 32×32×384) | layers 2, 5, 8, 11 (4 × 32×32×384) |
| RGB skip connections | none | trainable CNN stem with 32 / 64 / 96 / 128 ch at 448 / 224 / 112 / 56 |
| ViT feature fusion | direct upsample | 1×1 projection per layer + concat + ConvBlock at 32×32 |
| Mask head output | `sigmoid` applied in decoder | raw logits |
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

### File layout consequences

New files introduced for the current architecture:

| File | Purpose |
|---|---|
| `utils/losses.py` | `DiceBCELoss`, `masked_cosine_loss`, `edge_aware_normal_smoothness`, `angle_error_degrees`, `iou`. |
| `utils/augmentations.py` | `JointTrainTransform` applying geometric and photometric augmentations consistently across RGB, mask, and normal vectors. |

Rewritten files (originals preserved in `archive/v1/`):

| File | What it does now |
|---|---|
| `models/backbone.py` | Multi-scale DINOv2 backbone returning features from four intermediate transformer blocks. |
| `models/decoder.py` | Multi-scale fusion decoder with RGB skip connections; outputs raw logits for the mask head and gives the normal head its own refinement convs. |
| `utils/dataset.py` | UMD dataset with configurable intrinsics, principal-point correction after crop, optional augmentation, and optional depth tensor output. |
| `scripts/train.py` | Training entry point assembling all of the above; standard `argparse`; logs IoU and mean angular error in degrees. |

Edited:

- `config.py` — added per-sensor `CAMERA_INTRINSICS` plus `TRAIN_INTRINSICS`
  and `INFERENCE_INTRINSICS`.

### API consequences (breaking)

Anything that wrote against the original API needs updating before importing
from the current modules. Notebooks under `notebooks/` were written against
the baseline and will require these adjustments:

1. The decoder now accepts a **list** of ViT feature maps and the **RGB
   tensor**, rather than a single feature tensor:

   ```python
   # before
   mask_pred, normal_pred = decoder(features)
   # now
   mask_logits, normal_pred = decoder(vit_feats, rgb)
   ```

2. The mask output is now **logits**, not probabilities. Anything reading
   the decoder must apply `sigmoid()` itself when it wants a probability:

   ```python
   prob = torch.sigmoid(mask_logits)
   ```

3. The backbone returns a **list of tensors** (one per tapped layer), not a
   single tensor.

### How to reach the baseline behaviour

`archive/v1/` contains the original `models/backbone.py`, `models/decoder.py`,
`utils/dataset.py`, and `scripts/01_train.py`. To run that baseline you would
need to either import from `archive/v1/` directly or copy those files back
into the working tree. There is no supported path to use the current decoder
with the baseline backbone or vice-versa; the two stacks are documented for
historical reference, not for mixing.

---

## Deferred

Items sketched here so the next iteration can pick them up:

- **RGB-D variant.** `dataset.py` already supports `return_depth=True`; the
  decoder does not yet consume depth. The cleanest next step is a parallel
  shallow CNN encoder on the depth / normal map that fuses into the decoder at
  56 × 56 and 112 × 112, then an A/B against RGB-only.
- **Uncertainty head.** Either MC dropout or an evidential output for the
  mask, so the robot can gate execution on confidence. Important for the
  startup deployment, especially on adversarial materials.
- **Synthetic clutter augmentation.** Compositing two or three random UMD
  crops per scene with depth-aware blending. Addresses the cross-object
  occlusion edge case.
- **Multi-resolution test-time inference.** Run at 448 and 672 and average
  the mask, to recover thin sub-patch structures.
- **DINOv2 partial unfreeze.** Once the decoder converges, optionally
  fine-tune the last 2 – 4 ViT blocks at a 10× lower learning rate.
