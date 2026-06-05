# Changelog

This file records substantive architectural changes to the affordance pipeline.
Each entry describes the motivation, the change, the resulting file layout,
and any breaking API consequences.

The earlier baseline is preserved in `archive/v1/` so any quantitative result
attached to the original architecture remains reproducible. Nothing in the
current codebase imports from `archive/v1/`; the directory is a frozen
reference.

---

## Current architecture — multi-class (7-channel) affordance head

### Motivation

The previous mask head produced a single binary channel built from UMD class
IDs 1 (`grasp`) and 7 (`wrap-grasp`) only. Diagnostic inspection of the
data exploration notebook revealed that several tool families
(`bowl_*`, `turner_*`, `scoop_*`) have most or all of their affordance in
classes that were excluded from supervision: bowls only have `contain` (class
4), turners primarily `support` (class 6), scoops `scoop` (class 3). The
post-hoc evaluation IoU=0.0 on bowls was therefore an honest reflection of
the supervision target, not a model failure. Collapsing seven semantic
affordances into one "where do I touch this" binary mask also discards the
information a humanoid actually needs: a robot reasoning about "cut the
apple" needs to distinguish the blade from the handle, not just locate the
tool.

### Changes summary

| Aspect | Previous (binary) | Current (multi-class) |
|---|---|---|
| Mask head output channels | 1 (sigmoid binary) | 7 (sigmoid multi-label, one per affordance) |
| Supervision target | union of class IDs 1 + 7 | per-pixel multi-hot over class IDs 1–7 |
| Tool families with any supervision | knives, mugs, cups, trowels, scissors, spoons, mallets (subset) | all 21 UMD tool families |
| Mask loss | BCEWithLogitsLoss + soft Dice (single channel) | BCEWithLogitsLoss (element-wise) + per-channel Dice averaged over channels |
| Normals active mask | `gt_mask > 0` | `gt_mask.sum(channel) > 0` (union over affordances) |
| `iou` metric | binary IoU on a single channel | mean-IoU: per-class binary IoU averaged across the 7 channels |
| `iou_per_class` | n/a | per-channel IoU vector (None for absent classes) |
| Per-epoch logging | `iou`, `angle_deg` | `iou`, `iou_per_class`, `angle_deg` |
| Evaluator report | overall IoU + per-tool IoU + angle | overall + `per_class_overall` + per-tool `iou@0.5_per_class` |
| Visualization sample | 1×5 row (RGB / GT / Pred / GT-N / Pred-N) | 3×7 grid (summary row + per-class GT row + per-class Pred row) |
| `config.AFFORDANCE_CLASSES` | n/a | canonical 7-tuple defining the channel order |

### File layout consequences

No new files. Edits:

| File | What changed |
|---|---|
| `config.py` | Added `AFFORDANCE_CLASSES`, `N_AFFORDANCE_CLASSES`, `AFFORDANCE_LABEL_IDS`. |
| `utils/dataset.py` | Builds a (7, H, W) multi-hot mask from the raw `_label.mat` class IDs. Augmentation runs on the raw label image first; multi-hot expansion happens after. |
| `models/decoder.py` | `MultiTaskDecoder` accepts `n_classes` (default 7); the mask head's final 1×1 conv outputs `n_classes` channels. |
| `utils/losses.py` | `DiceBCELoss` computes per-channel Dice on spatial dims and means over batch+channels. `masked_cosine_loss` and `angle_error_degrees` use a union-of-channels active mask. New `iou_per_class` helper; `iou` is now mean-IoU. |
| `scripts/train.py` | Logs per-class IoU each epoch in `history.jsonl`. Persists `affordance_classes` in `run_config.json`. |
| `scripts/evaluate.py` | `batch_iou_per_class` replaces `batch_iou`; report contains `per_class_overall` and per-tool `iou@0.5_per_class`. |
| `scripts/visualize.py` | Sample dump renders 3 rows: summary, per-class GT, per-class predicted heatmaps. Uses a stable 8-color palette per UMD class. |

### API consequences (breaking)

1. **Dataset mask shape.** `batch['mask']` is now `[B, 7, H, W]` (was
   `[B, 1, H, W]`). Anything indexing channel 0 as "the" mask must change.

2. **Decoder output shape.** `mask_logits` is now `[B, 7, H, W]`. Inference
   code that flattened over the channel dim still works because everything
   downstream is broadcast over channels. Code that did `mask_logits[:, 0]`
   to recover the binary mask now needs to argmax + threshold or pick a
   specific affordance channel by index from `config.AFFORDANCE_CLASSES`.

3. **Checkpoint compatibility.** The previous binary `best.pth` cannot be
   loaded into the new decoder because the final `mask_head.weight` has
   shape `(7, 32, 1, 1)` instead of `(1, 32, 1, 1)`. Re-training is
   required. The earlier checkpoints should be kept (e.g. by moving
   `checkpoints/` to `checkpoints_binary/`) for the before/after comparison
   in the course report.

4. **`history.jsonl` continuity.** New runs append `iou_per_class` rows; old
   runs do not have this field. `visualize.py`'s history plotting tolerates
   either, but for clarity the new training run should write to a fresh
   checkpoint directory rather than appending to the binary run's JSONL.

### How to reach the previous (binary) behaviour

The binary mask is recoverable on the fly from the multi-class one without
re-training: take the union of channels 0 (`grasp`) and 6 (`wrap-grasp`)
from `batch['mask']`. The decoder's binary checkpoint is preserved in the
binary checkpoint directory if it was archived as suggested above.

### Optional: frequency-inverse class weights

A `--class_weights` flag was added to `scripts/train.py` after evaluation of
the first 25-epoch run revealed strong minority-class overfitting on
`support` (train IoU 0.89 → val 0.36) and `scoop` (0.88 → 0.58). The flag
scans the training set's `.mat` labels once at startup (cached to
`checkpoints/class_pixel_counts.json`) and derives a per-channel
`pos_weight = (N_neg / N_pos) ** weight_power`, clipped to `weight_clip`.
The default schedule (`weight_power=0.5`, `weight_clip=15.0`) is the
square-root inverse-frequency rule, applied conservatively. The vector is
recorded in `run_config.json` and in the first row of `history.jsonl`.

Safe to combine with `--resume`: the optimizer state from the previous
unweighted run re-equilibrates to the new gradient scale within a couple of
epochs; no checkpoint surgery required.

```bash
# Resume the existing run with class weights for 15 more epochs.
python scripts/train.py --resume --class_weights --epochs 40
```

---

## Previous architecture — multi-scale fusion + RGB skip connections

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
