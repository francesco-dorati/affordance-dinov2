# Geometric-Semantic Fusion for Autonomous Robotic Affordance

## 1. Project Overview

**Title:** Spatial Resolution Recovery and Multi-Task Geometric Estimation for
Robotic Affordance Perception.

**Objective:** Develop a high-precision perception pipeline that identifies
"affordances" (actionable regions) on unseen objects. The system bridges
high-level semantic understanding (what an object is) and low-level robotic
execution (where and how to grab it) by directly predicting pixel-perfect
affordance masks and surface normals.

**Motivation:** Most robotic manipulation systems rely on pre-defined 3D CAD
models. This project enables "zero-shot" interaction. When a robot encounters a
novel tool, it must identify the specific affordance region (e.g. the handle
for grasping) and the correct approach orientation (surface normal) directly
from sensory data, without real-time 3D point cloud reconstruction.

This codebase is the final project for a Computer Vision course and is also
intended to evolve into a perception module for a humanoid robotics startup.

---

## 2. High-Level System Architecture

The system accepts multi-modal sensory data and uses a multi-task neural
network to produce a deterministic robotic approach packet.

### Global inputs (sensory layer)

- **RGB image:** 448 × 448 × 3 (color and texture features).
- **Depth map:** 448 × 448 × 1 (used to generate ground-truth normals during
  training; reserved for an RGB-D variant on the input side).

### Global outputs (action layer)

- **Multi-class affordance mask:** 448 × 448 × 7 high-resolution segmentation,
  one independent sigmoid channel per UMD affordance class (multi-label, not
  multi-class softmax — a pixel may belong to several affordances at once).
  Channel order is defined by `config.AFFORDANCE_CLASSES`:

  | Idx | Class | Meaning for a humanoid |
  |---|---|---|
  | 0 | `grasp`       | Region you grip to pick the object up (handles, shafts). |
  | 1 | `cut`         | Sharp edge that severs material (knife / scissor blades). |
  | 2 | `scoop`       | Concave surface that lifts loose material (spoon bowl). |
  | 3 | `contain`     | Interior cavity that holds material (cup / mug / bowl interior). |
  | 4 | `pound`       | Heavy striking face that delivers impact (mallet head). |
  | 5 | `support`     | Broad flat region that supports another object (turner blade, trowel face). |
  | 6 | `wrap-grasp`  | Graspable region wrapped by the hand (mug handle, cylindrical grips). |

- **Dense surface normal map:** 448 × 448 × 3 unit-vector field representing
  surface orientation (`N_x, N_y, N_z`) for a collision-free approach.
- **3D approach centroid:** the (X, Y, Z) coordinate of the target, derived
  from the centroid of the selected affordance channel and its corresponding
  depth value, back-projected through the calibrated camera intrinsics. The
  caller picks which channel to use based on the intended task ("grasp this",
  "pour into this", "cut with this").

---

## 3. Detailed Pipeline Stages

### Stage 1 — Semantic feature extraction (the sensor)

A frozen DINOv2 ViT-Small extracts semantic features at multiple depths.
Rather than tapping only the final transformer block, the backbone returns
features from four intermediate layers (default: layers 2, 5, 8, 11). Earlier
blocks retain more local detail before global self-attention has fully
diffused it; this is the same insight that underpins the DPT architecture and
substantially improves downstream dense-prediction quality.

- **Implementation:** `models/backbone.py` → `DINOv2Backbone`
- **Output:** list of four tensors, each `[B, 384, 32, 32]`.

### Stage 2 — Multi-task convolutional refinement (the learning core)

The decoder performs "spatial resolution recovery" from a 32 × 32 token grid
back to a 448 × 448 dense prediction, and predicts the affordance mask and
local surface geometry simultaneously.

Three structural decisions matter for geometric precision:

1. **Multi-scale ViT fusion at 32 × 32.** Each of the four ViT layers is
   1 × 1-projected and concatenated, then fused by a ConvBlock.
2. **RGB skip connections.** A small trainable CNN stem (`RGBStem`) produces
   high-frequency features at 56 / 112 / 224 / 448 resolutions. These are
   concatenated into the decoder at each upsampling stage, supplying the
   spatial detail the ViT alone cannot reconstruct.
3. **LOGITS output for the mask head.** The decoder returns raw logits
   (paired with `BCEWithLogitsLoss` for numerical stability) instead of
   applying `sigmoid` internally. The normal head has its own small refinement
   block so it does not compete with the mask head's filters in the shared
   trunk.

- **Implementation:** `models/decoder.py` → `MultiTaskDecoder`
- **Output:** `(mask_logits, normal_pred)` where `normal_pred` is L2-normalized
  to unit vectors.

### Stage 3 — Actionable inference (the robotics core)

The final robotic command is extracted from the network outputs and combined
with the camera intrinsics:

- Compute the 2D centroid (u, v) of the predicted affordance mask.
- Sample depth Z at (u, v); use inverse perspective mapping to find (X, Y, Z).
- Sample the predicted normal map at (u, v) to get the approach vector.

The intrinsics come from `config.INFERENCE_INTRINSICS`, which can be swapped
per deployment without touching code.

**Output:** the final robotic pose (X, Y, Z, N_x, N_y, N_z).

---

## 4. Datasets and Technical Requirements

**Primary dataset:** UMD Part Affordance Dataset. Real-world RGB-D captures
from a Kinect sensor, 105 kitchen, workshop, and gardening tools, labeled
with seven per-pixel affordance classes (`grasp`, `cut`, `scoop`, `contain`,
`pound`, `support`, `wrap-grasp`). The model is supervised on all seven
classes simultaneously; each class is an independent binary prediction.

**In-the-wild test set:** custom captures from a modern depth camera in an
office, containing completely novel objects under varied lighting, used to
evaluate sim-to-real generalization qualitatively.

**Framework:** PyTorch.

**Evaluation metrics:**

- **Mean-IoU** at threshold 0.5: per-class binary IoU averaged across the
  seven affordance channels. Headline number for the mask head.
- **Per-class IoU**: full breakdown so weak classes (e.g. `support`, `pound`)
  are not hidden behind strong ones (`grasp`, `wrap-grasp`).
- **Mean angular error in degrees** (computed via `acos(cosine_similarity)`)
  for surface normal accuracy, evaluated over the union of all annotated
  affordance pixels per sample.

---

## 5. Augmentation and Loss Functions

### Joint augmentation pipeline (`utils/augmentations.py`)

Geometric augmentations are applied **consistently** across RGB, mask, and
normals. When the image is rotated by θ, the normal vectors are rotated by
the same θ in the image plane; when the image is horizontally flipped, the
normals' x-component is negated. Without these corrections the normal
supervision becomes physically inconsistent with the input.

| Augmentation | Default |
|---|---|
| Random rotation | ±15° |
| Random scale | 0.85 – 1.15 |
| Horizontal flip | p = 0.5 |
| Brightness / contrast / saturation jitter | ±0.2 / ±0.2 / ±0.1 |
| Hue jitter | ±0.05 |
| Gaussian noise | σ = 0.01 |
| Random erasing | p = 0.25 |

### Loss functions (`utils/losses.py`)

```
L_total = DiceBCELoss(mask_logits, gt_mask)
        + w_normal * masked_cosine_loss(pred_normals, gt_normals, gt_mask)
        + w_smooth * edge_aware_normal_smoothness(pred_normals, rgb)
```

with defaults `w_normal = 5.0` and `w_smooth = 0.5`.

- **`DiceBCELoss`** combines `BCEWithLogitsLoss` (element-wise across the 7
  affordance channels) with per-channel soft Dice averaged over both batch
  and channels. Per-channel Dice keeps rare classes (e.g. `pound`) from being
  washed out by abundant ones (`grasp`). Dice is important because affordance
  pixels are heavily outnumbered by background.
- **`masked_cosine_loss`** averages the cosine distance over the union of all
  annotated affordance pixels — wherever any of the 7 classes is active, the
  predicted normal is supervised.
- **`edge_aware_normal_smoothness`** is the classic
  `exp(-|grad RGB|)`-weighted smoothness term: encourages normals to be
  smooth inside flat regions while allowing breaks where the RGB image has
  edges.

---

## 6. Workflow

### Phase 1 — Data engineering

- **Label extraction:** load `.mat` label files containing per-pixel class
  IDs 0–7. The dataset returns a (7, H, W) multi-hot tensor (one channel per
  affordance class, in `config.AFFORDANCE_CLASSES` order). Augmentation runs
  on the raw label image first (with `INTER_NEAREST` to preserve class IDs);
  the multi-hot expansion happens afterwards.
- **On-the-fly normal generation:** back-project depth into 3D via the camera
  intrinsics, then compute normals from cross products of finite-difference
  tangents (`utils/geometry.compute_normals`). The dataset shifts the
  principal point to the cropped frame so the back-projection remains
  geometrically correct after center-cropping.
- **Data loader:** `utils/dataset.py` (`UMDAffordanceDataset`) with optional
  augmentation toggle and configurable intrinsics.

### Phase 2 — Neural perception

```bash
# Train with default augmentation
python scripts/train.py --epochs 25 --batch_size 8

# Ablation: training without augmentation
python scripts/train.py --epochs 25 --batch_size 8 --no_augment

# Resume from last checkpoint
python scripts/train.py --resume --epochs 50
```

The script splits the tool set 80 / 20 by name (deterministic seed) to
enforce an instance-split evaluation — the model is tested on tools it has
never seen during training. Checkpoints are written to `checkpoints/` (or
`/content/drive/MyDrive/robotic_affordance_project/checkpoints/` when
`--use_drive` is passed for Colab).

### Phase 3 — Evaluation and synthesis

- Quantitative: best validation IoU and mean angular error on held-out tools;
  the training script logs both every epoch.
- Qualitative: run inference on the in-the-wild office captures with the
  appropriate `INFERENCE_INTRINSICS` to ground 3D geometry on unseen real
  objects.

---

## 7. Monitoring and Evaluation

Training writes every epoch's metrics to disk so the run is auditable even if
you lose the terminal output (e.g. when training on a remote machine). The
checkpoint directory after a run looks like this:

```
checkpoints/
├── best.pth            # decoder state_dict at best val loss
├── last.pth            # full checkpoint for --resume
├── history.jsonl       # one JSON line per epoch, train + val rows
└── run_config.json     # the args this run was launched with
```

### Inspecting an in-flight run

`history.jsonl` is line-buffered and flushed every epoch. From the remote PC:

```bash
tail -f checkpoints/history.jsonl              # follow live
tail -n 1 checkpoints/history.jsonl | python -m json.tool   # last record, pretty
```

### Post-hoc evaluation (`scripts/evaluate.py`)

Loads any checkpoint and writes a detailed JSON report containing:

- Mean-IoU at thresholds 0.3, 0.4, 0.5, 0.6, 0.7 (per-class IoU averaged
  across the 7 affordance channels)
- Per-affordance-class IoU at each threshold (`per_class_overall`)
- Mean angular error in degrees over the union of GT affordance pixels
- Fraction of normal-vector pixels with angular error ≤ 11.25° / 22.5° / 30°
  (the standard NYUv2 surface-normal bins)
- Per-tool breakdown of mean-IoU, per-class IoU at 0.5, and mean angular error

```bash
# Evaluate the best checkpoint on the held-out val tools (default)
python scripts/evaluate.py

# Evaluate the last checkpoint on the entire dataset
python scripts/evaluate.py --checkpoint checkpoints/last.pth --split all
```

Report written to `checkpoints/evaluation_<split>.json`.

### Training curves and overfitting check (`scripts/visualize.py`)

Reads `history.jsonl` and produces a 2×2 plot (train vs val loss, IoU,
angular error, plus train-side component losses) with the best epoch marked.
A short text summary is printed to stdout and saved next to the PNG.

```bash
python scripts/visualize.py --history checkpoints/history.jsonl
```

Outputs:

```
checkpoints/
├── training_curves.png
└── training_summary.txt
```

The summary includes a heuristic overfitting flag (train loss decreasing
while val loss flat or rising in the last three epochs) and the "patience":
how many epochs since val loss last improved. A high patience with a still
falling train loss is the canonical overfitting signature.

### Qualitative prediction grids

Pass `--checkpoint` and `--n_samples N` to also dump per-sample multi-panel
PNGs for N random val samples. Each PNG has three rows: a summary row (RGB,
GT vs predicted multi-class colored overlay, GT vs predicted normals); a row
of per-class GT heatmaps; and a row of per-class predicted heatmaps. This
makes it easy to spot which affordance channels are working and which are
not.

```bash
python scripts/visualize.py --history checkpoints/history.jsonl \
    --checkpoint checkpoints/best.pth --n_samples 8
```

Outputs land in `checkpoints/samples/`.

---

## 8. Code Structure

```text
cv-project/
│
├── data/                         # Datasets (gitignored)
│   ├── raw/part-affordance-dataset/tools/
│   └── custom_test_set/
│
├── models/
│   ├── backbone.py               # frozen DINOv2, multi-scale (4 layers)
│   └── decoder.py                # multi-scale fusion + RGB skips + logits
│
├── utils/
│   ├── dataset.py                # UMD dataset + instance_split helper
│   ├── augmentations.py          # joint RGB / mask / normal augmentation
│   ├── losses.py                 # DiceBCE, smoothness, angular error, IoU
│   ├── training_logger.py        # JSONL per-epoch metrics logger
│   ├── geometry.py               # back-projection and normal computation
│   └── visualization.py
│
├── scripts/
│   ├── train.py                  # training loop with metrics logging
│   ├── evaluate.py               # detailed metrics on any checkpoint
│   └── visualize.py              # training curves + sample prediction grids
│
├── notebooks/
│   ├── colab_training.ipynb
│   ├── local_training.ipynb
│   └── data_exploration.ipynb
│
├── docs/
│   ├── CHANGELOG.md              # history of architectural changes
│   └── Project_Definition.md
│
├── archive/                      # historical baselines kept for comparison
│   └── v1/                       # original single-scale ViT + simple decoder
│
├── config.py                     # paths, training defaults, camera intrinsics
├── requirements.txt
└── README.md
```

---

## 9. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the UMD dataset under data/raw/part-affordance-dataset/tools/

# 3. Train
python scripts/train.py --epochs 25 --batch_size 8

# 4. After (or during) training, inspect the run
python scripts/visualize.py --history checkpoints/history.jsonl
python scripts/evaluate.py  --checkpoint checkpoints/best.pth
```

---

## 10. Roadmap

Items planned for future iterations, ordered by expected impact for the
robotics-startup use case:

1. **RGB-D variant.** `dataset.py` already exposes the depth tensor via
   `return_depth=True`; add a parallel depth encoder branch in the decoder and
   A/B against RGB-only.
2. **Uncertainty head.** Either MC dropout or an evidential output, so a
   humanoid can gate execution on prediction confidence.
3. **Synthetic clutter augmentation.** Depth-aware composition of multiple
   UMD crops to address cross-object occlusion.
4. **Multi-resolution test-time inference.** Average masks at 448 and 672 to
   recover thin sub-patch structures.
5. **Partial DINOv2 unfreeze.** Fine-tune the last 2–4 ViT blocks at a
   10× lower learning rate after the decoder converges.

---

## 11. Further Reading

- `docs/ARCHITECTURE.md` — beginner-friendly step-by-step explanation of
  every layer, every tensor shape, and the reasoning behind each design
  decision. Includes a paper-style diagram (`docs/architecture_diagram.svg`)
  and an end-to-end shape reference table.
- `docs/CHANGELOG.md` — how the current architecture evolved, with the
  rationale behind each change and pointers to the archived baseline.
- `archive/v1/` — original single-scale ViT + simple decoder, preserved for
  A/B comparison and reproducibility of any earlier results.
