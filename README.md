# Geometric–Semantic Fusion for Autonomous Robotic Affordance

A multi-task perception model that predicts pixel-precise **affordance masks**
(where on an object you can grasp, cut, pour into, etc.) and **dense surface
normals** (which way the surface is facing) from a single RGB image. Designed
as the perception primitive for a humanoid robotics stack: tells the robot
*what part of an object to interact with* and *how to approach it*, without
3D reconstruction.

This repository is the final project for a Computer Vision course and the
seed of a perception module for a humanoid robotics startup.

---

## What it does

Given an RGB image, the model produces:

- A **7-channel multi-label affordance mask** at 448×448 — one independent
  sigmoid channel per UMD affordance class (`grasp`, `cut`, `scoop`,
  `contain`, `pound`, `support`, `wrap-grasp`). A single pixel can belong to
  multiple affordances simultaneously (a flat surface can be both `support`
  and `scoop`).
- A **3-channel surface normal map** at 448×448 — unit-vector field for the
  visible geometry, used by the robot's grasp planner to choose approach
  direction.

Combined with depth (at inference time only, not required for the network),
these outputs yield a complete robotic-grasp packet: target centroid (X, Y, Z)
and approach vector (N_x, N_y, N_z).

## Approach in one paragraph

A frozen **DINOv2 ViT-Small** backbone extracts multi-scale semantic features
from four intermediate transformer layers. A **DPT-style multi-scale fusion
decoder** projects and concatenates those features at 32×32, then
progressively upsamples to 448×448 with **RGB skip connections** from a
small trainable CNN stem — the skips supply the high-frequency spatial
detail the ViT alone cannot reconstruct. Two separate heads then emit the
mask logits and the normal vectors. Training uses **per-channel BCE +
soft Dice loss** with **frequency-inverse class weights** to address heavy
class imbalance, **masked cosine loss** for normals over annotated affordance
pixels, and **edge-aware smoothness** to suppress normal jitter in flat
regions.

Full architectural detail with paper-style diagram in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Headline result

Trained on the UMD Part Affordance Dataset (single-object split, 21 tool
families, instance-split val of 5791 samples):

| Metric | Value |
|---|---|
| Mean-IoU @ 0.5 (val) | **0.7697** |
| Mean angular error (val) | 24.26° |
| Tool families with any prediction | 21 / 21 |
| Improvement over binary baseline | +0.057 on a strictly harder task |

The class-weighted multi-class model recovers tool families that were
**invisible** to the binary baseline: bowls went from 0% → 92–98% IoU, the
turner from 11% → 51%. See [`docs/RESULTS.md`](docs/RESULTS.md) for the full
per-class, per-tool, and qualitative breakdown.

## Repository structure

```
cv-project/
│
├── README.md             # this file — high-level overview
├── requirements.txt
├── config.py             # paths, training defaults, camera intrinsics
│
├── docs/                 # all documentation
│   ├── RESULTS.md        # canonical results: numbers, findings, limitations
│   ├── ARCHITECTURE.md   # design, layer shapes, paper-style diagram
│   ├── USAGE.md          # how to run every entry point
│   ├── CHANGELOG.md      # chronological code evolution
│   ├── FUTURE_DEVELOPMENT.md  # phased roadmap
│   └── Project_Definition.md
│
├── models/               # network components
│   ├── backbone.py       #   frozen DINOv2, multi-scale 4-layer tap
│   └── decoder.py        #   multi-scale fusion + RGB skips + heads
│
├── utils/                # data, loss, augmentation, geometry
│   ├── dataset.py        #   UMD dataset, instance split, class-pixel scan
│   ├── augmentations.py  #   joint RGB+mask+normal augmentation
│   ├── losses.py         #   DiceBCE + cosine + smoothness + IoU
│   ├── geometry.py       #   depth → normals via finite differences
│   ├── training_logger.py
│   └── visualization.py
│
├── scripts/              # entry points
│   ├── train.py          #   training with per-class loss weights
│   ├── evaluate.py       #   detailed per-class / per-tool metrics
│   ├── visualize.py      #   training curves + sample grids
│   ├── predict.py        #   inference on phone photos
│   ├── plot_comparison.py #  binary-vs-final comparison figure
│   └── archive_run.sh    #   snapshot checkpoints/ into runs/
│
├── notebooks/            # interactive exploration
│   ├── README.md         #   notebook status guide
│   ├── data_exploration.ipynb
│   ├── in_the_wild_inference.ipynb
│   ├── local_training.ipynb    (historical)
│   └── colab_training.ipynb    (historical)
│
├── data/                 # datasets (gitignored)
│   └── raw/part-affordance-dataset/
│
├── checkpoints/          # active working run (best.pth, history, etc.)
│
├── runs/                 # archived completed runs (append-only)
│   └── INDEX.md          #   table of all archived runs
│
├── reports/              # generated figures and predictions (gitignored)
│   ├── comparisons/      #   plot_comparison.py output
│   ├── predictions/      #   predict.py per-batch output
│   └── qualitative/      #   cherry-picked images for slides
│
└── archive/              # historical code and checkpoints
    └── v1/               #   original single-scale ViT + simple decoder
```

## Quick start

```bash
# Install
pip install -r requirements.txt

# Place the UMD dataset under data/raw/part-affordance-dataset/tools/

# Train (45 minutes per epoch on a 12 GB GPU, ~7 hours for the full 25)
python scripts/train.py --epochs 25 --batch_size 8

# Inspect the run
python scripts/visualize.py --history checkpoints/history.jsonl
python scripts/evaluate.py  --checkpoint checkpoints/best.pth

# Try it on phone photos
python scripts/predict.py --input_dir data/in_the_wild
```

Detailed CLI reference for every script and flag in
[`docs/USAGE.md`](docs/USAGE.md).

## Where to read next

- **What was achieved**: [`docs/RESULTS.md`](docs/RESULTS.md) — final numbers,
  per-class breakdown, in-the-wild findings, known limitations.
- **How the model works**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —
  layer-by-layer walkthrough with diagram.
- **How to run things**: [`docs/USAGE.md`](docs/USAGE.md) — every CLI entry
  point with examples.
- **How the code evolved**: [`docs/CHANGELOG.md`](docs/CHANGELOG.md) —
  three documented architectural changes (v1 → v2 multi-scale → multi-class
  → class-weighted).
- **What comes next**: [`docs/FUTURE_DEVELOPMENT.md`](docs/FUTURE_DEVELOPMENT.md) —
  five-phase roadmap from low-cost wins to the long-term VLA integration.

## Key design decisions and their rationale

| Decision | Why |
|---|---|
| Frozen DINOv2 backbone | Stable training, fast convergence, no language-specific fine-tuning needed. The decoder learns where the affordances are; DINOv2 brings the "what" features for free. |
| Multi-label sigmoid (not multi-class softmax) | A single pixel can serve multiple affordances (a trowel face is both `support` and `scoop`). Softmax would force a false exclusivity. |
| Per-channel Dice averaged across channels | Rare classes (`pound`, `support`) contribute equally to the loss. Without this, gradients are dominated by common classes and rare ones never learn. |
| Frequency-inverse `pos_weight` per channel | Empirically lifts overall mean-IoU from 0.713 to 0.770. The biggest gains are on the previously-weakest classes (`support`: 0.10 → 0.53). |
| Joint augmentation with normal-vector rotation | Without rotating the normal vectors when rotating the image, normal supervision becomes physically inconsistent — geometrically wrong but visually plausible. |
| Logits + `BCEWithLogitsLoss` (not sigmoid + `BCELoss`) | Numerical stability. The original v1 baseline shipped with the unstable form; correcting it was one of the v2 changes. |
| Instance split for evaluation | Tools at training time are *never* seen at validation — measures true cross-instance generalisation, not memorisation. |

## Status

The model is **converged and frozen** for the project deadline. Current
limitations and the prioritised list of next steps are in
[`docs/FUTURE_DEVELOPMENT.md`](docs/FUTURE_DEVELOPMENT.md). The three
highest-leverage next items are: per-class inference thresholds (clean up
in-the-wild deployment), UMD clutter-split zero-shot evaluation
(quantify the in-the-wild observations), and a SAM2-style object detection
front-end for cluttered scenes (the architectural bridge to humanoid
deployment).

## License and acknowledgements

Built on the [UMD Part Affordance Dataset](http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/)
(Myers et al., 2015) and Meta AI's [DINOv2](https://github.com/facebookresearch/dinov2)
self-supervised ViT.
