# Archived v1 baseline

This directory is a frozen snapshot of the original affordance pipeline. It is
kept for reproducibility of any earlier results and as a reference for the
architectural decisions described in `docs/CHANGELOG.md`. **Nothing in the
active codebase imports from here.**

## Contents

```
archive/v1/
├── models/
│   ├── backbone.py     # frozen DINOv2 ViT-Small, last-layer features only
│   └── decoder.py      # 3-stage bilinear upsample, no skip connections,
│                       #   sigmoid applied inside the network
├── utils/
│   └── dataset.py      # UMD dataset with hardcoded Kinect-v1 intrinsics,
│                       #   no augmentation, no principal-point correction
└── scripts/
    └── 01_train.py     # original training loop: Adam + nn.BCELoss + cosine
```

## Differences vs the current architecture

See `docs/CHANGELOG.md` for the full comparison table and the rationale behind
each change.

## Running this baseline

The active project does not invoke any of these files. To reproduce baseline
numbers you can either:

1. Copy the four files back into the working tree at their original locations
   (`models/backbone.py`, `models/decoder.py`, `utils/dataset.py`,
   `scripts/01_train.py`), overwriting the current versions; or
2. Add `archive/v1` to `sys.path` ahead of the working tree and import from
   the same module names.

Option (1) loses the current architecture until you restore it from git;
option (2) keeps both available but requires careful import management. In
practice the changelog table answers most questions without needing to run
the baseline again.
