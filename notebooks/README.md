# Notebooks — status and usage

This folder contains four notebooks at different stages of currency. Read this
file first to know which one to open.

## Current notebooks

### `in_the_wild_inference.ipynb` — interactive inference on phone photos
Capture-iterate-inspect loop. Drop RGB images into `data/in_the_wild/`, run
the Setup cell once, then use either the single-image cell (with `INDEX = N`)
or the batch cell to process the whole folder. Wraps `scripts/predict.py`.
Use this for qualitative evaluation on novel objects.

### `data_exploration.ipynb` — dataset inspection and label diagnostics
Loads UMD samples and renders RGB / depth / mask side by side. Contains the
diagnostic cells used to surface the bowl-class supervision issue
(`bowl_02` and `bowl_03` having only `contain` annotations, not the
`grasp`+`wrap-grasp` whitelist used by the binary baseline). Read together
with `docs/RESULTS.md` §1 and `docs/CHANGELOG.md` for context.

## Historical notebooks (left in place for reference)

### `local_training.ipynb`
Captures the earlier training workflow before `scripts/train.py` was
finalised as the entry point. May reference an older API (single-channel
mask head, no class weights). For current training, use:
```bash
python scripts/train.py --epochs 40
```

### `colab_training.ipynb`
Earlier Colab adapter for the v1 baseline. Predates the v2 architecture
(multi-scale fusion + RGB skips) and the multi-class refactor. Kept only
for reproducibility of the original baseline results.

## How to add a new notebook

If you build a new notebook for some experiment, add a one-paragraph
description here so it doesn't get orphaned. Notebooks that hard-code paths
or reference deprecated APIs should be noted in the historical section
rather than deleted, since they document the project's evolution.
