# Usage Guide

How to run every entry point in the project. Companion to `RESULTS.md`
(what was done), `ARCHITECTURE.md` (how the model is built), and
`CHANGELOG.md` (how the code evolved).

## 1. Installation

```bash
pip install -r requirements.txt
```

Place the UMD Part Affordance Dataset (single-object split) under
`data/raw/part-affordance-dataset/tools/`. Each tool subdirectory should
contain `*_rgb.jpg`, `*_depth.png`, and `*_label.mat` triplets.

For the clutter split (not yet evaluated), drop the tarball into
`data/raw/` and extract under `data/raw/part-affordance-clutter/`.

## 2. Training

```bash
# Default: 25 epochs, class-weighted loss ON, joint augmentation ON.
python scripts/train.py --epochs 25 --batch_size 8

# Without class weights (ablation comparison).
python scripts/train.py --epochs 25 --no-class_weights

# Without augmentation (ablation comparison).
python scripts/train.py --epochs 25 --no_augment

# Resume from last.pth for additional epochs.
python scripts/train.py --resume --epochs 40

# Train on Google Drive checkpoints (Colab variant).
python scripts/train.py --use_drive --epochs 25
```

Outputs land in `checkpoints/`:

| File | Description |
|---|---|
| `best.pth` | Decoder state dict at best val loss. Use for evaluation and inference. |
| `last.pth` | Full checkpoint for `--resume` (model + optimiser + epoch). |
| `history.jsonl` | Per-epoch metrics, line-buffered, survives crashes. |
| `run_config.json` | The args this run was launched with, plus class order. |
| `class_pixel_counts.json` | Cached per-class pixel scan (skipped on subsequent runs). |

The script logs both standard metrics (loss, mean-IoU, angular error) and
per-class IoU per epoch, so `scripts/visualize.py` can produce per-class
trajectory plots.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 25 | Target number of epochs. |
| `--batch_size` | 8 | Per-GPU batch size. |
| `--lr` | 1e-4 | AdamW learning rate. |
| `--w_normal` | 5.0 | Weight on `masked_cosine_loss` for normals. |
| `--w_smooth` | 0.5 | Weight on edge-aware normal smoothness. |
| `--no_augment` | off | Disable joint RGB/mask/normal augmentation. |
| `--class_weights` | **ON** | Per-channel `pos_weight` from frequency-inverse scan. |
| `--no-class_weights` | off | Opt out for the unweighted ablation. |
| `--weight_power` | 0.5 | `pos_weight = (N_neg/N_pos) ** weight_power`. |
| `--weight_clip` | 15.0 | Cap on per-class `pos_weight`. |
| `--resume` | off | Load `checkpoints/last.pth` and continue. |
| `--use_drive` | off | Write checkpoints to `/content/drive/MyDrive/...` (Colab). |

## 3. Evaluation

```bash
# Default: best.pth on the held-out val tools.
python scripts/evaluate.py

# Specific checkpoint, full dataset.
python scripts/evaluate.py --checkpoint checkpoints/last.pth --split all

# Write the JSON elsewhere.
python scripts/evaluate.py --output_dir reports/some_run
```

Produces `checkpoints/evaluation_<split>.json` with:

- IoU at thresholds 0.3 / 0.4 / 0.5 / 0.6 / 0.7 (overall, per-tool, per-class).
- Mean angular error in degrees over the union of all affordance pixels.
- NYUv2 angular bins (fraction ≤ 11.25° / 22.5° / 30°).
- `per_class_overall` and per-tool `iou@0.5_per_class` breakdowns.

A console summary prints the overall metrics plus per-class IoU @ 0.5.

## 4. Training Curves and Sample Grids

```bash
# Curves only.
python scripts/visualize.py --history checkpoints/history.jsonl

# Curves + N sample prediction grids.
python scripts/visualize.py --history checkpoints/history.jsonl \
    --checkpoint checkpoints/best.pth --n_samples 12

# Choose where outputs go.
python scripts/visualize.py --history checkpoints/history.jsonl \
    --output_dir reports/some_run
```

Outputs:

- `training_curves.png` — 2×2 grid (loss, mean-IoU, angular error,
  component-loss breakdown) with the best epoch marked.
- `training_summary.txt` — text summary including the heuristic overfitting
  flag and the patience (epochs since best val loss).
- `samples/<idx>_<tool>.png` — one 3-row figure per sampled val item:
  RGB + GT/Pred multi-class overlay + GT/Pred normals on the top row, then
  a row of per-class GT heatmaps and a row of per-class predicted heatmaps.

## 5. In-the-Wild Inference

```bash
# Folder of RGB images, no GT needed.
python scripts/predict.py --input_dir data/in_the_wild

# Different threshold for the argmax overlay.
python scripts/predict.py --input_dir data/in_the_wild --thresh 0.7

# Custom output location.
python scripts/predict.py --input_dir data/in_the_wild \
    --output_dir reports/predictions/kitchen_test_$(date +%Y-%m-%d)
```

Inputs are center-cropped to a square and resized to 448². Phone photos
(JPEG/PNG) work directly. HEIC files are *not* supported — convert to JPG
via Preview before running.

Each output PNG is a 2-row grid: RGB + predicted multi-class overlay +
predicted normals on the top row; the 7 per-class probability heatmaps on
the bottom row.

For interactive use (single-image inspection, capture-iterate loops), open
`notebooks/in_the_wild_inference.ipynb` and use the cells for "predict
single" or "predict folder."

## 6. Comparison Plot

```bash
python scripts/plot_comparison.py \
    --baseline archive/v2/checkpoints_binary/evaluation_val.json \
    --new      checkpoints/evaluation_val.json \
    --baseline_label "Binary (grasp + wrap-grasp)" \
    --new_label      "Multi-class + class weights" \
    --output reports/comparisons/binary_vs_multiclass.png
```

Produces a grouped horizontal bar chart per tool plus an overall summary
panel. Tools are sorted by delta (biggest improvement first). The script
also prints a per-tool delta table to stdout.

## 7. Archiving a Completed Run

```bash
bash scripts/archive_run.sh <descriptor>
```

Snapshots `checkpoints/` into `runs/<descriptor>_<YYYY-MM>/`. After
archiving, manually:

1. Append a row to `runs/INDEX.md` describing the run.
2. If the result is notable, add a subsection to `docs/RESULTS.md`.
3. Add a one-paragraph entry to `docs/CHANGELOG.md`.

The active `checkpoints/` directory is unchanged — archiving is a copy,
not a move. This keeps `scripts/train.py --resume` and all evaluation
commands working on the live run.

## 8. Notebook Workflows

| Notebook | Purpose |
|---|---|
| `notebooks/data_exploration.ipynb` | Inspect UMD samples, diagnose label issues, visualise raw class IDs. |
| `notebooks/in_the_wild_inference.ipynb` | Interactive inference on phone photos with per-image inspection. |
| `notebooks/local_training.ipynb` | Historical training notebook from before `scripts/train.py` was canonical. May reference a deprecated API. |
| `notebooks/colab_training.ipynb` | Historical Colab adapter for the v1 baseline. Predates the v2 architecture. |

See `notebooks/README.md` for the full status.

## 9. Monitoring an In-Flight Training Run

`history.jsonl` is line-buffered and flushed every epoch. From a remote
machine:

```bash
tail -f checkpoints/history.jsonl              # follow live
tail -n 1 checkpoints/history.jsonl | python -m json.tool   # last record, pretty
```

The training-curves regenerator picks up partial runs cleanly; you can run
`scripts/visualize.py --history checkpoints/history.jsonl` mid-training to
inspect progress without disturbing the run.
