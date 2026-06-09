# Runs Index

This directory is the canonical archive of completed training runs. Each entry
preserves the model weights, the per-epoch training history, the evaluation
report, the run configuration, and a snapshot of qualitative samples. The
goal is that any quantitative claim in `docs/RESULTS.md` is traceable to a
specific subdirectory here.

## Conventions

- One subdirectory per completed run, named `<descriptor>_<YYYY-MM>/`. Use
  short slugged descriptors (e.g. `binary_baseline`, `multiclass_unweighted`,
  `multiclass_weighted`) plus a trailing year-month for chronology.
- Each run directory mirrors the structure of `checkpoints/` at the time of
  archiving (best.pth, last.pth, history.jsonl, evaluation_val.json,
  run_config.json, samples/, training_curves.png, training_summary.txt,
  and class_pixel_counts.json if class weights were used).
- The active working directory `checkpoints/` is *not* an archive — it is
  whatever run is currently in flight or most recently saved. Snapshot it
  into `runs/` with `scripts/archive_run.sh` when a run is finalised.
- Large weights (`*.pth`) are gitignored. The metadata files (JSON, JSONL,
  PNG, MD) are tracked because they are small and they document the run.

## Active run

The live working directory is at the repo root:

```
checkpoints/
```

Contains the most recently saved model state and per-epoch history. Use
`scripts/archive_run.sh <name>` to snapshot it into this directory.

## Archived runs

| Directory | Architecture | Loss | Best epoch | Val mean-IoU @ 0.5 | Notes |
|---|---|---|---|---|---|
| `../archive/v2/checkpoints_binary/` | v2 multi-scale + RGB skips | BCE + Dice, single channel | 24 of 25 | **0.7128** (2-class) | Binary baseline. Supervised on union of UMD classes 1 (`grasp`) and 7 (`wrap-grasp`). Bowls and turners undersupervised. Preserved in `archive/v2/` rather than `runs/` because the source code at the time still emitted single-channel masks. |
| `multiclass_weighted_2026-06/` *(to be populated)* | v2 multi-scale + RGB skips | BCE (per-channel pos_weight) + Dice (per-channel) | 15 of 40 | **0.7697** (7-class mean) | Final model. Frequency-inverse class weights with `weight_power=0.5`, `weight_clip=15.0`. Run across 3 sessions, confirmed converged at epoch 15 (no improvement through epoch 40). Currently lives in `checkpoints/`. |

To populate the `multiclass_weighted_2026-06/` row, run:

```bash
bash scripts/archive_run.sh multiclass_weighted
```

(see the script for details — adds the date suffix automatically).

## Adding a new entry

After running `scripts/archive_run.sh <descriptor>`:

1. Open the new `runs/<descriptor>_YYYY-MM/` directory and verify the files
   are all there.
2. Append a row to the "Archived runs" table above with the four key fields:
   architecture, loss, best epoch, val mean-IoU.
3. If the run produced a notable finding, add a numbered subsection to the
   appropriate part of `docs/RESULTS.md` referencing this run's directory.
4. Update `docs/CHANGELOG.md` with a one-paragraph dated entry summarising
   what changed and why.

## Why not just keep everything in `checkpoints/`?

`checkpoints/` is mutable — it's overwritten by every training resume and
every new run. The `runs/` directory is *append-only* by convention. This
matters because months later, the answer to "what config produced result X?"
needs to point to an immutable snapshot, not a directory that may have been
overwritten three times since.
