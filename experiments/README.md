# ECCV workshop experiment runs

One self-contained, **resumable** training script per experiment for the ECCV 2026
workshop paper (deadline July 16). Each writes to its own folder under
`runs_eccv/` so runs never collide and each can be resumed independently — ideal
for an intermittently-available GPU. Re-running a script picks up from that run's
`last.pth` instead of restarting.

Run from anywhere (each script `cd`s to the project root):

```bash
bash experiments/01_main_headline.sh
```

## Run queue (priority order)

| # | Script | Output dir | Purpose | Needed for |
|---|--------|-----------|---------|-----------|
| 1 | `01_main_headline.sh` | `runs_eccv/main` | Multi-task affordance + normals, canonical split, fixes + class weights. **The headline number.** | Mandatory |
| 2 | `02_deeplab_baseline.sh` | `runs_eccv/baseline_deeplab` | DeepLabv3-ResNet50 control on the same split/metric. Validates the F-measure (≈0.733) and gives the apples-to-apples row. | Mandatory |
| 3 | `03_ablation_no_normals.sh` | `runs_eccv/ablation_no_normals` | Multi-task value: masks **without** normal supervision (`--w_normal 0`). | Strong |
| 4 | `04_ablation_no_class_weights.sh` | `runs_eccv/ablation_no_class_weights` | Class-weighting effect (`--no-class_weights`). | Strong |

If GPU time runs short, **1 + 2 plus the no-training assets** (the already-done
binary→multiclass→class-weighted progression in `docs/RESULTS.md`, the clutter-split
eval, and the in-the-wild study) are enough for a submittable paper. Runs 3–4 are
the clean on-canonical-split ablation table; ViT-B and RGB-skip ablations are
deferred to the full paper (heavy / need code changes).

## What each run produces

In its output dir: `best.pth` (selected on val mean-IoU), `best_wfb.pth` (selected
on val weighted F-measure — the metric the paper reports), `last.pth` (resume
checkpoint), `history.jsonl`, `run_config.json`, `split_novel_instance.json`, and
`evaluation_val.json` (for runs 1/3/4) or `evaluation_baseline.json` (run 2).

The **weighted F-measure $F_\beta^\omega$ ($\beta^2{=}0.3$)** in those JSONs is the
number to compare against AffordanceNet Table II (their average 0.799; DeepLab
0.733; ED-RGB 0.766). Report $F_\beta^\omega$, not IoU.

## Mapping to paper tables

- **Comparison table:** run 1 ($F_\beta^\omega$) vs run 2 (your DeepLab) vs cited
  AffordanceNet / ED-RGB numbers.
- **Ablation table:** run 1 vs run 3 (no normals) vs run 4 (no class weights),
  plus the binary/multiclass rows from `RESULTS.md`.

## Notes

- Run 2 uses `--batch_size 4` (ResNet-50 full fine-tune is memory-heavy on 12 GB);
  bump to 8 if it fits.
- `runs_eccv/` holds large `.pth` files — consider adding it to `.gitignore`.
- Archive a finished run into `runs/` with `bash scripts/archive_run.sh <name>`
  if you want it in the curated history.
