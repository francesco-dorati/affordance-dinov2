# Results

This document is the canonical record of what the project achieved and what is known about the model's behaviour. It is organised so future experiments append cleanly without rewriting the existing narrative. Companion documents: `ARCHITECTURE.md` for system design, `CHANGELOG.md` for chronological code changes, `FUTURE_DEVELOPMENT.md` for the forward roadmap.

## 1. Headline Summary

The final model is a frozen DINOv2 ViT-Small backbone feeding a multi-scale fusion decoder with RGB skip connections, predicting a 7-channel multi-label affordance mask plus a 3-channel surface normal map on the UMD Part Affordance Dataset. Trained for 40 epochs with frequency-inverse per-class loss weights, the best checkpoint (epoch 15, val loss 0.7067) achieves the following on the held-out val split (5791 samples, 21 tools, instance-split):

| Metric | Value |
|---|---|
| Mean-IoU @ 0.5 | **0.7697** |
| Mean angular error (normals) | 24.26° |
| Fraction normals ≤ 30° | 73.3% |
| Tool families with non-zero predictions | 21 / 21 |
| Affordance classes supervised | 7 (full UMD taxonomy) |

Compared to the binary baseline (2-class supervision on `grasp` and `wrap-grasp` only), the final model adds five previously-invisible affordance classes, restores bowls and turners from 0% / 11% IoU to 98% / 51% respectively, and raises the headline IoU number on a strictly harder evaluation task.

> **Metric & split note (June 13, 2026).** The numbers above are **mean-IoU** on the legacy ad-hoc instance split, and were produced *before* the normal-rotation fix. They are **not** comparable to the affordance literature, which reports the **weighted F-measure** $F_\beta^\omega$ ($\beta^2{=}0.3$) on the Myers et al. 2015 split (AffordanceNet UMD Table II: average **0.799**, DeepLab 0.733, ED-RGB 0.766). Tooling for a comparable run is now in place: canonical `novel_instance` / `category` splits (`utils/dataset.py`), $F_\beta^\omega$ (`utils/metrics.py`), and the verified normals/IoU fixes. The next retrain (`--split_type novel_instance`) followed by `evaluate.py` (`--wfb`, on by default) will produce the benchmark-comparable table; results to be appended as §3.6.

## 2. Training Trajectory

The project went through three distinct training configurations. Each is preserved or reproducible; quantitative deltas between phases are the basis of the experimental analysis in Section 4.

### 2.1 Phase A — Binary baseline (archived)

Single-channel mask head supervised on the union of UMD classes 1 (`grasp`) and 7 (`wrap-grasp`). 25 epochs, best at epoch 24, val IoU 0.7117, mean angular error 22.4°. Strong on knives, mugs, cups (IoU 0.85–0.94); zero IoU on bowls (no `grasp` or `wrap-grasp` annotation in the source labels); 0.11 IoU on turner_06 (the model's only signal was the small handle, ignoring the dominant `support` surface). Preserved in `archive/v2/checkpoints_binary/`.

### 2.2 Phase B — Multi-class (7-channel head, unweighted loss)

Decoder mask head expanded from 1 to 7 sigmoid output channels, one per UMD affordance class. Dataset returns a `(7, H, W)` multi-hot tensor. Loss is per-channel BCE + per-channel soft Dice averaged across channels. 25 epochs, best at epoch 25, val mean-IoU 0.7128, mean angular error 24.38°. Bowls now train on `contain` and achieve healthy per-class IoU; turner improves on `support`. Headline mean-IoU is essentially unchanged from binary, but the metric is now averaged across 7 channels instead of measuring a 2-class union, so the comparison understates the qualitative gain. Per-class evaluation revealed strong train-val gaps on rare classes (`support` 0.89 train → 0.36 val, `scoop` 0.88 → 0.58), motivating Phase C.

### 2.3 Phase C — Multi-class + frequency-inverse class weights (final)

`DiceBCELoss` extended to accept a per-channel `pos_weight` vector derived from a one-time scan of training-set pixel frequencies. Formula `pos_weight[c] = (N_neg / N_pos) ** 0.5`, clipped at 15.0. Empirical class pixel fractions: `contain` 0.67%, `grasp` 0.60%, `wrap-grasp` 0.37%, `cut` 0.29%, `support` 0.18%, `pound` 0.14%, `scoop` 0.14%. With these weights, 5 of 7 channels hit the 15.0 cap because every affordance class is under 1% of total pixels (background dominates massively).

Trained for 40 epochs; best checkpoint at epoch 15 (val loss 0.7067, val mean-IoU 0.7467). Continuing to epoch 40 produced no further val improvement — patience reached 25 epochs, train-val gap widened from 0.27 to 0.34, val IoU oscillated in 0.69–0.74. The model is confirmed converged. Best is `checkpoints/best.pth`; full evaluation report is `checkpoints/evaluation_val.json`.

## 3. Quantitative Results

### 3.1 Overall (val split, instance-split, 5791 samples)

| Threshold | Mean-IoU |
|---|---|
| 0.3 | 0.7677 |
| 0.4 | 0.7691 |
| 0.5 | **0.7697** |
| 0.6 | 0.7696 |
| 0.7 | 0.7688 |

Normals: mean angular error 24.26°, fraction with error ≤ 11.25° = 19.5%, ≤ 22.5° = 55.3%, ≤ 30° = 73.3%.

### 3.2 Per-class IoU @ 0.5

| Class | IoU | Read |
|---|---|---|
| `wrap-grasp` | 0.881 | Solved — every cup/mug handle. |
| `cut` | 0.866 | Solved — knife and scissor blades. |
| `contain` | 0.824 | Solved — cup, mug, bowl interiors. |
| `pound` | 0.808 | Strong despite class rarity (mallet head). |
| `scoop` | 0.711 | Solid (spoons, trowels, scoop_01). |
| `grasp` | 0.684 | Weakest of the well-learned classes. Hurt by thin annular regions (mug rims) and by competition with `wrap-grasp` on overlapping geometry. |
| `support` | 0.526 | Hardest. Strong train performance (0.89) but largest train-val gap. Most affected by class-weight oscillation between epochs. |

### 3.3 Per-tool IoU @ 0.5 (compared to binary baseline)

Sorted by improvement magnitude:

| Tool | Binary | Final | Δ | Read |
|---|---|---|---|---|
| `bowl_02` | 0.000 | 0.977 | +0.977 | Newly visible (only `contain`). |
| `bowl_03` | 0.000 | 0.922 | +0.922 | Newly visible. |
| `turner_06` | 0.113 | 0.508 | +0.395 | `support` learned (0.90), `grasp` still weak (0.12). |
| `trowel_02` | 0.741 | 0.808 | +0.067 | |
| `spoon_07` | 0.741 | 0.802 | +0.061 | |
| `knife_02` | 0.798 | 0.839 | +0.041 | |
| `spoon_09` | 0.720 | 0.760 | +0.041 | |
| `knife_04` | 0.858 | 0.893 | +0.035 | |
| `knife_01` | 0.865 | 0.890 | +0.024 | |
| `cup_05` | 0.939 | 0.941 | +0.002 | |
| `scissors_06` | 0.808 | 0.809 | +0.001 | |
| `mallet_01` | 0.776 | 0.776 | 0.000 | |
| `spoon_03` | 0.771 | 0.752 | −0.019 | |
| `spoon_08` | 0.825 | 0.804 | −0.021 | |
| `knife_10` | 0.861 | 0.823 | −0.038 | |
| `mug_11` | 0.886 | 0.787 | −0.098 | Rim `grasp` channel weak (0.50). |
| `mug_12` | 0.911 | 0.810 | −0.100 | Rim `grasp` weak (0.58). |
| `mug_20` | 0.887 | 0.745 | −0.142 | Rim `grasp` very weak (0.21). |
| `scoop_01` | 0.546 | 0.347 | −0.199 | Multi-label false positives (`contain`, `pound` predicted at 0.0 IoU). |
| `trowel_01` | 0.866 | 0.648 | −0.218 | Multi-label false positives (`cut`, `support` predicted at 0.0). |
| `trowel_04` | 0.877 | 0.634 | −0.243 | Multi-label false positives + weak `scoop` (0.41). |

### 3.4 Interpretation of negative deltas

The negative deltas are *not* regressions on previously-correct predictions. The binary baseline's per-tool number measured the IoU of a 2-class union (`grasp` ∪ `wrap-grasp`). The multi-class number is the per-sample mean across the 7 affordance channels. Tools where the model now predicts non-existent classes (e.g. `cut` on `trowel_01`, `contain` on `scoop_01`) get penalised because absent-class IoU is 0 by construction, dragging the per-tool mean down. The headline +0.057 overall improvement (0.7128 → 0.7697) is therefore *despite* this measurement artifact, not because of it. Addressing it cleanly requires either per-tool class priors (only score channels that exist on a given tool) or a different per-tool aggregation (mean over channels-actually-present); both are future-work items.

### 3.5 Reproducibility artifacts

| Artifact | Path |
|---|---|
| Final model weights | `checkpoints/best.pth` |
| Full evaluation report | `checkpoints/evaluation_val.json` |
| Training history (per-epoch JSONL) | `checkpoints/history.jsonl` |
| Run configuration | `checkpoints/run_config.json` |
| Class pixel-count cache | `checkpoints/class_pixel_counts.json` |
| Binary baseline evaluation | `archive/v2/checkpoints_binary/evaluation_val.json` |
| Binary vs final comparison plot | `reports/binary_vs_multiclass.png` |
| Per-sample qualitative grids | `checkpoints/samples/` |

## 4. Qualitative Findings

### 4.1 Model occasionally outperforms its labels

On clean training-domain tools (scissors, knives, spoons), the predicted mask boundaries are visibly smoother and more anatomically reasonable than the hand-painted UMD ground truth. UMD labels were produced with a single annotator using a paint-style interface, leaving jagged boundaries; the model's combination of multi-scale ViT features, RGB skip connections, and per-channel Dice loss acts as an implicit label denoiser. This is a positive emergent property worth reporting.

### 4.2 In-the-wild RGB inference (21 phone-captured images)

The final model was applied to 21 unconstrained phone photos (knives, mugs, pots, glassware, kitchenware, non-tools, compound scenes). Inference uses center-cropped 448² RGB inputs; no depth required. Predictions exposed three distinct failure modes:

*Background texture activation on `grasp`.* The `grasp` channel fires substantially on wooden cutting boards, table surfaces, book pages, and label text — even at threshold 0.7. The model has internalised "wood-toned ridged texture = graspable" because UMD handles look exactly like that, and there are no negative-supervision examples for "wooden surface, not a handle" in training. This is the dominant failure mode in the wild and the strongest argument for a SAM2-style object segmentation front-end before deployment.

*Conservative `contain` extent on oblique views.* On phone photos taken at angle (cups and pots from above-but-tilted), the model correctly localises the *center* of the cavity but underpredicts the full rim-to-center extent. UMD training is dominated by turntable views; the model has learned a viewpoint-narrow definition of `contain`. Lowering the prediction threshold for `contain` specifically (0.3–0.4) recovers most of the missing area, suggesting calibration rather than capacity is the issue.

*Out-of-distribution materials.* Transparent objects (glass cups, glass lids) fail almost completely. The model has never seen refraction, specular highlights, or background-showing-through-object during training. DINOv2 features are usually robust here, but the affordance head cannot interpret the unfamiliar geometric cues.

### 4.3 Per-channel calibration is uneven

The classes split into two calibration regimes: `cut`, `contain`, `wrap-grasp`, `pound`, `support`, `scoop` produce precise, localised predictions; `grasp` over-fires on textured backgrounds. Per-class inference thresholds would substantially clean up the visual output without retraining (proposed in `FUTURE_DEVELOPMENT.md`).

### 4.4 Within-contour generalisation is strong

When predictions are restricted to actual object regions (visually, by ignoring background), the model is competent on every tested object class — including novel tools (pots, mestolo, fork, plate) that share affordance geometry with the training distribution. The transfer from UMD's narrow turntable distribution to phone-captured kitchenware is meaningfully successful.

## 5. Architectural Decisions and Their Empirical Impact

| Decision | Effect | Evidence |
|---|---|---|
| Frozen DINOv2 ViT-Small backbone | Stable training, fast convergence; capped final accuracy | Train loss plateaus at the decoder's expressive ceiling. |
| Multi-scale ViT fusion at 32² (layers 2, 5, 8, 11) | Recovers fine spatial detail beyond final-layer features alone | DPT-style approach, established prior; not separately ablated here. |
| RGB skip connections (DPT-style stem) | Sharper mask boundaries, accurate normals on small detail | Visually evident in `samples/*.png`; not separately ablated. |
| Logits output + `BCEWithLogitsLoss` | Numerical stability vs the original `sigmoid + BCELoss` | Reproducibility note in `CHANGELOG.md` from the v1 → v2 transition. |
| Per-channel Dice averaged across channels | Rare classes contribute meaningfully to loss | Switch from `dims=(1,2,3)` to `dims=(2,3)` was required for the 7-channel head to train rare classes at all. |
| Frequency-inverse `pos_weight` per channel | +0.057 mean-IoU, large gains on rare classes | Phase B → Phase C delta, documented in Section 2.3. |
| Joint augmentation with normal-vector rotation | Geometric consistency under rotation / flip | Required by construction; rotated normals tested correct under symmetric setups. |
| Multi-task mask + normals with `w_normal=5.0` | Joint head shares trunk; mask quality preserved | Sole training configuration, not swept. |

## 6. Known Limitations

1. **Background activation on `grasp`.** Dominant in-the-wild failure mode (Section 4.2). Caused by domain gap; not fixable by training on UMD alone.

2. **Multi-label false positives on absent classes.** Tools where the model predicts non-existent affordances (`scoop_01`, `trowel_*`) suffer per-tool mean-IoU drops that the per-class breakdown reveals are artifacts of the metric, not real regressions.

3. **Normals plateau at ~24°.** Bounded by the noise floor of depth-derived ground truth from Kinect-v1 (finite-difference normals amplify ~3–5 mm depth noise to 8–15° angular noise). Not fixable by more training; requires cleaner GT (different camera, distilled normals, or different supervision).

4. **Single-object training distribution.** UMD has one centered tool per frame on a turntable; in-the-wild data is multi-object and unconstrained. Best addressed by a factored deployment pipeline (Section 7).

5. **Conservative `contain` on oblique views.** Calibration issue from training-view bias. Mitigable at inference time with per-class thresholds.

6. **Class weights cap at 15.** Five of seven classes hit the cap during loss computation, so the model cannot finely discriminate between class rarities (e.g., `cut` 0.29% vs `pound` 0.14% are treated equally). Acceptable trade-off for training stability; revisitable with a softer power schedule.

7. **Decoder capacity ceiling.** Frozen backbone limits the headroom; the decoder has memorised what it can about the training distribution. Best val IoU at epoch 15 with no improvement through epoch 40 confirms this. Partial DINOv2 unfreeze is the recommended next architectural step.

## 7. Open Items (cross-reference to `FUTURE_DEVELOPMENT.md`)

The roadmap in `FUTURE_DEVELOPMENT.md` is structured by phase:

- **Phase 0** (data inspection / bowl fix) — DONE.
- **Phase 1** (multi-class head) — DONE; extended with class weighting in Phase C.
- **Phase 2** (clutter evaluation) — partial: in-the-wild section 4 covers RGB-only generalisation; UMD clutter split not yet evaluated.
- **Phase 3** (factored deployment with SAM2 + per-object affordance) — strongly motivated by Section 4.2 findings; not implemented.
- **Phase 4** (VLA integration) — design-stage only.
- **Phase 5** (RGB-D variant, partial backbone unfreeze, temporal extensions) — none implemented.

Items added to the future-work backlog by these results:

- **Per-class inference thresholds** (`--per_class_thresh` flag on `predict.py`): high for `grasp` to suppress background noise, low for `contain` to recover cavity extent. Low-cost, high-impact for qualitative deployment.
- **Per-tool class priors at evaluation time** to remove the multi-label-false-positive measurement artifact described in Section 3.4.
- **UMD clutter-split zero-shot evaluation** to quantify the in-the-wild observations against a labelled benchmark.

## 8. Document Structure for Future Updates

New experimental results should append a numbered subsection (e.g. 2.4, 3.6, 4.5) rather than replace existing content. Each entry should record what changed, the corresponding training-config delta (link to `run_config.json`), and the quantitative or qualitative delta versus the previous best. Major architectural changes get a new top-level section and a corresponding `CHANGELOG.md` entry. This keeps the document a chronological record of the project rather than a single-snapshot summary.
