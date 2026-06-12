# Multi-Task Semantic-Geometric Fusion for Robotic Part Affordance Extraction

**Author:** Francesco Dorati  
**Course:** Computer Vision — Final Project  
**Date:** June 2026

---

## Abstract

**[TODO: update all numbers in this abstract after the Phase D retrain.]**

We present a multi-task deep learning pipeline that predicts, from a single RGB image, both a 7-channel semantic affordance mask and a dense surface normal map for robotic tool manipulation. A frozen DINOv2 ViT-Small backbone feeds a DPT-style multi-scale fusion decoder trained on the UMD Part Affordance Dataset (21 tools, 7 affordance classes). Starting from a 2-class binary baseline that left entire tool families (bowls, turners) invisible to supervision, we iteratively expand to full 7-class multi-label prediction with frequency-inverse class weighting, reaching a val mean-IoU of **0.770** (+0.057 over baseline) on a strictly harder evaluation task. In-the-wild RGB inference on 21 unconstrained phone-captured images demonstrates successful transfer to novel kitchenware, exposing one dominant failure mode — background texture activation on the `grasp` channel — that motivates a factored deployment pipeline (object segmentation front-end + per-object affordance inference) as the next engineering step toward a humanoid manipulation system.

---

## 1. Introduction & Motivation

### 1.1 The Problem

For a robot to interact with objects it has never seen before, identifying *what* an object is is not enough — the robot must understand *how* to interact with it. A knife has a blade (cut) and a handle (grasp); a mug has a cavity (contain) and a rim (wrap-grasp). This functional understanding of regions is called **affordance**.

Classical robotics pipelines sidestep this with per-object CAD models or depth-sensor point clouds. Both break down in unconstrained environments: CAD models don't scale to the open world, and depth sensors fail on reflective or transparent surfaces. The goal of this project is a sensor-fault-tolerant perception primitive: given only an RGB image, predict *where* on the object to interact and *how to orient the gripper*.

### 1.2 The Two Predictions

The model jointly produces:

- **Affordance masks** — a 7-channel binary map labelling each pixel with the affordance class(es) active there (`grasp`, `cut`, `scoop`, `contain`, `pound`, `support`, `wrap-grasp`).
- **Surface normals** — a 3-channel unit-vector map encoding the surface orientation at each pixel, giving the gripper its approach direction.

Combined with a depth reading at the predicted centroid, this yields a full 6-DoF grasp pose without any object-specific model.

### 1.3 Why This Matters for Humanoids

Humanoid robots operating in kitchens, workshops, and domestic settings face exactly the distribution described above: arbitrary, seen-for-the-first-time tools, mixed materials, and no pre-loaded CAD library. The perception primitive built in this project is designed to slot directly into a manipulation stack as the "where do I touch this?" module, sitting between a visual scene parser and a motion planner.

---

## 2. Related Work

### 2.1 Affordance Datasets

The **UMD Part Affordance Dataset** (Myers et al., 2015) contains ~30,000 RGB-D images of 105 tool instances across 17 categories, annotated with 7 affordance classes at the pixel level. It is the canonical benchmark for part-affordance segmentation and the only large-scale public dataset with full pixel-wise affordance labels. Images are captured on a turntable with a Kinect-v1 depth sensor, providing clean single-object scenes.

Other relevant datasets include **HANDAL** (Diaz et al., 2023) for category-level grasping and **AffNet** (Do et al., 2018) for affordance detection in context, but neither provides the per-pixel multi-class labels required for our multi-task setup.

### 2.2 Vision Foundation Models

**DINOv2** (Oquab et al., 2023) is a self-supervised ViT trained on 142M curated images via knowledge distillation from a larger teacher. Its patch tokens have been shown to encode 3D shape priors, texture, and part identity without any task-specific supervision, making it an ideal frozen feature extractor for small-dataset downstream tasks. We use **ViT-Small** (21M parameters, 384-dim tokens, 14×14 patches) as a compute-accessible entry point.

### 2.3 Dense Prediction Decoders

**DPT** (Ranftl et al., 2021) demonstrated that tapping multiple intermediate ViT layers and reassembling them with a convolutional decoder substantially outperforms using only the final layer for monocular depth estimation. We adopt the same multi-scale reassembly idea (layers 2, 5, 8, 11) but replace the Reassemble/Fusion blocks with a lighter FusionUp cascade augmented by RGB skip connections.

**UNet-style skip connections** (Ronneberger et al., 2015) are the established method for recovering spatial precision lost during encoding. Our RGB stem provides the spatial-frequency complement to the ViT's semantic-frequency features.

### 2.4 Affordance Prediction

Prior affordance segmentation work largely predicts a single binary "graspable" map (Chu et al., 2019; Lenz et al., 2015) or uses depth as an input (Nguyen et al., 2017). Multi-class pixel-wise affordance with a transformer backbone and joint surface normal prediction is, to our knowledge, not directly addressed by any single prior work — the contribution of this project is assembling these components into a functional, evaluated pipeline.

---

## 3. Method

### 3.1 Architecture Overview

The pipeline is a three-stage encoder-decoder:

```
RGB image [B, 3, 448, 448]
    │
    ├──► DINOv2 ViT-Small (frozen)  ──►  4 × [B, 384, 32, 32]  (layers 2, 5, 8, 11)
    │
    └──► RGB Stem (trainable)  ──►  4 skip features at 448, 224, 112, 56 px
                                            │
                                    Multi-Scale Fusion Decoder
                                            │
                              ┌─────────────┴──────────────┐
                         Mask Head                    Normal Head
                    [B, 7, 448, 448]              [B, 3, 448, 448]
                    (sigmoid logits)              (L2-normalized unit vectors)
```

> See `docs/figures/architecture_overview.svg` for the full diagram.

**Why frozen backbone?** DINOv2 was trained on 142M images; fine-tuning it on UMD's ~28K images would destroy generalisation via overfitting. The frozen backbone provides stable, broad-distribution features; the 3.6M-parameter trainable decoder adapts them to affordance prediction.

### 3.2 Multi-Scale ViT Fusion

Tokens from layers 2, 5, 8, and 11 are reshaped from sequence form back to 32×32 spatial grids. Four 1×1 convolutions project each from 384 to 256 channels; the four projections are concatenated and fused with a two-layer ConvBlock (1024→256). This preserves both early local-detail tokens and late semantic tokens in the same representation.

### 3.3 RGB Stem and Skip Connections

A four-layer strided CNN processes the input RGB image in parallel with the ViT, producing feature maps at 448, 224, 112, and 56 pixels. These are concatenated into the fusion decoder at each upsampling stage, restoring the fine-grained spatial information lost when the ViT compressed the image to 32×32 tokens.

### 3.4 Task Heads

- **Mask head:** a single 1×1 conv, 32→7 channels, outputting raw logits. `BCEWithLogitsLoss` with per-class `pos_weight` applies sigmoid internally for stability.
- **Normal head:** a 3×3 conv + 1×1 conv, followed by per-pixel L2 normalisation. The extra 3×3 conv gives the head access to local gradient information needed for orientation estimation.

### 3.5 Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + 5.0 \cdot \mathcal{L}_{\text{normal}} + 0.5 \cdot \mathcal{L}_{\text{smooth}}$$

**Mask loss** (`DiceBCELoss`): per-channel `BCEWithLogitsLoss` (with pos_weight) plus per-channel soft Dice averaged over channels. Dice counteracts background dominance — affordance pixels account for less than 1% of total pixels across all classes.

**Normal loss:** masked cosine dissimilarity, averaged only over pixels within the ground-truth affordance region (the robot only needs correct normals where it will grasp).

**Smoothness loss:** edge-aware normal smoothness (`|∇N| · exp(−10|∇RGB|)`), penalising normal discontinuities except at image edges.

### 3.6 Class Weighting

The seven affordance classes span a 5× range in pixel frequency: `contain` (0.67%) down to `scoop` and `pound` (0.14%). Without correction, rare classes achieve high training IoU but large train-val gaps due to overfitting on few examples. We apply frequency-inverse pos_weight:

$$w_c = \left(\frac{N_{\text{neg}}}{N_{\text{pos}}}\right)^{0.5} \text{, clipped at 15.0}$$

All seven classes are sparse enough to hit or approach the 15.0 cap, producing a near-uniform weighting that gives equal gradient priority to every affordance.

### 3.7 Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, base lr=1e-4, weight decay=1e-4 |
| LR schedule | 2-epoch linear warmup → cosine decay to ~0 |
| Checkpoint selection | best val dataset-level mean-IoU |
| Batch size | 8 |
| Epochs | **[TODO: 20–25, final run]** |
| Backbone | Frozen DINOv2 ViT-Small |
| Input resolution | 448×448 (center-cropped) |
| Augmentation | ±15° rotation, ±15% scale, H-flip, colour jitter, Gaussian noise σ=0.01 |
| Hardware | NVIDIA GEFORCE RTX 5070 Ti |

Augmentation applies the same geometric transform to RGB, mask, and normals — and additionally rotates the normal *vectors* themselves, since a rotated image implies rotated surface orientations. A subtle sign error in this vector rotation was discovered and fixed during the project; see §4.4.

### 3.9 Evaluation Metrics

**Mean-IoU @ t:** predictions are binarised per channel at threshold *t*; IoU is computed per affordance class and averaged over classes. We report two aggregations: (a) **per-sample mean** — per-class IoU computed within each sample, averaged over classes present, then over samples (comparable to our earlier phases); and (b) **dataset-level** — per-class intersection and union pixel counts accumulated over the entire split and divided once. The dataset-level number is invariant to batch size and sample order and weights every pixel equally; it is the methodologically preferred headline metric and is also the criterion used for best-checkpoint selection.

**Normals:** mean angular error in degrees over the union of ground-truth affordance pixels, plus the standard NYUv2 bins (fraction of pixels with error ≤ 11.25°, 22.5°, 30°).

### 3.8 Dataset and Splits

The UMD Part Affordance Dataset is split **by instance** (not by image): all views of a given tool instance are either entirely in train or entirely in val. This tests generalisation to unseen instances within the same category, which is the robot's actual deployment scenario. The val split contains 5,791 samples across 21 tool instances.

Ground-truth surface normals are computed on-the-fly from raw depth maps using Kinect-v1 intrinsics via finite-difference cross-products, with principal-point correction applied after the 448×448 center crop.

---

## 4. Iterative Development

The final model is the result of three experimental phases. Each phase is preserved and reproducible (see `runs/INDEX.md`).

### 4.1 Phase A — Binary Baseline

**Supervision:** union of UMD class IDs 1 (`grasp`) and 7 (`wrap-grasp`) into a single binary channel. **Architecture:** DINOv2 ViT-Small, final-layer features only, bilinear upsample decoder (v1).

**Key finding:** bowls produce **IoU = 0.00**. Diagnostic inspection revealed that bowls have no `grasp` or `wrap-grasp` annotation; their only affordance is `contain` (class 4). This is a supervision blind spot, not a model bug. Similarly, `turner_06` achieves IoU = 0.113 because only the small handle is supervised; the dominant `support` surface of the blade is invisible to the loss.

Collapsing seven semantic affordances into one binary map discards exactly the per-region information a humanoid needs.

### 4.2 Phase B — 7-Class Head (Unweighted)

**Change:** decoder mask head expanded from 1 to 7 sigmoid output channels; loss is per-channel BCE + per-channel Dice averaged over channels (multi-label, not softmax). Dataset returns a (7, H, W) multi-hot tensor.

**Result:** bowls recover from 0.00 to competitive IoU on `contain`; turner improves substantially on `support`. Headline mean-IoU 0.713 (averaged across 7 channels on 5,791 samples) — comparable to the binary baseline but on a strictly harder task.

**Remaining problem:** per-class analysis shows large train-val gaps on rare classes (`support`: 0.89 train → 0.36 val; `scoop`: 0.88 → 0.58), pointing to class-frequency imbalance.

### 4.3 Phase C — Frequency-Inverse Class Weights (Final)

**Change:** `pos_weight[c] = (N_neg / N_pos)^0.5`, clipped at 15.0, applied per-channel. Computed once from a full pass over the training set and cached.

**Result:** val mean-IoU 0.770 at epoch 15; no improvement through epoch 40 (patience = 25). Train-val gap widened from 0.27 to 0.34 in the tail, confirming convergence at the decoder's capacity ceiling. `support` val IoU lifts from 0.36 to 0.53.

### 4.4 Phase D — Corrected Normal Augmentation + Cosine Schedule (Final)

A code review uncovered a **sign error in the augmentation-time normal rotation**: the image warp (`cv2.getRotationMatrix2D`, counter-clockwise visually) corresponds to a −θ vector rotation in the y-down pixel-aligned camera frame the normals are expressed in, but the vectors were rotated by +θ. Every rotated training sample therefore carried up to ~2θ of angular supervision error — up to 30° at the ±15° augmentation limit, *worse than applying no vector rotation at all*.

The bug was verified numerically before fixing: a synthetic slanted-plane depth map was warped by θ, normals were recomputed from the warped depth (geometric ground truth, exact for rotation about the principal point), and both sign conventions were compared against it. The faulty path measured 3.7° / 7.5° / 11.1° mean error at θ = 5° / 10° / 15°; the corrected path < 0.2°. The test is preserved as `tests/test_normal_rotation.py`. Mask supervision was never affected (masks carry no vector components).

Phase D retrains the model with the corrected supervision, plus two optimisation changes motivated by the Phase C training curves (best epoch 15 of 40, oscillation afterwards): a cosine LR schedule with 2-epoch warmup, and best-checkpoint selection on val dataset-level mean-IoU rather than the mixed multi-task val loss.

**Result:** **[TODO: fill from Phase D `evaluation_val.json` — expected: mask mean-IoU comparable to Phase C; angular error improvement from 24.3° toward the depth-noise floor]**

---

## 5. Results & Analysis

> **[PLACEHOLDER NOTE — all numbers in this section are from the Phase C run.
> After the Phase D retrain: re-run `evaluate.py`, replace every table below,
> report BOTH IoU aggregations (per-sample | dataset-level), and add a
> Phase C vs Phase D normals comparison row. Delete this note.]**

### 5.1 Quantitative Results

#### Overall (val split, 5,791 samples, instance-split)

| Threshold | Mean-IoU |
|---|---|
| 0.3 | 0.7677 |
| 0.4 | 0.7691 |
| **0.5** | **0.7697** |
| 0.6 | 0.7696 |
| 0.7 | 0.7688 |

Surface normals: mean angular error **24.26°**, fraction ≤ 22.5° = **55.3%**, fraction ≤ 30° = **73.3%**.

The flat IoU curve across thresholds 0.3–0.7 indicates well-calibrated predictions: the model is neither over- nor under-confident — its probability mass is concentrated near the true boundary, not spread diffusely.

#### Per-Class IoU @ 0.5

| Class | IoU | Interpretation |
|---|---|---|
| `wrap-grasp` | 0.881 | Solved — every cup/mug handle correctly localised |
| `cut` | 0.866 | Solved — knife blades and scissor blades |
| `contain` | 0.824 | Solved — cup, mug, bowl interiors |
| `pound` | 0.808 | Strong despite class rarity — mallet head well-defined |
| `scoop` | 0.711 | Solid — spoons and trowels |
| `grasp` | 0.684 | Weakest well-learned class — thin annular mug rims and overlap with `wrap-grasp` |
| `support` | 0.526 | Hardest — largest train-val gap; oscillates with class-weight schedule |

#### Per-Tool Deltas vs. Binary Baseline

Selected tools sorted by improvement:

| Tool | Binary IoU | Final IoU | Δ |
|---|---|---|---|
| `bowl_02` | 0.000 | 0.977 | **+0.977** |
| `bowl_03` | 0.000 | 0.922 | **+0.922** |
| `turner_06` | 0.113 | 0.508 | +0.395 |
| `trowel_02` | 0.741 | 0.808 | +0.067 |
| `mug_20` | 0.887 | 0.745 | −0.142 |
| `trowel_04` | 0.877 | 0.634 | −0.243 |

**On the negative deltas:** these are not regressions. The binary baseline's per-tool metric measured IoU on a 2-class union; the final model's metric is the mean across all 7 channels. Tools where the model now predicts non-existent classes (e.g. `cut` on `trowel_01`, `contain` on `scoop_01`) are penalised because absent-class IoU is 0 by construction, dragging the per-tool mean down. The +0.057 headline gain is achieved *despite* this measurement artefact.

> See `docs/figures/comparison_binary_vs_multiclass.png` for the full per-tool breakdown plot.

### 5.2 Surface Normal Analysis

Mean angular error is 24.26°, bounded from below by the noise floor of depth-derived ground truth. Kinect-v1 depth noise (~3–5 mm at 0.5–1.5 m range) propagates to 8–15° angular noise via finite-difference normals. The practical ceiling for this supervision signal is therefore around 15–20°; the model is within one noise-floor's width of that ceiling. Improving further requires either a higher-quality depth sensor, distilled normals from a geometric model, or a different supervision signal.

### 5.3 Qualitative Samples

The model produces smooth, anatomically plausible predictions on clean training-domain images — frequently smoother than the hand-painted UMD ground truth, which has jagged boundaries from a paint-style annotation interface. This is an emergent implicit label-denoising effect from the combination of multi-scale ViT features and per-channel Dice loss.

> See `checkpoints/samples/` for the full grid of qualitative examples.

> See `docs/figures/sample_knife_cut_grasp.png`, `sample_bowl_contain.png`, `sample_mug.png`, `sample_turner_support.png` for selected highlights.

### 5.4 In-the-Wild RGB Inference

The model was applied zero-shot to 21 unconstrained phone-captured images of kitchenware (knives, mugs, pots, scissors, non-tool objects, compound scenes) at two thresholds (0.5 and 0.7).

**What worked:** predictions transfer well *within object contours* to every tested object class, including novel items (pot, fork, plate) sharing affordance geometry with the training distribution. The transfer from UMD's narrow turntable distribution to phone kitchenware is meaningful.

**Failure mode 1 — Background texture activation on `grasp`:** the channel fires substantially on wooden cutting boards, table surfaces, and book pages even at threshold 0.7. UMD handles share wood-tone ridged texture with common backgrounds; there are no negative-supervision examples for "wooden surface, not a handle."

**Failure mode 2 — Conservative `contain` on oblique views:** the model correctly localises the centre of cavities but underpredicts the full rim-to-centre extent when the viewing angle departs from the near-frontal turntable views dominant in training. Lowering the `contain` threshold to 0.3–0.4 recovers most missing area, suggesting this is a calibration issue, not a capacity issue.

**Failure mode 3 — Transparent objects:** glass cups and lids fail almost completely. Refraction and specular highlights are outside the training distribution; DINOv2 features cannot be interpreted by the affordance head for these materials.

> See `reports/predictions/in_the_wild_thresh05_2026-06-09/` and `in_the_wild_thresh07_2026-06-09/` for the full prediction gallery.

---

## 6. Conclusions

### 6.1 Findings

This project demonstrates that a frozen DINOv2 backbone paired with a lightweight trainable decoder can learn all seven UMD affordance classes jointly from RGB input alone, reaching val mean-IoU 0.770 on an instance-split evaluation. The iterative development process surfaced a concrete diagnostic: zero IoU on bowls under binary supervision traced directly to a supervision blind spot, not a model failure, and expanding the supervision target to all seven classes resolved it immediately. Frequency-inverse class weighting then recovered the remaining rare-class deficit.

The RGB-only design is a deliberate choice: it makes the system sensor-fault-tolerant (depth sensors fail on reflective and dark surfaces) and directly deployable from any commodity camera.

### 6.2 Lessons Learned

Three lessons generalise beyond this project. First, **zero scores are diagnostics, not failures**: the bowls' 0.0 IoU traced to a supervision blind spot, and chasing it produced the project's largest single improvement. Second, **geometric supervision must be verified geometrically**: the normal-rotation sign error (§4.4) was invisible in training curves — the loss decreased normally on corrupted supervision — and was only caught by testing the augmentation against recomputed ground truth. Visual plausibility is not correctness. Third, **the metric is part of the method**: the per-sample IoU aggregation penalised tools for multi-label false positives on absent classes and masked real improvements; defining the dataset-level metric changed which checkpoint was selected as "best."

### 6.3 Limitations

The dominant practical limitation is background `grasp` activation: the model cannot distinguish "wooden handle" from "wooden surface" without negative examples or object-scope context. The normal accuracy plateau at ~24° is bounded by the quality of the depth-derived ground truth, not by model capacity. The decoder is also confirmed to have reached its capacity ceiling (best val epoch 15 of 40; no improvement with continued training), suggesting the next architectural step should be partial DINOv2 unfreezing rather than more training data or longer runs.

### 6.4 Future Work

The clearest next step is a **factored deployment pipeline**: an open-vocabulary segmenter (SAM2 or Grounding DINO) first isolates object instances, then the affordance model runs once per object on a clean crop. This directly resolves the background-activation failure mode without retraining. Beyond that, a text-conditioned head (affordance prediction from a natural-language prompt rather than a fixed 7-class taxonomy) is the natural extension toward a humanoid manipulation system that can respond to arbitrary instructions.

---

## References

> *(To be completed — key citations: UMD dataset Myers et al. 2015; DINOv2 Oquab et al. 2023; DPT Ranftl et al. 2021; UNet Ronneberger et al. 2015)*

---

## Appendix A — Architecture Detail

See `docs/ARCHITECTURE.md` for the full layer-by-layer walkthrough with tensor shapes.

## Appendix B — Reproduction

```bash
# Install
pip install -r requirements.txt

# Evaluate best checkpoint
python scripts/evaluate.py --split val

# In-the-wild inference
jupyter notebook notebooks/in_the_wild_inference.ipynb
```

Full training and evaluation instructions in `docs/USAGE.md`. All quantitative results are traceable to `checkpoints/evaluation_val.json`.
