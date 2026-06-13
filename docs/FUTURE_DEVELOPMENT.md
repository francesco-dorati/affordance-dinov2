# Future Development Roadmap

This roadmap reorganizes the project's forward path around two coupled goals: turning the affordance model into a **real VLA skill** (a perception primitive a vision-language-action stack can call) and making the work **publishable** at a peer-reviewed venue. The two goals share most of the same work — the VLA framing is the paper's novelty, and the paper's rigor is what makes the skill credible. Steps are ordered by dependency. Each is described in a few lines; see `RESULTS.md` for current numbers and the failure-mode evidence that motivates them.

## Goals

- **VLA skill:** the model is a callable, swappable perception primitive — given an image (or an image + a language goal), it returns per-object affordance heatmaps + surface normals that a planner or VLA consumes to choose a grasp.
- **Publishable:** a sharp novel claim, an honest benchmark against external baselines, ablations, a robustness study, and a demonstration of the VLA integration.

## Current State (June 2026)

Frozen DINOv2 ViT-Small backbone → DPT-style multi-scale fusion decoder → 7-channel multi-label affordance mask + 3-channel surface normals, trained on the UMD single-object split. Best checkpoint: val mean-IoU 0.7697, mean angular error 24.26°, +0.057 over the binary baseline. In-the-wild RGB inference on 21 phone photos done; dominant failure mode is `grasp` firing on wood-grain backgrounds. **Done:** Phase 0 (bowl/supervision-scope fix), Phase 1 (multi-class head + frequency-inverse class weights). **Partial:** Phase 2 (in-the-wild done; UMD clutter split not yet evaluated). Full detail in `RESULTS.md`.

---

## Pre-retrain tooling status (June 13, 2026)

Before the scheduled retrain, the publish-blocking infrastructure was put in place so the single retrain produces directly-comparable, honest numbers:

- **Bug fixes verified live.** Normal-rotation `-angle` fix confirmed numerically (<0.2° vs 3.7–11.1° for the old `+angle`) in `tests/test_normal_rotation.py`; the unbiased dataset-level IoU accumulation (`iou_accumulate`/`iou_from_accumulated`) is the reported path in `evaluate.py`. Both are active for any new run.
- **Canonical UMD splits implemented** (`utils/dataset.py`): `novel_instance` (Myers per-category instance holdout — the AffordanceNet Table II protocol; 80 train / 25 val tools, all 17 categories on both sides), `category` (whole-category holdout), and `file` (drop in official UMD split lists for exact comparability). Exposed via `--split_type` on `train.py` and `evaluate.py`; the exact assignment is written to `split_<type>.json` per run.
- **Weighted F-measure implemented** (`utils/metrics.py`): Margolin et al. $F_\beta^\omega$, $\beta^2{=}0.3$ — the metric AffordanceNet reports. Integrated into `evaluate.py` (per-class + average, AffordanceNet-Table-II format); self-test passes.

What remains is to run the retrain on the canonical split and execute the baseline (below).

# Track A — Make it publishable

The minimum set of work that turns the current course project into a paper. Items A1–A2 are blocking (publish-or-perish quality gates); A3–A6 build the actual contribution.

## A1. Retrain on honest numbers [blocking — tooling done, run pending]

The current headline (val mean-IoU 0.7697, 24.26° normals) was produced with the normal-rotation sign bug and the biased per-batch IoU. Both are now fixed and verified; what's left is to **run** the retrain on the canonical split so the reported numbers come from the corrected path. Until that run lands, no current number is citable. ~Half a day of compute; everything downstream depends on it.

## A2. Standard benchmark protocol + external baseline [blocking]

The canonical comparison is **AffordanceNet (Do, Nguyen & Reid, ICRA 2018), UMD Table II**, metric $F_\beta^\omega$ ($\beta^2{=}0.3$), on the Myers et al. 2015 split — average **0.799**, with DeepLab **0.733** and ED-RGB **0.766** on the same split. The split + metric are now implemented, so: (1) report your retrained model's $F_\beta^\omega$ in that table's format; (2) **cite the published baseline numbers directly** (don't re-run AffordanceNet's Caffe code) once your split/metric match; (3) train one **DeepLabv3+** yourself on the same split as a same-code control — validated by reproducing DeepLab ≈0.733 (the trainer is now implemented at `scripts/train_baseline.py`). Note the architectural difference in the writeup (AffordanceNet is detection-based multiclass softmax; ours is pure segmentation, multi-label sigmoid, RGB-only, *plus* normals). A second dataset (IIT-AFF) strengthens it but is optional.

## A3. Ablation table

Run the ablations already implied by the architecture so each design choice is justified: with/without surface-normal head, with/without RGB skip connections, ViT-S vs ViT-B backbone, and the class-weight `pos_weight` cap. Several of these are currently marked "not separately ablated" — turning them into a table is the difference between an engineering writeup and a paper.

## A4. Robustness study (clutter + in-the-wild)

Finish Phase 2: extract the already-downloaded UMD clutter tarball, write a small dataset adapter, and run the model zero-shot to get a quantitative generalization-gap number. Pair it with the existing qualitative in-the-wild failure-mode analysis. Together they give the paper an honest "where it breaks" section, which strong papers have and weak ones omit.

## A5. Sharpen the contribution statement

State the novelty explicitly rather than burying it: the multi-task affordance + surface-normal "grasp packet" as a single primitive, *and* its role as a callable VLA skill (Track B). The related-work section must situate the work against both affordance-segmentation and VLA literature. Without a one-sentence answer to "what's new?", the rest doesn't matter.

## A6. Write to venue format

Draft in the target venue's template (LNCS for ECCV-family), with related work, method, the A2–A4 results, ablations, and a limitations section (material for which already exists in `RESULTS.md`). For a workshop paper this is 4–8 pages; confirm the specific workshop and its exact deadline before committing scope.

---

# Track B — Make it a real VLA skill

Three integration patterns, ordered by effort. B1 is the smallest credible demonstration and is the right contribution for a near-term workshop paper. B2 and B3 are the longer research arc and the startup's moat.

## B1. Factored per-object pipeline [foundation]

Stop treating the model as whole-scene. Put an open-vocabulary segmenter (Grounding DINO + SAM2, or OWL-ViT) in front; run the affordance model once per detected object mask. This directly fixes the dominant wood-grain false-positive failure mode (background is never fed to the affordance head) and is the standard structure modern robotics stacks use. No retraining required — the current multi-class head drops straight in.

## B2. The VLA verifier skill [smallest publishable VLA result]

The cheapest credible VLA integration: the VLA proposes a grasp, the affordance model independently scores whether that location actually has the requested affordance and rejects bad proposals. Gives a measurable headline result ("reduces VLA grasp hallucinations by X%"), a concrete safety/reliability story, and a real Track-A contribution — all without changing the model architecture. This is the recommended near-term VLA deliverable.

## B3. The VLA tool call

Wire the affordance model as a tool the VLA invokes: instruction in ("pour water into the mug") → VLA calls SAM2 + affordance model → gets per-object `contain` heatmaps + normals → picks the object → hands a grasp packet to the planner. The model stays language-agnostic. Matches how Pi-0 / OpenVLA stacks are built; the right first *product* integration for the startup even if the paper leans on B2.

## B4. Text-conditioned affordance head [research moat, long horizon]

Replace the fixed 7-channel output with a head conditioned on a text embedding (FiLM or cross-attention over a CLIP/LM embedding), producing a heatmap for whatever affordance the prompt names — the RoboPoint / ManipLLM / Affordance-LLM direction. This is the true "VLA skill" and the differentiating research bet, but UMD's 7 classes won't get you there: it needs web-scraped or VLM-teacher-labeled affordance data. A ~six-month project, not a course extension — target a 2027 venue (ICCV / CoRL / RSS) or ECCV 2028.

---

# Model improvement backlog

Independent of the two tracks, these raise the ceiling and several feed directly into A1/A3. Ordered by return on effort.

- **Bigger frozen backbone.** Swap ViT-S → `dinov2_vitb14_reg` (zero new trainable params; register variants give cleaner dense features). Likely the cheapest large accuracy gain. One hub string + `embed_dim` 384→768.
- **LR schedule + checkpoint selection.** Constant 1e-4 oscillates after epoch 15. Add cosine decay with warmup; select best on val mean-IoU, not val loss; consider EMA of decoder weights.
- **Loss calibration.** Five of seven classes saturate the `pos_weight` cap at 15, trading precision for recall — and the main in-the-wild failure is a precision failure. Lower the cap to ~5 and let per-channel Dice carry rare classes, or switch to focal loss. Pair with per-class inference thresholds.
- **Negative background supervision.** UMD has no "textured surface that isn't a handle" negatives. Composite segmented UMD tools onto random indoor backgrounds. Highest-value data-side fix for the startup deployment case.
- **Per-class inference thresholds.** A `--per_class_thresh` flag on `predict.py` (high for `grasp`, low for `contain`) cleans up qualitative output with no retraining. 10-line change, trivial.
- **Per-tool class priors at eval.** Score only channels that exist on a given tool, removing the multi-label false-positive measurement artifact in `RESULTS.md` §3.4.
- **Partial backbone unfreeze.** After decoder convergence, unfreeze the last 2–4 ViT blocks at ~1e-5 (or LoRA adapters) — the standard escape from the observed plateau.
- **Cleaner normal GT + scale alignment.** Bilateral-filter depth before differentiation to lift the 24° normals ceiling; match train/predict preprocessing so deployed objects appear at the trained scale.

---

# Recommended critical path

1. **A1** (fix bugs, retrain) — nothing is publishable until the numbers are honest.
2. **A2** (standard protocol + external baseline) — the gate every reviewer checks.
3. **B1** (factored per-object pipeline) — fixes the headline failure mode, foundation for the VLA story.
4. **B2** (VLA verifier) — the smallest real VLA contribution, gives a measurable result.
5. **A3 + A4** (ablations + robustness) — round out the experimental section.
6. **A5 + A6** (contribution framing + write-up) — assemble the paper.

B3 is the first startup product milestone; B4 is the long-term research bet and the next paper. Confirm the target workshop's exact deadline before locking scope — it decides whether this is a few-week workshop sprint or a 2027 full-paper effort.
