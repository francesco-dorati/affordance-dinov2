# Final Presentation — Slide Structure

Target: ~12–15 min talk + demo + Q&A. One idea per slide. Every slide lists
the rubric criterion it earns points on (technical quality 35%, clarity 20%,
results & analysis depth 25%, creativity & novelty 20%).

**[TODO markers]** = fill after the Phase D retrain (Monday).

---

## Slide 1 — Title
Project title, name, course, date. One full-bleed hero image: the mug
prediction (`docs/figures/sample_mug.png`) — three affordances firing on one
object. Sets the visual hook before a single word.

## Slide 2 — The Problem (Introduction & Motivation)
*Rubric: clarity + novelty.*
A robot sees a tool it has never encountered. Knowing "this is a knife" is
not enough — it must know *where to grab* and *how to approach*. Define
affordance in one sentence. One image: a knife with blade/handle regions
coloured differently.

## Slide 3 — Why This Matters (Motivation, robotics framing)
*Rubric: novelty.*
Humanoids in kitchens/workshops: no CAD models, no guaranteed depth (fails on
glass/reflective). Goal: RGB-only perception primitive → 6-DoF grasp packet
(where + approach direction). Position it as the "where do I touch this?"
module of a manipulation stack. This is the project's identity — built for a
real robotics use case, not just a benchmark.

## Slide 4 — Task & Dataset (Methodology)
*Rubric: technical quality.*
Two joint outputs: 7-channel multi-label affordance mask + dense surface
normals. UMD dataset: ~28K RGB-D turntable images, 7 classes, pixel labels.
Key point: **instance split** — val tools never seen in training. Show one
GT sample (RGB / label / normals).

## Slide 5 — Architecture (Methodology)
*Rubric: technical quality.*
The architecture figure (`docs/figures/architecture_overview.svg`).
Three talking points only: frozen DINOv2 (why: 142M-image priors, 28K images
would destroy them), DPT-style multi-scale taps (layers 2/5/8/11 — early =
spatial detail, late = semantics), RGB skip stem (the ViT's 32×32 grid cannot
recover 448×448 edges — the stem keeps them alive). 3.6M trainable params.

## Slide 6 — Losses (Methodology)
*Rubric: technical quality.*
L = mask + 5·normals + 0.5·smoothness. Mask: per-channel BCE + Dice
(affordances < 1% of pixels — BCE alone collapses to "all background").
Normals: masked cosine (only where the robot will touch). One line on
multi-label-not-softmax: a trowel face is both `support` and `scoop`.

## Slide 7 — The Iterative Story, Part 1: the invisible bowls
*Rubric: analysis depth + creativity. This is a narrative slide.*
Phase A binary baseline: bowls score IoU = 0.000. Not a model failure — a
supervision blind spot (bowls only have `contain`, which wasn't supervised).
Phase B: 7-class head → bowls 0.00 → 0.98. Lesson on screen: "zero scores
are diagnostics, not failures."

## Slide 8 — The Iterative Story, Part 2: rare classes
*Rubric: analysis depth.*
Phase B per-class analysis: `support` 0.89 train / 0.36 val. Phase C:
frequency-inverse pos_weight (√(N_neg/N_pos), clip 15) → `support` 0.36 →
0.53, headline 0.713 → 0.770. Honest caveat (one bullet): heavy pos_weight
buys recall, costs precision — foreshadows the in-the-wild failure mode.

## Slide 9 — The Bug We Almost Shipped (lessons learned)
*Rubric: creativity + technical quality. Likely the most memorable slide.*
Augmentation rotated normal *vectors* by +θ; the y-down camera frame needs
−θ. Up to 30° of supervision error on rotated samples — and training curves
looked completely normal. Show the verification table (synthetic plane,
recomputed GT): wrong sign 11.1° error @ θ=15°, fixed < 0.2°. Lesson on
screen: "geometric supervision must be verified geometrically — loss curves
won't tell you."

## Slide 10 — Final Results (Results & Analysis)
*Rubric: analysis depth.*
**[TODO: Phase D numbers]** Headline table: mean-IoU (per-sample AND
dataset-level — one bullet on why two aggregations), angular error
Phase C → Phase D, NYUv2 bins. Per-class bar chart. Phase A→B→C→D progression
chart (this is the money plot: each architecture decision → measured delta).

## Slide 11 — What the Model Gets Right (qualitative)
Sample grid: knife (cut+grasp), bowl (contain), turner (support), mug
(3 affordances at once). One observation for depth: predictions are often
smoother than the hand-painted GT — implicit label denoising.

## Slide 12 — In-the-Wild: what breaks and why (Results & Analysis)
*Rubric: analysis depth — failure analysis scores higher than success slides.*
21 phone photos, zero-shot. Works within object contours on novel kitchenware.
Three failure modes, each with a one-line cause: `grasp` fires on wood texture
(no negative supervision), `contain` conservative on oblique views
(turntable viewpoint bias — calibration, not capacity), transparent objects
fail (out of distribution). **[TODO: regenerate gallery with Phase D model]**

## Slide 13 — Limitations & Future Work (Conclusions)
*Rubric: technical quality.*
Honest limits: single-object training distribution, ~24° normals bounded by
Kinect GT noise, decoder capacity ceiling. Future: factored pipeline
(SAM2 finds objects → affordance model per crop — directly fixes failure
mode 1), text-conditioned affordance head, VLA integration as perception
tool / safety verifier. One roadmap graphic, course → startup.

## Slide 14 — Conclusions
Three takeaways, verbatim from the report's lessons learned: zero scores are
diagnostics; verify geometric supervision geometrically; the metric is part
of the method. Final numbers recap **[TODO]**.

## Slide 15 — Demo
Live: `python scripts/predict.py --input_dir <folder with 2–3 fresh phone
photos>` — narrate the 7-channel output while it runs (~seconds per image).
**Fallback (mandatory): pre-rendered outputs of the same images in the deck
appendix, in case of environment failure.** Optionally end with one
never-before-seen object photographed in the room.

---

## Q&A preparation (likely questions, crisp answers)

- *Why frozen backbone?* 28K narrow-domain images vs 142M-image priors;
  fine-tuning destroys generalisation. Partial unfreeze is listed future work.
- *Why multi-label, not softmax?* Regions legitimately carry several
  affordances (mug: wrap-grasp handle + contain cavity + grasp rim).
- *Why two IoU numbers?* Per-sample mean kept for continuity with earlier
  phases; dataset-level is batch-size-invariant and pixel-weighted — the
  defensible headline.
- *Did the bug fix change mask results?* No — masks carry no vector
  components; only normal supervision was corrupted.
- *Why not train on clutter?* UMD clutter labels are incomplete; loss must
  ignore unlabeled regions — a project in itself; zero-shot evaluation listed
  as next step.
- *Normals plateau?* Kinect-v1 noise (3–5 mm) → 8–15° GT noise via finite
  differences; we're near the supervision ceiling, not the model ceiling.

## Timing budget (15 min)

Intro/motivation 2 min (slides 1–3) · methodology 4 min (4–6) · story +
results 5 min (7–12) · conclusions 2 min (13–14) · demo 2 min (15).
