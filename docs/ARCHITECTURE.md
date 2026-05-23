# Architecture Deep Dive

A step-by-step walkthrough of how the model works, written for someone who is
new to deep learning. Every operation is explained — what it does, what shape
goes in and what shape comes out, and **why** it is there.

If you only want the shapes-and-types reference, jump to section 11. If you
want the picture, see `architecture_diagram.svg` in this folder.

![Architecture diagram](architecture_diagram.svg)

---

## 1. The problem in plain English

A robot is looking at a tool it has never seen before — say, a screwdriver
lying on a table. We want the robot to answer two questions from a single
camera frame:

- **Where on the object should I grab it?**  (the *affordance mask* — the
  handle, not the metal tip.)
- **How should I orient my gripper as I approach?**  (the *surface normal* —
  the direction sticking out of the handle, so the gripper doesn't collide.)

Most existing robot systems answer this by building a 3D model of the object
ahead of time. We want to avoid that: the robot should figure it out from
sensor data alone, even for objects it has never been trained on. That is the
"zero-shot" goal.

The model takes one RGB image (and optionally depth) and produces two dense
prediction maps — one for "is this pixel grabbable?" and one for "what
direction is the surface here pointing?". Combine those with a depth reading
and you get a 6-DoF pose (3 positions + 3 orientations) that a motion planner
can execute.

---

## 2. Shape notation

Throughout the document I write tensor shapes in PyTorch's "channels-first"
convention:

```
[B, C, H, W]
 │  │  │  └── width  in pixels
 │  │  └───── height in pixels
 │  └──────── number of feature channels
 └─────────── batch size (how many images at once)
```

So `[8, 3, 448, 448]` means *eight images, three channels (R/G/B), 448 pixels
tall, 448 pixels wide*.

We always use a square 448×448 crop because DINOv2 wants its inputs to be
divisible by 14 (its patch size), and 448 = 14 × 32, giving a tidy 32×32 grid
of patches.

---

## 3. Phase 0 — Preparing the data

The class `UMDAffordanceDataset` in `utils/dataset.py` is responsible for
turning the raw UMD files on disk into the tensors the model expects.

For one sample it produces a dictionary:

| Key           | Shape                | Type    | Meaning                                              |
|---------------|----------------------|---------|------------------------------------------------------|
| `rgb`         | `[3, 448, 448]`      | float32 | RGB image, ImageNet-normalized                       |
| `mask`        | `[1, 448, 448]`      | float32 | Binary ground-truth affordance mask (0 or 1)         |
| `normals`     | `[3, 448, 448]`      | float32 | Ground-truth surface normals as unit vectors         |
| `tool_name`   | (string)             | str     | Name of the source tool, kept for per-tool reporting |

The `DataLoader` then stacks N samples to add the batch dimension, so the
model actually sees `[B, 3, 448, 448]`, `[B, 1, 448, 448]`, etc.

**Why ImageNet-normalized?** Because the backbone (DINOv2) was trained with
images normalized using `mean = [0.485, 0.456, 0.406]` and
`std = [0.229, 0.224, 0.225]`. If we feed it differently-normalized images,
its learned features make no sense.

**Where do the ground-truth normals come from?** From the depth map. For each
pixel we back-project (u, v, depth) into a 3D point (X, Y, Z) using the
camera intrinsics, build small 3D triangles between neighbouring pixels, and
take the cross product to get the surface normal. This is done on the fly in
`utils/geometry.py::compute_normals`, with the principal point shifted to
account for the center crop so the geometry stays correct.

---

## 4. Phase 1 — The backbone (DINOv2 ViT-Small, frozen)

**File:** `models/backbone.py` → class `DINOv2Backbone`

### 4.1 What is DINOv2?

DINOv2 is a *self-supervised* vision transformer trained by Meta on 142 million
images. "Self-supervised" means it learned without any human labels — it
played a kind of guessing game on image patches and emerged with extremely
strong visual features. We use the smallest variant, **ViT-Small**, which has
about 21 million parameters.

### 4.2 Why frozen?

DINOv2 already knows what tool handles look like, what edges and textures and
3D shapes look like. If we fine-tune it on our small dataset, we will almost
certainly *destroy* that broad knowledge with our narrow training signal. So
we **freeze** it: all of its weights are marked `requires_grad=False` and
never change during training. The decoder we train on top is the only thing
that learns.

### 4.3 What is a patch token?

A Vision Transformer doesn't process pixels directly. It chops the image into
small non-overlapping patches (14×14 pixels each for ViT-Small), flattens
each patch into a vector, and projects it to a 384-dimensional embedding.
After that, each of those embeddings is a **token** that gets refined by 12
layers of *self-attention*, in which every token sees every other token.

For a 448×448 input:

```
patches per side = 448 / 14 = 32
total patches    = 32 × 32  = 1024
token dimension  = 384  (ViT-Small)
```

### 4.4 Why multi-scale (4 layer taps)?

The classic approach is to take only the **last** transformer block's output.
That's what the v1 baseline did. The problem: by the time information has
gone through 12 layers of global self-attention, fine spatial detail has been
heavily smoothed out into broad semantic concepts ("this region is metal,"
"this region is plastic").

The DPT paper (Ranftl et al.) showed that earlier blocks retain crisper
local detail before attention has fully mixed everything. So we tap **four**
blocks — layers 2, 5, 8, and 11 — and pass all of them down to the decoder.
The decoder gets the best of both: shape-level cues from early blocks and
semantic-level cues from late blocks.

### 4.5 Input → output

| Stage                                  | Tensor                              |
|----------------------------------------|-------------------------------------|
| Input                                  | `rgb`  `[B, 3, 448, 448]`           |
| Patch embed (14×14 conv stride 14)     | `[B, 1024, 384]`                    |
| + Positional embedding                 | `[B, 1024, 384]`                    |
| + CLS token (1 extra global token)     | `[B, 1025, 384]`                    |
| 12 transformer blocks                  | `[B, 1025, 384]`                    |
| `get_intermediate_layers(2,5,8,11)`    | **list of 4 tensors** (CLS dropped) |
| Reshape back to a 32×32 grid           | **each:**  `[B, 384, 32, 32]`       |

**Output of Phase 1:** a Python list of 4 tensors, each
`[B, 384, 32, 32]` and frozen. These are the *semantic* features. They know
*what* the image contains; they are weak on *where exactly* the boundaries
are.

---

## 5. Phase 2 — The decoder

**File:** `models/decoder.py` → class `MultiTaskDecoder`

The decoder's job is to turn that 32×32 grid back into 448×448 pixel-precise
predictions. To do that it weaves together three streams of information:

1. The 4 ViT feature maps (coarse but semantic).
2. The original RGB image, processed by a small trainable CNN called the
   `RGBStem` (fine spatial detail).
3. Its own progressively-upsampled feature map.

Think of it like drawing: the ViT gives us a rough sketch of *what* is where,
and the RGB stem gives us the line work — *where exactly* the edges sit.

### 5.1 Project and fuse the ViT features

The 4 ViT outputs are still in their native 32×32 resolution. We need to
reduce their channel count (384 each is a lot to carry through the decoder)
and merge them into one tensor.

| Step    | Operation                                           | Output                       |
|---------|-----------------------------------------------------|------------------------------|
| 5.1.a   | `vit_proj`: one 1×1 Conv per layer, 384 → 256       | 4 × `[B, 256, 32, 32]`        |
| 5.1.b   | `torch.cat(dim=1)` — stack along channels            | `[B, 1024, 32, 32]`           |
| 5.1.c   | `vit_fuse`: ConvBlock (two 3×3 convs), 1024 → 256   | `[B, 256, 32, 32]`            |

**What is a 1×1 convolution?** It applies a small learned matrix to each
pixel independently, mixing channels but not pixels. It's the cheapest way
to change the number of channels.

**What is a 3×3 convolution?** Same idea, but each pixel's output also
depends on its 8 neighbours. That lets the network learn small local
patterns. The `ConvBlock` is `Conv3×3 → BatchNorm → ReLU → Conv3×3 →
BatchNorm → ReLU`. `BatchNorm` rescales the activations so training is
stable; `ReLU` introduces non-linearity (without it, stacking conv layers is
mathematically equivalent to a single layer).

### 5.2 The RGB Stem (parallel branch)

Independently of the ViT path, we send the same RGB image through a tiny
trainable CNN that produces four feature maps at four spatial resolutions.
These are the **skip connections** that will give the decoder fine-grained
spatial information.

| Layer    | Operation                              | Output                       | Name |
|----------|----------------------------------------|------------------------------|------|
| `stem`   | Conv 3×3 (3→32), Conv 3×3 (32→32)      | `[B, 32, 448, 448]`           | `s0` |
| `down1`  | Conv 3×3 stride 2 (32→64)              | `[B, 64, 224, 224]`           | `s1` |
| `down2`  | Conv 3×3 stride 2 (64→96)              | `[B, 96, 112, 112]`           | `s2` |
| `down3`  | Conv 3×3 stride 2 (96→128)             | `[B, 128, 56, 56]`            | `s3` |

**Why do we need this at all?** The ViT compressed the image to 32×32 — it
has thrown away 196× of spatial information. There is no way to recover
sharp 448×448 edges from a 32×32 token map; the information simply isn't
there. The RGB stem keeps that information alive in parallel.

**Why so few channels?** Because we don't need the stem to do heavy
reasoning — that's the ViT's job. The stem just preserves edges and local
texture. Keeping it shallow also keeps training fast.

### 5.3 The upsampling cascade — four `FusionUp` blocks

Now we combine the streams. Each `FusionUp` does three things:

1. **Bilinear upsample** the current feature map to twice (or 1.75×) its
   spatial resolution.
2. **Concatenate** the corresponding skip from the RGB stem along the
   channel dimension.
3. Pass through a **ConvBlock** to mix everything.

| Stage    | Input                  | After bilinear ↑       | + Skip                       | After Concat              | ConvBlock        | Output                       |
|----------|------------------------|------------------------|------------------------------|---------------------------|------------------|------------------------------|
| `up1`    | `[B, 256, 32, 32]`     | `[B, 256, 56, 56]`     | `s3` `[B, 128, 56, 56]`      | `[B, 384, 56, 56]`        | 384 → 192        | `[B, 192, 56, 56]`           |
| `up2`    | `[B, 192, 56, 56]`     | `[B, 192, 112, 112]`   | `s2` `[B, 96, 112, 112]`     | `[B, 288, 112, 112]`      | 288 → 128        | `[B, 128, 112, 112]`         |
| `up3`    | `[B, 128, 112, 112]`   | `[B, 128, 224, 224]`   | `s1` `[B, 64, 224, 224]`     | `[B, 192, 224, 224]`      | 192 → 64         | `[B, 64, 224, 224]`          |
| `up4`    | `[B, 64, 224, 224]`    | `[B, 64, 448, 448]`    | `s0` `[B, 32, 448, 448]`     | `[B, 96, 448, 448]`       | 96 → 32          | `[B, 32, 448, 448]`          |

**What is bilinear upsampling?** A simple math operation: to make an image
twice as large, you compute each new pixel as a weighted average of the
four nearest old pixels. It's smooth and parameter-free.

**Why not transposed convolution?** Transposed convs (sometimes called
"deconvolutions") can produce a characteristic *checkerboard* artifact in
dense prediction outputs because of how kernel strides overlap. Bilinear
upsampling sidesteps that entirely. We let the conv that *follows* do all
the learning.

**Why concatenate instead of add?** Adding requires the two tensors to have
the same number of channels and forces the network to learn to merge them
without picking favorites. Concatenating just stacks them together and lets
the ConvBlock decide how much weight to give each source. It costs more
parameters but is more flexible — important when one stream (ViT) is
semantic and the other (RGB stem) is low-level texture.

After `up4`, we have a single feature map at the full input resolution: a
32-channel **shared trunk** carrying both semantic and spatial information.

---

## 6. Phase 3 — Two task heads

The shared trunk now splits into two specialized heads. Each tries to predict
a different thing from the same shared features.

### 6.1 Mask head — affordance probability

| Step  | Operation               | Output                       |
|-------|-------------------------|------------------------------|
| 6.1   | 1×1 Conv (32 → 1)       | `mask_logits`  `[B, 1, 448, 448]` |

**Output type:** raw *logits* — unbounded real numbers. To convert to a
probability you apply `torch.sigmoid(mask_logits)`, which maps any real
number into (0, 1). The training loss does this internally for numerical
stability, so we do *not* apply sigmoid inside the model.

### 6.2 Normal head — 3D orientation

| Step    | Operation                                  | Output                          |
|---------|--------------------------------------------|---------------------------------|
| 6.2.a   | 3×3 Conv (32 → 32) + BatchNorm + ReLU       | `[B, 32, 448, 448]`              |
| 6.2.b   | 1×1 Conv (32 → 3)                          | raw normals `[B, 3, 448, 448]`   |
| 6.2.c   | `F.normalize(p=2, dim=1)` (L2-normalize)   | `normal_pred` (unit vectors)    |

**Why the extra 3×3 conv in the normal head?** Because surface normals need
*local* gradient-like information (how the surface tilts from one pixel to
the next), but the mask head only needs region-level semantics. Giving the
normal head a dedicated mid-resolution convolution lets it specialize
without competing for filters with the mask head.

**Why L2-normalize?** A surface normal is, by definition, a unit vector in
3D — a direction, not a magnitude. After the 1×1 conv we have an arbitrary
3-vector per pixel; dividing each by its length forces it to lie on the
unit sphere, which is the only place a valid normal can live.

### 6.3 Multi-task learning — why share the trunk?

We could train two completely separate networks. We don't, because:

- The features needed to recognize "this is a handle" are also useful for
  estimating "this is a cylindrical surface." Sharing the trunk lets both
  tasks reinforce each other.
- It's drastically more parameter-efficient: one shared decoder, two tiny
  heads.
- The auxiliary task (normals) acts as a regularizer: the model can't cheat
  on the mask by memorizing tool shapes, because it also has to be right
  about the geometry, which is unique to each viewpoint.

---

## 7. The losses (training objective)

**File:** `utils/losses.py`. The total training loss is

```
L_total = L_mask + 5.0 * L_normal + 0.5 * L_smooth
```

### 7.1 `DiceBCELoss` (the mask loss)

Two ingredients added together.

**BCEWithLogitsLoss** is the standard binary classification loss. For each
pixel it asks "should this be 1 or 0?" and penalizes a wrong guess. The
"with logits" suffix means it expects raw logits as input and combines the
sigmoid step into the math, which is much more numerically stable than
applying sigmoid yourself and then using `nn.BCELoss`.

**Dice loss** measures *shape overlap*:

```
Dice = 1 − (2 · |Pred ∩ GT| + 1) / (|Pred| + |GT| + 1)
```

It is 0 when the predicted mask perfectly overlaps the ground truth, and
approaches 1 when they don't overlap at all. The `+ 1` in numerator and
denominator is a smoothing term that avoids division by zero on
all-background batches.

**Why both?** Affordance pixels are a tiny minority — the handle covers
maybe 5% of the image, and the rest is background. If we used only BCE,
the model could score near-zero loss by simply predicting "background
everywhere." Dice cares specifically about how well the *predicted shape*
matches the ground truth shape, so it punishes that trivial solution.

### 7.2 Masked cosine loss (the normal loss)

For two unit vectors `a` and `b`, the cosine similarity is `a · b`. It
equals `+1` when they point the same direction, `0` when perpendicular,
`-1` when opposite. We use `1 − cosine_similarity` as the per-pixel loss
(so 0 is best, 2 is worst).

We then **average this loss only over GT mask pixels** — the regions a robot
will actually try to grasp. Predictions in the background are not
penalized; the robot doesn't care which way a piece of empty table is
facing.

### 7.3 Edge-aware smoothness

```
L_smooth = (|∇x N| · exp(−10 · |∇x RGB|)).mean()
         + (|∇y N| · exp(−10 · |∇y RGB|)).mean()
```

In plain English: penalize the predicted normals from differing between
neighbouring pixels — **except** where the RGB image itself has a strong
edge (where `exp(−|∇RGB|)` becomes small). The effect: predicted normals
stay smooth across flat surfaces but are allowed to discontinuously change
at object boundaries. This is a well-known trick from self-supervised
depth estimation (Godard et al., Monodepth).

### 7.4 Why those specific weights?

`w_normal = 5.0` and `w_smooth = 0.5` were picked so that all three losses
sit at comparable orders of magnitude on early epochs. Both are exposed as
CLI flags (`--w_normal`, `--w_smooth`) so you can A/B them.

---

## 8. Augmentations (only during training)

**File:** `utils/augmentations.py` → `JointTrainTransform`.

This is applied inside the dataset *before* normalization, only when
`augment=True` (set automatically by the training script unless you pass
`--no_augment`). It applies the same geometric transform to all three of
RGB, mask, and normals — *and crucially also rotates the normal vectors
themselves* in the image plane. If we rotated the image without rotating
the vectors, the supervision would become physically inconsistent: the
network would be told "look at this rotated handle, but predict normals
as if it weren't rotated."

The defaults are conservative: ±15° rotation, ±15% scale, 50% horizontal
flip (with normal x-component negated), light photometric jitter, Gaussian
noise σ = 0.01, and 25% probability random erasing. See the table in the
main README for the full list.

---

## 9. From dense maps to a 6-DoF robot pose (inference)

At test time the dense predictions are post-processed into a single robotic
target. There are four small steps:

| Step    | What you do                                                                                                       |
|---------|-------------------------------------------------------------------------------------------------------------------|
| 1       | `prob = torch.sigmoid(mask_logits)` — turn logits into probabilities                                              |
| 2       | Binarize: `mask = prob > 0.5`. Optional: keep only the largest connected component.                              |
| 3       | Compute the 2D centroid `(u, v)` of the binary mask in pixel coordinates.                                         |
| 4       | Lookup depth: `Z = depth[v, u]` in meters. Back-project: `X = (u − cx) · Z / fx`, `Y = (v − cy) · Z / fy`.        |
| 5       | Lookup the predicted normal at that pixel: `(Nx, Ny, Nz) = normal_pred[:, v, u]`.                                |

Output: a 6-DoF pose `(X, Y, Z, Nx, Ny, Nz)`. That's everything a motion
planner needs: where the target is in 3D space, and which direction the
gripper should approach from.

The camera intrinsics `(fx, fy, cx, cy)` come from
`config.INFERENCE_INTRINSICS`, which you swap per deployment camera without
touching the code.

---

## 10. Parameter count and compute

| Component                     | Parameters | Trainable? |
|-------------------------------|------------|------------|
| DINOv2 ViT-Small backbone     | ≈ 21.0 M   | no (frozen)|
| Decoder (ViT proj + cascade)  | ≈ 3.6 M    | yes         |
| Mask head + Normal head       | ≈ 0.04 M   | yes         |
| **Total trainable**           | **≈ 3.6 M**| —           |

A forward pass at batch size 8 runs in roughly 80 ms on a single recent
GPU. Backward + step is similar. So a full training epoch on UMD (a few
thousand samples) takes a couple of minutes on a 3060-class GPU.

---

## 11. End-to-end shape table (quick reference)

```
Input
  rgb                          [B, 3, 448, 448]      float32

DINOv2Backbone (frozen)
  out (list of 4)              4 × [B, 384, 32, 32]  float32

Decoder
  vit_proj (4× 1×1 Conv)       4 × [B, 256, 32, 32]
  concat                       [B, 1024, 32, 32]
  vit_fuse (ConvBlock)         [B, 256, 32, 32]

  RGBStem
    s0                         [B, 32, 448, 448]
    s1                         [B, 64, 224, 224]
    s2                         [B, 96, 112, 112]
    s3                         [B, 128, 56, 56]

  FusionUp1   + skip s3        [B, 192, 56, 56]
  FusionUp2   + skip s2        [B, 128, 112, 112]
  FusionUp3   + skip s1        [B, 64, 224, 224]
  FusionUp4   + skip s0        [B, 32, 448, 448]

Heads
  mask_head    (1×1 Conv)      mask_logits  [B, 1, 448, 448]   (logits)
  normal_head  (3×3 + 1×1)     normal_pred  [B, 3, 448, 448]   (unit vectors)
```

---

## 12. FAQ

**Q: Why is DINOv2 frozen if it would be even better tuned to my data?**
Because tuning a 21 M-parameter network on a few thousand UMD images
quickly *overfits* — the network "memorizes" the training tools and forgets
the broad world-knowledge it had. Freezing preserves the generalization.
If, after the decoder converges, you have time and data, you can
*partially* unfreeze the last 2–4 ViT blocks at a 10× lower learning rate.

**Q: What does the `B` dimension actually do in training?**
It lets us compute losses on multiple images in parallel on the GPU. Big
batches train faster and produce smoother gradient estimates; small
batches use less memory. Our default is 8, which is a reasonable
compromise on consumer GPUs.

**Q: Why 448×448 specifically? My camera is 640×480.**
The 14×32 = 448 constraint comes from DINOv2's patch size. The dataset
center-crops the 640×480 input to 448×448; for deployment you have two
choices: (a) center-crop your camera image too, or (b) pad/letterbox to a
size that is divisible by 14 and pass it through. Option (a) is simpler.

**Q: How do I know if training is overfitting?**
Look at `training_curves.png` produced by `scripts/visualize.py`. If train
loss keeps going down but val loss plateaus or rises, that's overfitting.
The text summary printed by the same script gives an explicit overfitting
flag and a "patience" number.

**Q: What should I expect for a good IoU and angular error?**
On the held-out UMD val split, the v1 baseline achieves modest numbers
(roughly IoU ≈ 0.45, angular error ≈ 30°). The current architecture
should improve both. Production-quality is closer to IoU ≥ 0.6 and
angular error ≤ 20° — that's where you can start trusting the robot to
actually grasp.

**Q: What's the difference between `last.pth` and `best.pth`?**
`last.pth` is a full checkpoint (model + optimizer state) saved after
*every* epoch, used by `--resume`. `best.pth` is the bare model state
saved only when val loss improves; it's the one you use at inference.

**Q: My GPU runs out of memory at batch size 8. What do I do?**
Drop `--batch_size 4` or `--batch_size 2`. The architecture is small but
DINOv2's attention activations at 32×32 still consume a lot of memory.
You can also enable mixed-precision (`torch.amp`) if you're comfortable
modifying the training loop.
