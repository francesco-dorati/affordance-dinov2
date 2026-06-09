"""
predict.py — Run the trained checkpoint on a folder of RGB images.

No ground truth needed. Designed for phone photos, in-the-wild captures, or
any natural images you want to visualize predictions on. The model predicts
all 7 affordance channels plus surface normals from RGB alone.

Inputs are center-cropped to a square and resized to 448x448 to match the
training distribution; rectangular phone photos work fine.

Usage:
    python scripts/predict.py \\
        --input_dir   data/in_the_wild \\
        --checkpoint  checkpoints/best.pth \\
        --output_dir  reports/in_the_wild_predictions
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import AFFORDANCE_CLASSES, N_AFFORDANCE_CLASSES
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder


# Stable color palette shared with scripts/visualize.py and the data
# exploration notebook. Index 0 is background.
_AFFORDANCE_PALETTE = [
    "#000000",  # 0 background
    "#e41a1c",  # 1 grasp        — red
    "#377eb8",  # 2 cut          — blue
    "#4daf4a",  # 3 scoop        — green
    "#984ea3",  # 4 contain      — purple
    "#ff7f00",  # 5 pound        — orange
    "#ffff33",  # 6 support      — yellow
    "#a65628",  # 7 wrap-grasp   — brown
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE = 448


# =====================================================================
# 1. Image loading
# =====================================================================
def load_and_prepare(path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Read an RGB image and center-crop+resize to (target_size, target_size, 3).

    Uses Python's open() + cv2.imdecode for cross-platform robustness
    (avoids the macOS DataLoader-fork issue).
    """
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not decode {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]

    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)
    return img


# =====================================================================
# 2. Model loading
# =====================================================================
def load_model(checkpoint_path: str, device: str):
    """Return (backbone, decoder, imagenet-normalize)."""
    backbone = DINOv2Backbone(freeze=True).to(device).eval()
    decoder = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4,
                               n_classes=N_AFFORDANCE_CLASSES).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    decoder.load_state_dict(state)
    decoder.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return backbone, decoder, normalize


# =====================================================================
# 3. Inference
# =====================================================================
@torch.no_grad()
def predict_image(rgb_np: np.ndarray, backbone, decoder, device,
                  normalize) -> tuple:
    """Run forward pass on a single RGB image.

    Returns:
        pred_prob:  (C, H, W) float32 per-class sigmoid probabilities
        pred_norm:  (H, W, 3) float32 predicted unit-vector normals
    """
    rgb_t = torch.from_numpy(rgb_np.copy()).float().permute(2, 0, 1) / 255.0
    rgb_t = normalize(rgb_t).unsqueeze(0).to(device)
    vit_feats = backbone(rgb_t)
    mask_logits, pred_normals = decoder(vit_feats, rgb_t)
    pred_prob = torch.sigmoid(mask_logits)[0].cpu().numpy()
    pred_norm = pred_normals[0].cpu().numpy().transpose(1, 2, 0)
    return pred_prob, pred_norm


# =====================================================================
# 4. Visualization
# =====================================================================
def _probs_to_classmap(probs: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    has_any = probs.max(axis=0) > thresh
    argmax = probs.argmax(axis=0) + 1
    return np.where(has_any, argmax, 0).astype(np.int32)


def render_prediction(rgb_np: np.ndarray, pred_prob: np.ndarray,
                      pred_norm: np.ndarray, title: str,
                      out_path: Path, thresh: float = 0.5) -> None:
    """2-row figure: summary row + per-class heatmap row.

    No GT row (we don't have GT for in-the-wild images).
    """
    pred_cls = _probs_to_classmap(pred_prob, thresh=thresh)
    pred_norm_vis = ((pred_norm + 1) / 2).clip(0, 1)

    cmap = mcolors.ListedColormap(_AFFORDANCE_PALETTE)
    norm_cmap = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, len(_AFFORDANCE_PALETTE) + 0.5),
        ncolors=cmap.N,
    )

    fig, axes = plt.subplots(2, 7, figsize=(22, 7))

    # Row 0: RGB | predicted affordance overlay | predicted normals | hidden
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB", fontsize=11); axes[0, 0].axis("off")
    axes[0, 1].imshow(rgb_np)
    axes[0, 1].imshow(pred_cls, cmap=cmap, norm=norm_cmap, alpha=0.55)
    axes[0, 1].set_title(f"Predicted affordances (>{thresh})", fontsize=11)
    axes[0, 1].axis("off")
    axes[0, 2].imshow(pred_norm_vis)
    axes[0, 2].set_title("Predicted normals", fontsize=11); axes[0, 2].axis("off")
    for c in range(3, 7):
        axes[0, c].axis("off")

    # Row 1: per-class predicted heatmaps
    for c in range(N_AFFORDANCE_CLASSES):
        axes[1, c].imshow(pred_prob[c], cmap="Reds", vmin=0, vmax=1)
        axes[1, c].set_title(AFFORDANCE_CLASSES[c], fontsize=10)
        axes[1, c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5. CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser("Predict affordances on in-the-wild RGB images")
    p.add_argument("--input_dir", required=True,
                   help="Folder of RGB images (jpg / png / etc).")
    p.add_argument("--checkpoint", default="checkpoints/best.pth",
                   help="Trained model checkpoint.")
    p.add_argument("--output_dir", default="reports/in_the_wild_predictions",
                   help="Where to write one PNG per input image.")
    p.add_argument("--thresh", type=float, default=0.5,
                   help="Sigmoid threshold for the argmax classmap overlay.")
    args = p.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE} | Checkpoint: {args.checkpoint}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )
    if not images:
        raise SystemExit(
            f"No images with extensions {sorted(IMG_EXTS)} found in {input_dir}"
        )
    print(f"Found {len(images)} images")

    backbone, decoder, normalize = load_model(args.checkpoint, DEVICE)

    for img_path in images:
        try:
            rgb_np = load_and_prepare(img_path)
            pred_prob, pred_norm = predict_image(
                rgb_np, backbone, decoder, DEVICE, normalize
            )
            out_path = output_dir / f"{img_path.stem}_prediction.png"
            render_prediction(rgb_np, pred_prob, pred_norm,
                              img_path.stem, out_path, thresh=args.thresh)
            print(f"  OK   {img_path.name}  ->  {out_path.name}")
        except Exception as e:
            print(f"  FAIL {img_path.name}  ({e})")

    print(f"\nDone. Output in {output_dir}")


if __name__ == "__main__":
    main()
