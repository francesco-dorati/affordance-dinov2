"""
visualize.py — Plot training history and optionally dump sample predictions.

WHAT YOU GET:
  1. training_curves.png  — 2x2 grid:
       (a) Train vs Val loss with best-epoch marker
       (b) Train vs Val mean-IoU (averaged across the 7 affordance classes)
       (c) Train vs Val angular error (degrees)
       (d) Component losses (mask, normal, smoothness) on the train side
     A short text summary is printed and also saved next to the PNG.

  2. (optional, with --checkpoint) samples/000_<tool>.png ... — one PNG per
     sample showing RGB, GT vs predicted multi-class affordance overlay (one
     color per UMD class), GT vs predicted normals, and one row of per-class
     predicted heatmaps — for visual sanity-checking of the multi-class model.

WHAT TO LOOK FOR — OVERFITTING:
  - Healthy training: train and val curves stay close; both keep improving.
  - Mild overfitting: train keeps improving, val plateaus.
  - Severe overfitting: train keeps improving, val gets WORSE.
  The summary prints the "patience" — how many epochs since val loss last
  improved. If that number is large and train is still decreasing, you are
  overfitting.

Usage:
    # Just plot the curves
    python scripts/visualize.py --history checkpoints/history.jsonl

    # Plot AND dump 8 prediction grids from the best checkpoint
    python scripts/visualize.py --history checkpoints/history.jsonl \\
        --checkpoint checkpoints/best.pth --n_samples 8

    # Choose where the output goes
    python scripts/visualize.py --history checkpoints/history.jsonl \\
        --output_dir reports/run_2026_05_21
"""

import sys
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.training_logger import load_history


# =====================================================================
# 1. CLI
# =====================================================================
def get_args():
    p = argparse.ArgumentParser("Visualize training")
    p.add_argument('--history',    type=str, default='checkpoints/history.jsonl')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='If given (with --n_samples > 0), dump sample prediction grids.')
    p.add_argument('--n_samples',  type=int, default=0,
                   help='How many sample prediction grids to dump.')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Where to write outputs (default: history file directory).')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed for sample selection.')
    return p.parse_args()


# =====================================================================
# 2. History → arrays per phase
# =====================================================================
def to_series(records):
    """Group epoch records by phase: returns {'train': {key: list}, 'val': ...}."""
    out = {'train': defaultdict(list), 'val': defaultdict(list)}
    for r in records:
        phase = r.get('phase')
        if phase not in out:
            continue
        for k, v in r.items():
            if k in ('phase', 'timestamp', 'n_batches'):
                continue
            out[phase][k].append(v)
    return out


def best_val_epoch(records):
    """Returns (best_epoch, best_loss) by minimum val loss; (None, None) if absent."""
    candidates = [(r.get('epoch'), r.get('loss')) for r in records
                  if r.get('phase') == 'val' and r.get('loss') is not None]
    if not candidates:
        return None, None
    return min(candidates, key=lambda x: x[1])


def summarize(records):
    """Build a short text summary."""
    val_records = [r for r in records if r.get('phase') == 'val']
    train_records = [r for r in records if r.get('phase') == 'train']
    if not val_records:
        return "No val records found in history."

    best_ep, best_loss = best_val_epoch(records)
    last_val = val_records[-1]
    last_epoch = last_val.get('epoch')
    patience = last_epoch - best_ep if (last_epoch and best_ep) else 0

    # overfitting heuristic: train improving but val flat-or-worsening
    if len(train_records) >= 3 and len(val_records) >= 3:
        train_recent = [r['loss'] for r in train_records[-3:] if r.get('loss')]
        val_recent   = [r['loss'] for r in val_records[-3:]   if r.get('loss')]
        train_trend = train_recent[-1] - train_recent[0] if len(train_recent) == 3 else 0
        val_trend   = val_recent[-1]   - val_recent[0]   if len(val_recent)   == 3 else 0
        overfit_flag = (train_trend < 0 and val_trend >= 0)
    else:
        train_trend = val_trend = 0
        overfit_flag = False

    last_train_loss = train_records[-1].get('loss') if train_records else float('nan')
    last_val_loss   = last_val.get('loss', float('nan'))
    gap = last_val_loss - last_train_loss

    lines = [
        f"Epochs completed   : {last_epoch}",
        f"Best val epoch     : {best_ep}    (val loss {best_loss:.4f})",
        f"Last val loss      : {last_val_loss:.4f}",
        f"Last val IoU       : {last_val.get('iou', float('nan')):.4f}",
        f"Last val angle err : {last_val.get('angle_deg', float('nan')):.2f}°",
        f"Train-Val gap      : {gap:+.4f}   (val - train at last epoch)",
        f"Patience           : {patience} epochs since best val",
        f"Overfitting flag   : {'YES — train improving, val flat/worse' if overfit_flag else 'no'}",
    ]
    return "\n".join(lines)


# =====================================================================
# 3. Plot
# =====================================================================
def plot_history(records, out_path):
    series = to_series(records)
    best_ep, _ = best_val_epoch(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _plot(ax, key, title, ylabel):
        for phase, color in [('train', 'tab:blue'), ('val', 'tab:orange')]:
            s = series[phase]
            if key in s and 'epoch' in s and s[key]:
                ax.plot(s['epoch'], s[key], color=color, marker='o',
                        markersize=3, label=phase)
        if best_ep is not None:
            ax.axvline(best_ep, color='tab:green', linestyle='--', alpha=0.6,
                       label=f'best epoch ({best_ep})')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    _plot(axes[0, 0], 'loss',      'Loss',              'loss')
    _plot(axes[0, 1], 'iou',       'IoU',               'IoU')
    _plot(axes[1, 0], 'angle_deg', 'Angular Error',     'degrees')

    # Component losses on train side
    ax = axes[1, 1]
    s = series['train']
    for key, color in [('loss_mask',   'tab:red'),
                       ('loss_normal', 'tab:purple'),
                       ('loss_smooth', 'tab:gray')]:
        if key in s and s[key]:
            ax.plot(s['epoch'], s[key], color=color, marker='o',
                    markersize=3, label=key)
    if best_ep is not None:
        ax.axvline(best_ep, color='tab:green', linestyle='--', alpha=0.6)
    ax.set_title('Train component losses')
    ax.set_xlabel('epoch')
    ax.set_ylabel('component loss')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Training history — {Path(out_path).parent.name}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 4. (Optional) sample prediction grids
# =====================================================================
# Distinct colors for the 7 UMD affordance classes (+ index 0 = background).
# Stable across the codebase so notebook and script visualizations match.
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


def _multihot_to_classmap(multihot: np.ndarray) -> np.ndarray:
    """(C, H, W) multi-hot → (H, W) class indices (0=background, 1..C=class).

    Argmax over channels gives the dominant class; pixels where ALL channels
    are zero stay at background (0). Used for the GT visualisation.
    """
    has_any = multihot.sum(axis=0) > 0
    argmax = multihot.argmax(axis=0) + 1  # shift so channel 0 → class 1
    out = np.where(has_any, argmax, 0).astype(np.int32)
    return out


def _probs_to_classmap(probs: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """(C, H, W) probabilities → (H, W) class indices using argmax + threshold."""
    has_any = probs.max(axis=0) > thresh
    argmax = probs.argmax(axis=0) + 1
    out = np.where(has_any, argmax, 0).astype(np.int32)
    return out


def dump_samples(checkpoint_path, output_dir, n_samples, seed):
    """Load model, pick n_samples random val items, save side-by-side grids."""
    import torch
    import matplotlib.colors as mcolors
    from torch.utils.data import Subset

    from config import (
        RAW_TOOLS, TRAIN_INTRINSICS,
        AFFORDANCE_CLASSES, N_AFFORDANCE_CLASSES,
    )
    from models.backbone import DINOv2Backbone
    from models.decoder import MultiTaskDecoder
    from utils.dataset import UMDAffordanceDataset, instance_split

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UMDAffordanceDataset(raw_dir=RAW_TOOLS,
                              intrinsics=TRAIN_INTRINSICS, augment=False)
    _, val_idx = instance_split(ds, seed=42, val_frac=0.2)
    rng = random.Random(seed)
    picks = rng.sample(val_idx, k=min(n_samples, len(val_idx)))

    backbone = DINOv2Backbone(freeze=True).to(DEVICE).eval()
    decoder  = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4,
                                n_classes=N_AFFORDANCE_CLASSES).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    decoder.load_state_dict(state)
    decoder.eval()

    # Denormalization constants (same as ImageNet used in dataset)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    cmap = mcolors.ListedColormap(_AFFORDANCE_PALETTE)
    norm_cmap = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, len(_AFFORDANCE_PALETTE) + 0.5), ncolors=cmap.N
    )

    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, idx in enumerate(picks):
            item = ds[idx]
            rgb = item['rgb'].unsqueeze(0).to(DEVICE)
            gt_mask    = item['mask'].cpu().numpy()          # (C, H, W)
            gt_normals = item['normals']
            tool = item['tool_name']

            vit_feats = backbone(rgb)
            mask_logits, pred_normals = decoder(vit_feats, rgb)
            pred_prob = torch.sigmoid(mask_logits)[0].cpu().numpy()   # (C, H, W)
            pred_norm = pred_normals[0].cpu().numpy().transpose(1, 2, 0)

            rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
            rgb_np = (rgb_np * std + mean).clip(0, 1)

            gt_norm_np = gt_normals.cpu().numpy().transpose(1, 2, 0)
            gt_norm_vis   = ((gt_norm_np  + 1) / 2).clip(0, 1)
            pred_norm_vis = ((pred_norm   + 1) / 2).clip(0, 1)

            gt_cls   = _multihot_to_classmap(gt_mask)
            pred_cls = _probs_to_classmap(pred_prob, thresh=0.5)

            # 3-row figure: row 0 summary (5 panels), row 1 per-class GT (7),
            # row 2 per-class predicted (7).
            fig, axes = plt.subplots(3, 7, figsize=(22, 10))

            # Row 0: hide unused columns
            for c in range(5, 7):
                axes[0, c].axis('off')

            axes[0, 0].imshow(rgb_np);                                 axes[0, 0].set_title('RGB')
            axes[0, 1].imshow(rgb_np)
            axes[0, 1].imshow(gt_cls, cmap=cmap, norm=norm_cmap, alpha=0.55)
            axes[0, 1].set_title('GT affordances')
            axes[0, 2].imshow(rgb_np)
            axes[0, 2].imshow(pred_cls, cmap=cmap, norm=norm_cmap, alpha=0.55)
            axes[0, 2].set_title('Pred affordances')
            axes[0, 3].imshow(gt_norm_vis);                            axes[0, 3].set_title('GT normals')
            axes[0, 4].imshow(pred_norm_vis);                          axes[0, 4].set_title('Pred normals')
            for c in range(5):
                axes[0, c].axis('off')

            # Row 1: per-class GT heatmaps
            for c in range(N_AFFORDANCE_CLASSES):
                axes[1, c].imshow(gt_mask[c], cmap='Reds', vmin=0, vmax=1)
                axes[1, c].set_title(f'GT {AFFORDANCE_CLASSES[c]}', fontsize=9)
                axes[1, c].axis('off')

            # Row 2: per-class predicted heatmaps
            for c in range(N_AFFORDANCE_CLASSES):
                axes[2, c].imshow(pred_prob[c], cmap='Reds', vmin=0, vmax=1)
                axes[2, c].set_title(f'Pred {AFFORDANCE_CLASSES[c]}', fontsize=9)
                axes[2, c].axis('off')

            fig.suptitle(f"{i:03d}  tool={tool}", fontsize=12)
            fig.tight_layout()
            fig.savefig(samples_dir / f"{i:03d}_{tool}.png", dpi=110,
                        bbox_inches='tight')
            plt.close(fig)

    print(f"Wrote {len(picks)} sample grids to {samples_dir}")


# =====================================================================
# 5. Main
# =====================================================================
def main():
    args = get_args()
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"No history found at {history_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else history_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_history(history_path)
    print(f"Loaded {len(records)} records from {history_path}")

    # ---- Curves ----
    curves_png = out_dir / "training_curves.png"
    plot_history(records, curves_png)
    print(f"Wrote curves to {curves_png}")

    # ---- Summary ----
    summary = summarize(records)
    print("\n" + summary + "\n")
    (out_dir / "training_summary.txt").write_text(summary + "\n")

    # ---- Optional: sample predictions ----
    if args.checkpoint and args.n_samples > 0:
        dump_samples(args.checkpoint, out_dir, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
