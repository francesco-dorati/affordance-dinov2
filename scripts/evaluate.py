"""
evaluate.py — Detailed post-hoc evaluation of a trained checkpoint.

Loads a checkpoint, runs through a split, and produces a JSON report with:
  - IoU at multiple thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
  - Mean angular error (degrees) over GT mask pixels
  - Angular error percentile bins:
      fraction of pixels with angular error <= 11.25°, <= 22.5°, <= 30°
      (these are the standard surface-normal evaluation bins used in NYUv2)
  - Per-tool breakdown of mean IoU and mean angle
  - Total sample count, elapsed wall time

Usage:
    # Default: evaluate best.pth on the held-out val tools
    python scripts/evaluate.py

    # Evaluate a specific checkpoint on the full dataset
    python scripts/evaluate.py --checkpoint checkpoints/last.pth --split all

    # Write the JSON elsewhere
    python scripts/evaluate.py --output_dir reports/run_2026_05_21
"""

import sys
import json
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_TOOLS, TRAIN_INTRINSICS
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder
from utils.dataset import UMDAffordanceDataset, instance_split


# =====================================================================
# 1. CLI
# =====================================================================
def get_args():
    p = argparse.ArgumentParser("Evaluate Affordance Checkpoint")
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--split',      type=str, default='val',
                   choices=['val', 'train', 'all'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output_dir', type=str, default=None,
                   help='Where to write evaluation_<split>.json '
                        '(default: same directory as the checkpoint)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_frac', type=float, default=0.2)
    return p.parse_args()


# =====================================================================
# 2. Metric helpers (vectorized over a batch)
# =====================================================================
IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
ANGLE_BINS_DEG = [11.25, 22.5, 30.0]


def batch_iou(logits, target, thresh):
    """Per-sample IoU (returns list of floats, length B)."""
    pred = (torch.sigmoid(logits) > thresh).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter / (union + 1e-6)).cpu().numpy()
    return iou.tolist()


def batch_angle_stats(pred_normals, gt_normals, gt_mask):
    """
    For each sample in the batch, return:
        mean_angle_deg, frac_le_11_25, frac_le_22_5, frac_le_30
    over the GT mask pixels of that sample. Samples with empty masks return NaN.
    """
    p = F.normalize(pred_normals, p=2, dim=1)
    g = F.normalize(gt_normals,   p=2, dim=1)
    cos = (p * g).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    deg = torch.acos(cos) * (180.0 / np.pi)               # [B, H, W]
    active = (gt_mask > 0).squeeze(1)                     # [B, H, W]

    B = deg.shape[0]
    out = []
    for b in range(B):
        m = active[b]
        if m.sum() == 0:
            out.append((float('nan'),) * (1 + len(ANGLE_BINS_DEG)))
            continue
        d = deg[b][m]
        row = [d.mean().item()]
        for t in ANGLE_BINS_DEG:
            row.append((d <= t).float().mean().item())
        out.append(tuple(row))
    return out


# =====================================================================
# 3. Main
# =====================================================================
def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE} | Checkpoint: {ckpt_path}")

    # ---- Data ----
    ds = UMDAffordanceDataset(
        raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS, augment=False,
    )
    train_idx, val_idx = instance_split(ds, seed=args.seed, val_frac=args.val_frac)
    if args.split == 'val':
        indices = val_idx
    elif args.split == 'train':
        indices = train_idx
    else:
        indices = list(range(len(ds)))
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    print(f"Split: {args.split} | n_samples: {len(indices)}")

    # ---- Model ----
    backbone = DINOv2Backbone(freeze=True).to(DEVICE).eval()
    decoder  = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # Accept either a bare state_dict (best.pth) or a full checkpoint (last.pth)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    decoder.load_state_dict(state)
    decoder.eval()

    # ---- Collect per-sample metrics ----
    per_sample = []   # list of dicts
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            rgb        = batch['rgb'].to(DEVICE)
            gt_mask    = batch['mask'].to(DEVICE)
            gt_normals = batch['normals'].to(DEVICE)
            tools      = batch['tool_name']

            vit_feats = backbone(rgb)
            mask_logits, pred_normals = decoder(vit_feats, rgb)

            ious_per_thresh = {
                t: batch_iou(mask_logits, gt_mask, t) for t in IOU_THRESHOLDS
            }
            angle_rows = batch_angle_stats(pred_normals, gt_normals, gt_mask)

            for i in range(rgb.shape[0]):
                rec = {'tool': tools[i]}
                for t in IOU_THRESHOLDS:
                    rec[f'iou@{t:.1f}'] = ious_per_thresh[t][i]
                ang = angle_rows[i]
                rec['angle_deg_mean'] = ang[0]
                for j, b in enumerate(ANGLE_BINS_DEG):
                    rec[f'frac_le_{b}'] = ang[1 + j]
                per_sample.append(rec)

    elapsed = time.time() - t0

    # ---- Aggregate (overall) ----
    def _avg(key, drop_nan=True):
        vals = [r[key] for r in per_sample]
        if drop_nan:
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return None
        return float(np.mean(vals))

    overall = {f'iou@{t:.1f}': _avg(f'iou@{t:.1f}') for t in IOU_THRESHOLDS}
    overall['angle_deg_mean'] = _avg('angle_deg_mean')
    for b in ANGLE_BINS_DEG:
        overall[f'frac_le_{b}'] = _avg(f'frac_le_{b}')

    # ---- Aggregate (per tool) ----
    per_tool = defaultdict(list)
    for r in per_sample:
        per_tool[r['tool']].append(r)
    per_tool_summary = {}
    for tool, rows in per_tool.items():
        agg = {'n_samples': len(rows)}
        for t in IOU_THRESHOLDS:
            agg[f'iou@{t:.1f}'] = float(np.mean([r[f'iou@{t:.1f}'] for r in rows]))
        ang_vals = [r['angle_deg_mean'] for r in rows
                    if not (isinstance(r['angle_deg_mean'], float)
                            and np.isnan(r['angle_deg_mean']))]
        agg['angle_deg_mean'] = float(np.mean(ang_vals)) if ang_vals else None
        per_tool_summary[tool] = agg

    # ---- Write report ----
    report = {
        'checkpoint': str(ckpt_path),
        'split': args.split,
        'n_samples': len(per_sample),
        'n_tools': len(per_tool_summary),
        'elapsed_s': elapsed,
        'overall': overall,
        'per_tool': per_tool_summary,
    }
    out_path = out_dir / f"evaluation_{args.split}.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ---- Console summary ----
    print("\n=== OVERALL ===")
    for t in IOU_THRESHOLDS:
        print(f"  IoU @ {t:.1f} : {overall[f'iou@{t:.1f}']:.4f}")
    print(f"  Mean angular error : {overall['angle_deg_mean']:.2f}°")
    for b in ANGLE_BINS_DEG:
        print(f"  Fraction <= {b:>5.2f}° : {overall[f'frac_le_{b}']:.3f}")
    print(f"\nFull report written to: {out_path}")


if __name__ == "__main__":
    main()
