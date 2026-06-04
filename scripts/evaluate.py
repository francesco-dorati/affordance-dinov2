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

from config import (
    RAW_TOOLS,
    TRAIN_INTRINSICS,
    AFFORDANCE_CLASSES,
    N_AFFORDANCE_CLASSES,
)
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


def batch_iou_per_class(logits, target, thresh):
    """Per-sample per-class IoU. Returns array [B, C] with NaN for absent classes.

    A class is "absent" for a sample when both prediction and target have zero
    positive pixels for that channel — IoU is undefined there, so we use NaN
    and let `_avg(..., drop_nan=True)` skip it during aggregation.
    """
    pred = (torch.sigmoid(logits) > thresh).float()
    dims = (2, 3)  # spatial only — keep per-class
    inter = (pred * target).sum(dim=dims)            # [B, C]
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter / union.clamp(min=1e-6)).cpu().numpy()
    # Mark absent classes (union == 0) as NaN.
    absent = (union == 0).cpu().numpy()
    iou[absent] = float('nan')
    return iou  # ndarray [B, C]


def batch_angle_stats(pred_normals, gt_normals, gt_mask):
    """
    For each sample in the batch, return:
        mean_angle_deg, frac_le_11_25, frac_le_22_5, frac_le_30
    over the UNION of all affordance pixels for that sample. Samples whose
    full multi-class mask is empty (no annotated affordance pixels) return NaN.
    """
    p = F.normalize(pred_normals, p=2, dim=1)
    g = F.normalize(gt_normals,   p=2, dim=1)
    cos = (p * g).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    deg = torch.acos(cos) * (180.0 / np.pi)               # [B, H, W]
    active = gt_mask.sum(dim=1) > 0                       # [B, H, W]

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
    decoder  = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4,
                                n_classes=N_AFFORDANCE_CLASSES).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # Accept either a bare state_dict (best.pth) or a full checkpoint (last.pth)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    decoder.load_state_dict(state)
    decoder.eval()

    # ---- Collect per-sample metrics ----
    # Per-sample dicts hold mean-IoU at each threshold (averaged across the
    # affordance classes that are present for that sample) plus angular stats.
    # Per-class per-sample IoU arrays are kept separately for class-wise
    # aggregation later.
    per_sample = []   # list of dicts
    per_sample_iou_class = {t: [] for t in IOU_THRESHOLDS}  # list of np.ndarray [C]
    per_sample_tool = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            rgb        = batch['rgb'].to(DEVICE)
            gt_mask    = batch['mask'].to(DEVICE)
            gt_normals = batch['normals'].to(DEVICE)
            tools      = batch['tool_name']

            vit_feats = backbone(rgb)
            mask_logits, pred_normals = decoder(vit_feats, rgb)

            iou_class_per_thresh = {
                t: batch_iou_per_class(mask_logits, gt_mask, t)  # [B, C]
                for t in IOU_THRESHOLDS
            }
            angle_rows = batch_angle_stats(pred_normals, gt_normals, gt_mask)

            B = rgb.shape[0]
            for i in range(B):
                rec = {'tool': tools[i]}
                for t in IOU_THRESHOLDS:
                    sample_class_iou = iou_class_per_thresh[t][i]  # [C]
                    # Sample-level mean: average over classes present (non-NaN).
                    valid = sample_class_iou[~np.isnan(sample_class_iou)]
                    rec[f'iou@{t:.1f}'] = float(valid.mean()) if valid.size else float('nan')
                    per_sample_iou_class[t].append(sample_class_iou)
                ang = angle_rows[i]
                rec['angle_deg_mean'] = ang[0]
                for j, b in enumerate(ANGLE_BINS_DEG):
                    rec[f'frac_le_{b}'] = ang[1 + j]
                per_sample.append(rec)
                per_sample_tool.append(tools[i])

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

    # ---- Aggregate (per affordance class, overall) ----
    # Stack into [N, C] then nan-mean over samples.
    per_class_overall = {}
    for t in IOU_THRESHOLDS:
        arr = np.stack(per_sample_iou_class[t], axis=0)  # [N, C]
        with np.errstate(all='ignore'):
            means = np.nanmean(arr, axis=0).tolist()
        per_class_overall[f'iou@{t:.1f}'] = {
            cls: (None if (isinstance(m, float) and np.isnan(m)) else float(m))
            for cls, m in zip(AFFORDANCE_CLASSES, means)
        }

    # ---- Aggregate (per tool) ----
    per_tool = defaultdict(list)
    for r in per_sample:
        per_tool[r['tool']].append(r)
    # Per-tool per-class IoU arrays (sample indices grouped by tool)
    tool_to_indices = defaultdict(list)
    for idx, t in enumerate(per_sample_tool):
        tool_to_indices[t].append(idx)
    per_tool_summary = {}
    for tool, rows in per_tool.items():
        agg = {'n_samples': len(rows)}
        for t in IOU_THRESHOLDS:
            agg[f'iou@{t:.1f}'] = float(np.mean([r[f'iou@{t:.1f}'] for r in rows
                                                 if not np.isnan(r[f'iou@{t:.1f}'])]))
        ang_vals = [r['angle_deg_mean'] for r in rows
                    if not (isinstance(r['angle_deg_mean'], float)
                            and np.isnan(r['angle_deg_mean']))]
        agg['angle_deg_mean'] = float(np.mean(ang_vals)) if ang_vals else None
        # Per-class IoU at 0.5 for this tool.
        idxs = tool_to_indices[tool]
        arr = np.stack([per_sample_iou_class[0.5][i] for i in idxs], axis=0)
        with np.errstate(all='ignore'):
            means = np.nanmean(arr, axis=0).tolist()
        agg['iou@0.5_per_class'] = {
            cls: (None if (isinstance(m, float) and np.isnan(m)) else float(m))
            for cls, m in zip(AFFORDANCE_CLASSES, means)
        }
        per_tool_summary[tool] = agg

    # ---- Write report ----
    report = {
        'checkpoint': str(ckpt_path),
        'split': args.split,
        'n_samples': len(per_sample),
        'n_tools': len(per_tool_summary),
        'n_affordance_classes': N_AFFORDANCE_CLASSES,
        'affordance_classes': list(AFFORDANCE_CLASSES),
        'elapsed_s': elapsed,
        'overall': overall,
        'per_class_overall': per_class_overall,
        'per_tool': per_tool_summary,
    }
    out_path = out_dir / f"evaluation_{args.split}.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ---- Console summary ----
    print("\n=== OVERALL (mean-IoU across affordance classes) ===")
    for t in IOU_THRESHOLDS:
        print(f"  IoU @ {t:.1f} : {overall[f'iou@{t:.1f}']:.4f}")
    print(f"  Mean angular error : {overall['angle_deg_mean']:.2f}°")
    for b in ANGLE_BINS_DEG:
        print(f"  Fraction <= {b:>5.2f}° : {overall[f'frac_le_{b}']:.3f}")

    print("\n=== PER AFFORDANCE CLASS (IoU @ 0.5) ===")
    for cls in AFFORDANCE_CLASSES:
        v = per_class_overall['iou@0.5'][cls]
        v_str = "  n/a" if v is None else f"{v:.4f}"
        print(f"  {cls:12s} : {v_str}")
    print(f"\nFull report written to: {out_path}")


if __name__ == "__main__":
    main()
