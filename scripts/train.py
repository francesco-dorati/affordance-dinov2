"""
train.py — Training entry point for the affordance pipeline.

Components:
  - DINOv2Backbone (frozen, multi-scale ViT features)
  - MultiTaskDecoder (multi-scale fusion + RGB skip connections + logits output)
  - DiceBCELoss + masked cosine + edge-aware normal smoothness
  - Joint augmentations (RGB + mask + normals)
  - Camera intrinsics from config.TRAIN_INTRINSICS

Outputs (all under the checkpoint directory):
  - last.pth           — checkpoint of last completed epoch (for --resume)
  - best.pth           — model state dict at best val loss
  - history.jsonl      — per-epoch metrics (train + val rows); append-only,
                         line-buffered, survives mid-epoch crashes.
                         Use scripts/visualize.py to plot it.
  - run_config.json    — the args this run was launched with.

Run:
    python scripts/train.py --epochs 25 --batch_size 8
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_TOOLS, TRAIN_INTRINSICS, AFFORDANCE_CLASSES, N_AFFORDANCE_CLASSES
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder
from utils.dataset import UMDAffordanceDataset, instance_split
from utils.losses import (
    DiceBCELoss,
    masked_cosine_loss,
    edge_aware_normal_smoothness,
    angle_error_degrees,
    iou,
    iou_per_class,
)
from utils.training_logger import JSONLLogger


# =====================================================================
# 1. CLI
# =====================================================================
def get_args():
    p = argparse.ArgumentParser("Train Affordance Decoder")
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--use_drive',  action='store_true')
    p.add_argument('--epochs',     type=int,   default=25)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--w_normal',   type=float, default=5.0)
    p.add_argument('--w_smooth',   type=float, default=0.5)
    p.add_argument('--no_augment', action='store_true')
    return p.parse_args()


# =====================================================================
# 2. Helpers
# =====================================================================
def evaluate_loader(backbone, decoder, loader, device, mask_loss_fn, w_normal):
    """Run one full pass over `loader` and return aggregated metrics."""
    decoder.eval()
    sums = dict(loss=0.0, loss_mask=0.0, loss_normal=0.0,
                iou=0.0, angle_deg=0.0, n=0)
    # Per-class IoU: sum + count separately so absent classes (None) don't
    # poison the average for a given epoch.
    iou_c_sum = [0.0] * N_AFFORDANCE_CLASSES
    iou_c_n   = [0]   * N_AFFORDANCE_CLASSES
    with torch.no_grad():
        for batch in loader:
            rgb        = batch['rgb'].to(device)
            gt_mask    = batch['mask'].to(device)
            gt_normals = batch['normals'].to(device)
            vit_feats  = backbone(rgb)
            mask_logits, pred_normals = decoder(vit_feats, rgb)

            l_mask = mask_loss_fn(mask_logits, gt_mask)
            l_norm = masked_cosine_loss(pred_normals, gt_normals, gt_mask)
            l = l_mask + w_normal * l_norm

            sums['loss']        += l.item()
            sums['loss_mask']   += l_mask.item()
            sums['loss_normal'] += l_norm.item()
            sums['iou']         += iou(mask_logits, gt_mask)
            for c, v in enumerate(iou_per_class(mask_logits, gt_mask)):
                if v is not None:
                    iou_c_sum[c] += v
                    iou_c_n[c]   += 1
            ang = angle_error_degrees(pred_normals, gt_normals, gt_mask)
            if not torch.isnan(ang):
                sums['angle_deg'] += ang.item()
            sums['n'] += 1
    n = max(sums['n'], 1)
    out = {k: (v / n if k != 'n' else v) for k, v in sums.items()}
    out['iou_per_class'] = [
        (iou_c_sum[c] / iou_c_n[c]) if iou_c_n[c] > 0 else None
        for c in range(N_AFFORDANCE_CLASSES)
    ]
    return out


# =====================================================================
# 3. Main
# =====================================================================
def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CKPT_DIR = (Path("/content/drive/MyDrive/robotic_affordance_project/checkpoints")
                if args.use_drive else PROJECT_ROOT / "checkpoints")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Checkpoints: {CKPT_DIR}")

    # Persist the run config so we know what produced this checkpoint dir.
    # Record the affordance class order so downstream tools (evaluate.py,
    # visualize.py) interpret the 7 mask channels correctly.
    run_cfg = dict(vars(args))
    run_cfg["affordance_classes"] = list(AFFORDANCE_CLASSES)
    run_cfg["n_affordance_classes"] = N_AFFORDANCE_CLASSES
    with open(CKPT_DIR / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    # ---- Data ----
    train_ds = UMDAffordanceDataset(
        raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
        augment=not args.no_augment,
    )
    val_ds = UMDAffordanceDataset(
        raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
        augment=False,
    )
    train_idx, val_idx = instance_split(train_ds, seed=42, val_frac=0.2)

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---- Model ----
    backbone = DINOv2Backbone(freeze=True).to(DEVICE)
    decoder  = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4,
                                n_classes=N_AFFORDANCE_CLASSES).to(DEVICE)

    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    mask_loss_fn = DiceBCELoss()

    # ---- Resume ----
    start_epoch, best_val = 0, float('inf')
    last_ckpt = CKPT_DIR / "last.pth"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=DEVICE)
        decoder.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optim'])
        start_epoch = ck['epoch'] + 1
        best_val = ck['best_val']
        print(f"Resumed at epoch {start_epoch}  | best val {best_val:.4f}")

    # ---- Logger ----
    logger = JSONLLogger(CKPT_DIR / "history.jsonl")
    logger.log(event="start", args=vars(args), device=DEVICE,
               n_train=len(train_idx), n_val=len(val_idx),
               start_epoch=start_epoch, target_epochs=args.epochs)

    # ---- Loop ----
    try:
        for epoch in range(start_epoch, args.epochs):
            # ---- TRAIN ----
            decoder.train()
            t0 = time.time()
            sums = dict(loss=0.0, loss_mask=0.0, loss_normal=0.0,
                        loss_smooth=0.0, iou=0.0, angle_deg=0.0, n=0)
            iou_c_sum_tr = [0.0] * N_AFFORDANCE_CLASSES
            iou_c_n_tr   = [0]   * N_AFFORDANCE_CLASSES
            pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs} TRAIN")
            for batch in pbar:
                rgb        = batch['rgb'].to(DEVICE)
                gt_mask    = batch['mask'].to(DEVICE)
                gt_normals = batch['normals'].to(DEVICE)

                optimizer.zero_grad()
                with torch.no_grad():
                    vit_feats = backbone(rgb)
                mask_logits, pred_normals = decoder(vit_feats, rgb)

                l_mask   = mask_loss_fn(mask_logits, gt_mask)
                l_norm   = masked_cosine_loss(pred_normals, gt_normals, gt_mask)
                l_smooth = edge_aware_normal_smoothness(pred_normals, rgb)
                loss = l_mask + args.w_normal * l_norm + args.w_smooth * l_smooth

                loss.backward()
                optimizer.step()

                # accumulate train-side metrics (cheap, no eval pass)
                sums['loss']        += loss.item()
                sums['loss_mask']   += l_mask.item()
                sums['loss_normal'] += l_norm.item()
                sums['loss_smooth'] += l_smooth.item()
                sums['iou']         += iou(mask_logits.detach(), gt_mask)
                for c, v in enumerate(iou_per_class(mask_logits.detach(), gt_mask)):
                    if v is not None:
                        iou_c_sum_tr[c] += v
                        iou_c_n_tr[c]   += 1
                with torch.no_grad():
                    ang = angle_error_degrees(pred_normals, gt_normals, gt_mask)
                if not torch.isnan(ang):
                    sums['angle_deg'] += ang.item()
                sums['n'] += 1
                pbar.set_postfix(L=f"{loss.item():.3f}",
                                 m=f"{l_mask.item():.3f}",
                                 n=f"{l_norm.item():.3f}",
                                 s=f"{l_smooth.item():.3f}")

            train_dur = time.time() - t0
            n = max(sums['n'], 1)
            train_metrics = {k: (v / n if k != 'n' else v) for k, v in sums.items()}
            train_iou_per_class = [
                (iou_c_sum_tr[c] / iou_c_n_tr[c]) if iou_c_n_tr[c] > 0 else None
                for c in range(N_AFFORDANCE_CLASSES)
            ]
            logger.log(epoch=epoch + 1, phase="train",
                       loss=train_metrics['loss'],
                       loss_mask=train_metrics['loss_mask'],
                       loss_normal=train_metrics['loss_normal'],
                       loss_smooth=train_metrics['loss_smooth'],
                       iou=train_metrics['iou'],
                       iou_per_class=train_iou_per_class,
                       angle_deg=train_metrics['angle_deg'],
                       lr=optimizer.param_groups[0]['lr'],
                       duration_s=train_dur,
                       n_batches=n)

            # ---- VAL ----
            t0 = time.time()
            val_metrics = evaluate_loader(backbone, decoder, val_loader,
                                          DEVICE, mask_loss_fn, args.w_normal)
            val_dur = time.time() - t0
            logger.log(epoch=epoch + 1, phase="val",
                       loss=val_metrics['loss'],
                       loss_mask=val_metrics['loss_mask'],
                       loss_normal=val_metrics['loss_normal'],
                       iou=val_metrics['iou'],
                       iou_per_class=val_metrics['iou_per_class'],
                       angle_deg=val_metrics['angle_deg'],
                       duration_s=val_dur,
                       n_batches=val_metrics['n'])

            print(f"E{epoch+1} | "
                  f"Train L {train_metrics['loss']:.4f} IoU {train_metrics['iou']:.3f} Ang {train_metrics['angle_deg']:.2f}° "
                  f"| Val L {val_metrics['loss']:.4f} IoU {val_metrics['iou']:.3f} Ang {val_metrics['angle_deg']:.2f}°")

            # ---- SAVE ----
            torch.save({'epoch': epoch, 'model': decoder.state_dict(),
                        'optim': optimizer.state_dict(), 'best_val': best_val},
                       last_ckpt)
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                torch.save(decoder.state_dict(), CKPT_DIR / "best.pth")
                logger.log(event="best", epoch=epoch + 1,
                           val_loss=val_metrics['loss'],
                           val_iou=val_metrics['iou'],
                           val_angle_deg=val_metrics['angle_deg'])
                print("   new best saved")

        logger.log(event="end", epochs_completed=args.epochs,
                   best_val_loss=best_val)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
