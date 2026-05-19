"""
train.py — Training entry point for the affordance pipeline.

Components:
  - DINOv2Backbone (frozen, multi-scale ViT features)
  - MultiTaskDecoder (multi-scale fusion + RGB skip connections + logits output)
  - DiceBCELoss + masked cosine + edge-aware normal smoothness
  - Joint augmentations (RGB + mask + normals, with vector-correct rotation/flip)
  - Camera intrinsics from config.TRAIN_INTRINSICS

Run:
    python scripts/train.py --epochs 25 --batch_size 8
"""

import sys
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

from config import RAW_TOOLS, TRAIN_INTRINSICS
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder
from utils.dataset import UMDAffordanceDataset
from utils.losses import (
    DiceBCELoss,
    masked_cosine_loss,
    edge_aware_normal_smoothness,
    angle_error_degrees,
    iou,
)


# =====================================================================
# 1. CLI
# =====================================================================
def get_args():
    p = argparse.ArgumentParser("Train Affordance Decoder v2")
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
# 2. Main
# =====================================================================
def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CKPT_DIR = (Path("/content/drive/MyDrive/robotic_affordance_project/checkpoints")
                if args.use_drive else PROJECT_ROOT / "checkpoints")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Checkpoints: {CKPT_DIR}")

    # ---- Data ----
    train_ds = UMDAffordanceDataset(
        raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
        augment=not args.no_augment,
    )
    val_ds = UMDAffordanceDataset(
        raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
        augment=False,
    )

    all_tools = sorted({s[0] for s in train_ds.samples})
    np.random.seed(42); np.random.shuffle(all_tools)
    split = int(0.8 * len(all_tools))
    train_set = set(all_tools[:split])
    train_idx = [i for i, s in enumerate(train_ds.samples) if s[0]     in train_set]
    val_idx   = [i for i, s in enumerate(val_ds.samples)   if s[0] not in train_set]

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---- Model ----
    backbone = DINOv2Backbone(freeze=True).to(DEVICE)
    decoder  = MultiTaskDecoder(embed_dim=backbone.embed_dim, n_vit_scales=4).to(DEVICE)

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

    # ---- Loop ----
    for epoch in range(start_epoch, args.epochs):
        # --- train ---
        decoder.train()
        train_loss_sum = 0.0
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
            train_loss_sum += loss.item()
            pbar.set_postfix(L=f"{loss.item():.3f}",
                             m=f"{l_mask.item():.3f}",
                             n=f"{l_norm.item():.3f}",
                             s=f"{l_smooth.item():.3f}")

        # --- val ---
        decoder.eval()
        val_loss_sum = 0.0
        ang_sum, iou_sum, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                rgb        = batch['rgb'].to(DEVICE)
                gt_mask    = batch['mask'].to(DEVICE)
                gt_normals = batch['normals'].to(DEVICE)
                vit_feats  = backbone(rgb)
                mask_logits, pred_normals = decoder(vit_feats, rgb)

                l = (mask_loss_fn(mask_logits, gt_mask)
                     + args.w_normal * masked_cosine_loss(pred_normals, gt_normals, gt_mask))
                val_loss_sum += l.item()

                ang = angle_error_degrees(pred_normals, gt_normals, gt_mask)
                if not torch.isnan(ang):
                    ang_sum += ang.item()
                    iou_sum += iou(mask_logits, gt_mask)
                    n_batches += 1

        avg_val = val_loss_sum / max(len(val_loader), 1)
        avg_ang = ang_sum / max(n_batches, 1)
        avg_iou = iou_sum / max(n_batches, 1)
        print(f"E{epoch+1} | Train {train_loss_sum/len(train_loader):.4f}  "
              f"| Val {avg_val:.4f}  | AngErr {avg_ang:.2f}°  | IoU {avg_iou:.3f}")

        # --- save ---
        torch.save({'epoch': epoch, 'model': decoder.state_dict(),
                    'optim': optimizer.state_dict(), 'best_val': best_val},
                   last_ckpt)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(decoder.state_dict(), CKPT_DIR / "best.pth")
            print("   new best saved")


if __name__ == "__main__":
    main()
