"""
train_baseline.py — Standard semantic-segmentation BASELINE for the paper's
comparison table. Trains torchvision DeepLabv3 (ResNet-50, ImageNet-pretrained
backbone) on UMD affordances, on the SAME split and scored with the SAME
metrics as the main model (scripts/train.py + utils/metrics.py).

WHY a separate file:
  This is a control, not the contribution. Keeping it isolated means the main
  model's training code is never entangled with the baseline. AffordanceNet's
  published UMD numbers (Table II: DeepLab 0.733, ED-RGB 0.766, AffordanceNet
  0.799 in weighted F-measure) are reassuring but were trained elsewhere years
  ago; a DeepLab you train yourself on your split, scored by your evaluate
  code, is the apples-to-apples comparison — and reproducing DeepLab ~0.733
  validates the F_beta^omega implementation.

WHAT it is (vs the main model):
  - Affordance MASK only (no surface normals) — the standard affordance baseline.
  - ResNet-50 backbone is TRAINED end-to-end (the main model freezes DINOv2).
  - Multi-label: 7 sigmoid channels, DiceBCELoss (same mask loss as the main
    model), so the only thing that changes is the architecture.

Run:
    python scripts/train_baseline.py --epochs 40 --split_type novel_instance
    # then read checkpoints_baseline/evaluation_baseline.json
"""

import sys
import json
import time
import argparse
from pathlib import Path
from argparse import BooleanOptionalAction

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_TOOLS, TRAIN_INTRINSICS, AFFORDANCE_CLASSES, N_AFFORDANCE_CLASSES
from utils.dataset import UMDAffordanceDataset, make_split, save_split_definition
from utils.losses import DiceBCELoss, iou, iou_accumulate, iou_from_accumulated
from utils.metrics import weighted_f_measure_per_class


def get_args():
    p = argparse.ArgumentParser("Train DeepLabv3 affordance baseline")
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--aux_weight', type=float, default=0.4,
                   help="Weight on DeepLab's auxiliary classifier loss.")
    p.add_argument('--split_type', type=str, default='novel_instance',
                   choices=['novel_instance', 'category', 'file', 'instance'])
    p.add_argument('--split_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='checkpoints_baseline')
    p.add_argument('--resume', action='store_true',
                   help="Resume from <output_dir>/last.pth if present. Safe to "
                        "re-run the same command on an intermittent GPU.")
    p.add_argument('--val_wfb', action=BooleanOptionalAction, default=True)
    p.add_argument('--val_wfb_batches', type=int, default=20,
                   help="Per-epoch F-measure on first N val batches (0 = all).")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_frac', type=float, default=0.2)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, compute_wfb, wfb_max_batches):
    model.eval()
    loss_sum, n = 0.0, 0
    inter = torch.zeros(N_AFFORDANCE_CLASSES, device=device)
    union = torch.zeros(N_AFFORDANCE_CLASSES, device=device)
    wfb_rows = []
    # Spread the WFb-scored batches evenly across the whole (unshuffled, tool-
    # sorted) split, otherwise the first N contiguous batches are a single tool
    # family (e.g. all bowls -> only the 'contain' class has GT). Deterministic.
    wfb_batch_ids = None  # None => score every batch
    if compute_wfb and wfb_max_batches > 0:
        total = len(loader)
        if wfb_max_batches < total:
            wfb_batch_ids = set(
                int(i) for i in
                np.linspace(0, total - 1, wfb_max_batches).round().astype(int)
            )
    for b_idx, batch in enumerate(loader):
        rgb = batch['rgb'].to(device)
        gt = batch['mask'].to(device)
        logits = model(rgb)['out']
        loss_sum += loss_fn(logits, gt).item()
        n += 1
        bi, bu = iou_accumulate(logits, gt)
        inter += bi
        union += bu
        if compute_wfb and (wfb_batch_ids is None or b_idx in wfb_batch_ids):
            probs = torch.sigmoid(logits).cpu().numpy()
            gtn = gt.cpu().numpy()
            for s in range(probs.shape[0]):
                wfb_rows.append(weighted_f_measure_per_class(probs[s], gtn[s]))
    mean_iou, per_class_iou = iou_from_accumulated(inter, union)
    out = {'loss': loss_sum / max(n, 1), 'iou_dataset': mean_iou,
           'iou_dataset_per_class': per_class_iou, 'wfb': None, 'wfb_per_class': None}
    if compute_wfb and wfb_rows:
        arr = np.stack(wfb_rows, axis=0)
        with np.errstate(all='ignore'):
            cms = np.nanmean(arr, axis=0)
        out['wfb_per_class'] = [None if np.isnan(m) else float(m) for m in cms]
        valid = cms[~np.isnan(cms)]
        out['wfb'] = float(valid.mean()) if valid.size else None
    return out


def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Device: {DEVICE} | Baseline checkpoints: {out_dir}")

    # ---- Data (same split as the main model) ----
    train_ds = UMDAffordanceDataset(raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
                                    augment=True)
    val_ds = UMDAffordanceDataset(raw_dir=RAW_TOOLS, intrinsics=TRAIN_INTRINSICS,
                                  augment=False)
    train_idx, val_idx = make_split(train_ds, split_type=args.split_type,
                                    seed=args.seed, val_frac=args.val_frac,
                                    split_file=args.split_file)
    save_split_definition(train_ds, train_idx, val_idx,
                          out_dir / f"split_{args.split_type}.json")
    print(f"Split: {args.split_type} | n_train={len(train_idx)} n_val={len(val_idx)}")
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model: DeepLabv3-ResNet50, ImageNet backbone, 7 affordance channels ----
    model = deeplabv3_resnet50(weights=None,
                               weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                               num_classes=N_AFFORDANCE_CLASSES, aux_loss=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = DiceBCELoss()  # same multi-label mask loss as the main model

    start_epoch, best_wfb, best_miou = 0, -1.0, -1.0
    last_ckpt = out_dir / "last.pth"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optim'])
        start_epoch = ck['epoch'] + 1
        best_wfb = ck.get('best_wfb', -1.0)
        best_miou = ck.get('best_miou', -1.0)
        print(f"Resumed baseline at epoch {start_epoch} | best WFb {best_wfb:.4f}")

    history = out_dir / "history.jsonl"
    hist_fh = open(history, "a", buffering=1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs} TRAIN")
        for batch in pbar:
            rgb = batch['rgb'].to(DEVICE)
            gt = batch['mask'].to(DEVICE)
            optimizer.zero_grad()
            out = model(rgb)
            loss = loss_fn(out['out'], gt) + args.aux_weight * loss_fn(out['aux'], gt)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            nb += 1
            pbar.set_postfix(L=f"{loss.item():.3f}")
        train_loss = run_loss / max(nb, 1)

        val = evaluate(model, val_loader, DEVICE, loss_fn,
                       compute_wfb=args.val_wfb, wfb_max_batches=args.val_wfb_batches)
        wfb_str = f" WFb {val['wfb']:.3f}" if val['wfb'] is not None else ""
        print(f"E{epoch+1} | Train L {train_loss:.4f} | Val L {val['loss']:.4f} "
              f"dIoU {val['iou_dataset']:.3f}{wfb_str} | {time.time()-t0:.0f}s")
        hist_fh.write(json.dumps({
            'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val['loss'],
            'iou_dataset': val['iou_dataset'], 'wfb': val['wfb'],
            'wfb_per_class': val['wfb_per_class'],
        }) + "\n")

        if val['iou_dataset'] > best_miou:
            best_miou = val['iou_dataset']
            torch.save(model.state_dict(), out_dir / "best.pth")
        if val['wfb'] is not None and val['wfb'] > best_wfb:
            best_wfb = val['wfb']
            torch.save(model.state_dict(), out_dir / "best_wfb.pth")
            print(f"   new best baseline (val F_beta^omega = {best_wfb:.4f})")
        # Richer last.pth so --resume can continue on an intermittent GPU.
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'best_wfb': best_wfb, 'best_miou': best_miou},
                   out_dir / "last.pth")
    hist_fh.close()

    # ---- Final authoritative eval (full val set, full F-measure) ----
    print("\nFinal eval on full val set (F_beta^omega over all images)...")
    final = evaluate(model, val_loader, DEVICE, loss_fn,
                     compute_wfb=True, wfb_max_batches=0)
    report = {
        'model': 'deeplabv3_resnet50',
        'split_type': args.split_type,
        'iou_dataset': final['iou_dataset'],
        'iou_dataset_per_class': dict(zip(AFFORDANCE_CLASSES, final['iou_dataset_per_class'])),
        'wfb_average': final['wfb'],
        'wfb_per_class': (dict(zip(AFFORDANCE_CLASSES, final['wfb_per_class']))
                          if final['wfb_per_class'] else None),
        'reference_affordancenet_table_II': {
            'DeepLab': 0.733, 'ED-RGB': 0.766, 'AffordanceNet': 0.799},
    }
    with open(out_dir / "evaluation_baseline.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== BASELINE weighted F-measure (compare to AffordanceNet Table II) ===")
    if final['wfb_per_class']:
        for cls, w in zip(AFFORDANCE_CLASSES, final['wfb_per_class']):
            print(f"  {cls:12s} : {'n/a' if w is None else f'{w:.4f}'}")
    avg = final['wfb']
    print(f"  {'AVERAGE':12s} : {'n/a' if avg is None else f'{avg:.4f}'}"
          f"   (reference DeepLab 0.733)")
    print(f"\nReport: {out_dir / 'evaluation_baseline.json'}")


if __name__ == "__main__":
    main()
