import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse  # Added for Colab inputs

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_TOOLS
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder
from utils.dataset import UMDAffordanceDataset

# =====================================================================
# >> 1. ARGUMENT PARSER (The "Inputs")
# =====================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Train Affordance Decoder")
    parser.add_y_argument = parser.add_argument
    parser.add_y_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_y_argument('--use_drive', action='store_true', help='Save/Load from Google Drive')
    parser.add_y_argument('--epochs', type=int, default=25)
    parser.add_y_argument('--batch_size', type=int, default=8)
    parser.add_y_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

# --- Loss Function ---
def calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask):
    pred_normals = F.normalize(pred_normals, p=2, dim=1)
    gt_normals = F.normalize(gt_normals, p=2, dim=1)
    similarity = F.cosine_similarity(pred_normals, gt_normals, dim=1)
    loss_map = 1 - similarity
    active_mask = (gt_mask > 0).squeeze(1)
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_normals.device, requires_grad=True)
    return loss_map[active_mask].mean()

def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Drive Logic ---
    if args.use_drive:
        CHECKPOINT_DIR = Path("/content/drive/MyDrive/robotic_affordance_project/checkpoints")
    else:
        CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    # --- Data Prep ---
    dataset = UMDAffordanceDataset(raw_dir=RAW_TOOLS)
    all_tools = sorted(list(set([s[0] for s in dataset.samples])))
    np.random.seed(42)
    np.random.shuffle(all_tools)
    split_idx = int(0.8 * len(all_tools))
    train_tools = set(all_tools[:split_idx])
    
    train_indices = [i for i, s in enumerate(dataset.samples) if s[0] in train_tools]
    val_indices = [i for i, s in enumerate(dataset.samples) if s[0] not in train_tools]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model Setup ---
    backbone = DINOv2Backbone(freeze=True).to(DEVICE)
    decoder = MultiTaskDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    mask_criterion = nn.BCELoss()

    # =====================================================================
    # >> 2. RESUME LOGIC
    # =====================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = CHECKPOINT_DIR / "last_checkpoint.pth"

    if args.resume and checkpoint_path.exists():
        print(f"🔄 Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from Epoch {start_epoch}, Best Val Loss was: {best_val_loss:.4f}")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        # [TRAIN PHASE]
        decoder.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch in pbar:
            rgb, gt_mask, gt_normals = batch['rgb'].to(DEVICE), batch['mask'].to(DEVICE), batch['normals'].to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad(): features = backbone(rgb)
            pred_mask, pred_normals = decoder(features)
            
            loss_mask = mask_criterion(pred_mask, gt_mask)
            loss_norm = calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask)
            total_loss = loss_mask + (5.0 * loss_norm)
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

        # [VAL PHASE]
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb, gt_mask, gt_normals = batch['rgb'].to(DEVICE), batch['mask'].to(DEVICE), batch['normals'].to(DEVICE)
                features = backbone(rgb)
                pred_mask, pred_normals = decoder(features)
                total_loss = mask_criterion(pred_mask, gt_mask) + (5.0 * calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask))
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- SAVE CHECKPOINT (Save everything needed to resume) ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_data, checkpoint_path) # Save as the "latest"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), CHECKPOINT_DIR / "best_decoder.pth")
            print(f"🎉 New Best Model Saved!")

if __name__ == "__main__":
    main()