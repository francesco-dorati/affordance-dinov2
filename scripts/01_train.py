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

# --- Path Setup ---
# Add the project root to the Python path to allow for absolute imports
# This is essential for running the script from any location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Custom Module Imports ---
# Configuration, models, and dataset utilities defined in other files
from config import RAW_TOOLS, PROJECT_ROOT
from models.backbone import DINOv2Backbone
from models.decoder import MultiTaskDecoder
from utils.dataset import UMDAffordanceDataset

# =====================================================================
# >> 1. HYPERPARAMETERS & CONFIGURATION
# =====================================================================

# --- Training Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# --- Data & Model Paths ---
DATA_DIR = RAW_TOOLS
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True) # Create directory if it doesn't exist

# --- Loss Balancing ---
# These weights determine the importance of each task.
# If the model struggles with one task, adjusting these can help.
MASK_LOSS_WEIGHT = 1.0
NORMAL_LOSS_WEIGHT = 5.0 # Give normals a higher weight to encourage geometric learning


def calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask):
    """
    Calculates the Cosine Similarity loss only on the 'active' pixels 
    of the ground truth affordance mask. This prevents the model from being
    penalized for random normal predictions in the background.

    Args:
        pred_normals (Tensor): Predicted normals [B, 3, H, W]
        gt_normals (Tensor): Ground truth normals [B, 3, H, W]
        gt_mask (Tensor): Ground truth affordance mask [B, 1, H, W]

    Returns:
        Tensor: A single scalar value for the batch loss.
    """
    # Ensure normals are unit vectors, which is crucial for cosine similarity
    pred_normals = F.normalize(pred_normals, p=2, dim=1)
    gt_normals = F.normalize(gt_normals, p=2, dim=1)

    # Calculate cosine similarity across the channel dimension (dim=1)
    # The result is a map of similarity scores, shape [B, H, W]
    similarity = F.cosine_similarity(pred_normals, gt_normals, dim=1)

    # The loss is 1 minus the similarity. We want to maximize similarity (-> 1), 
    # which means minimizing the loss (-> 0).
    loss_map = 1 - similarity

    # Create a boolean mask where the ground truth affordance is present
    active_mask = (gt_mask > 0).squeeze(1) # Shape: [B, H, W]

    # If there are no active pixels in the batch, return a zero loss
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_normals.device, requires_grad=True)

    # Apply the mask to the loss map to select values only from active regions
    masked_loss = loss_map[active_mask]

    # Return the mean of the loss over the active pixels
    return masked_loss.mean()


def main():
    """
    Main function to orchestrate the model training pipeline.
    """
    print(f"Using device: {DEVICE}")
    print(f"Loading data from: {DATA_DIR}")

    # =====================================================================
    # >> 2. DATA LOADING
    # =====================================================================
    # Instantiate the dataset. Normals are computed on-the-fly.
    dataset = UMDAffordanceDataset(raw_dir=DATA_DIR)

    # --- Instance-Based Data Split ---
    # To properly test generalization, we must ensure that the model is validated
    # on tool *instances* it has never seen during training. A simple random
    # split would cause data leakage (e.g., frames of 'knife_01' in both train and val).
    all_tools = sorted(list(set([s[0] for s in dataset.samples])))
    np.random.seed(42) # Use a fixed seed for reproducible splits
    np.random.shuffle(all_tools)

    # Split the list of tool names into 80% train, 20% validation
    split_idx = int(0.8 * len(all_tools))
    train_tools = set(all_tools[:split_idx])
    val_tools = set(all_tools[split_idx:])

    print(f"Found {len(all_tools)} unique tool instances.")
    print(f"Splitting into {len(train_tools)} training instances and {len(val_tools)} validation instances.")

    # Create lists of indices corresponding to the train/val tool instances
    train_indices = [i for i, s in enumerate(dataset.samples) if s[0] in train_tools]
    val_indices = [i for i, s in enumerate(dataset.samples) if s[0] in val_tools]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # =====================================================================
    # >> 3. MODEL, OPTIMIZER, AND LOSS INITIALIZATION
    # =====================================================================
    # Load the frozen DINOv2 backbone
    backbone = DINOv2Backbone(freeze=True).to(DEVICE)
    backbone.eval() # Set to evaluation mode as it's frozen

    # Initialize the custom multi-task decoder
    decoder = MultiTaskDecoder().to(DEVICE)

    # The optimizer only needs to update the weights of the decoder
    optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    # Standard Binary Cross-Entropy for the binary affordance mask
    mask_criterion = nn.BCELoss()

    # =====================================================================
    # >> 4. TRAINING & VALIDATION LOOP
    # =====================================================================
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        decoder.train()
        train_loss_mask, train_loss_normal, train_loss_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]")
        for batch in pbar:
            rgb = batch['rgb'].to(DEVICE)
            gt_mask = batch['mask'].to(DEVICE)
            gt_normals = batch['normals'].to(DEVICE)

            optimizer.zero_grad()

            # --- Forward Pass ---
            with torch.no_grad(): # Backbone is frozen, no gradients needed
                features = backbone(rgb)
            pred_mask, pred_normals = decoder(features)

            # --- Loss Calculation ---
            loss_mask = mask_criterion(pred_mask, gt_mask)
            loss_normal = calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask)
            total_loss = (MASK_LOSS_WEIGHT * loss_mask) + (NORMAL_LOSS_WEIGHT * loss_normal)

            # --- Backward Pass & Optimization ---
            total_loss.backward()
            optimizer.step()

            train_loss_mask += loss_mask.item()
            train_loss_normal += loss_normal.item()
            train_loss_total += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

        # --- Validation Phase ---
        decoder.eval()
        val_loss_mask, val_loss_normal, val_loss_total = 0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VAL]")
            for batch in pbar_val:
                rgb = batch['rgb'].to(DEVICE)
                gt_mask = batch['mask'].to(DEVICE)
                gt_normals = batch['normals'].to(DEVICE)

                features = backbone(rgb)
                pred_mask, pred_normals = decoder(features)

                loss_mask = mask_criterion(pred_mask, gt_mask)
                loss_normal = calculate_masked_cosine_loss(pred_normals, gt_normals, gt_mask)
                total_loss = (MASK_LOSS_WEIGHT * loss_mask) + (NORMAL_LOSS_WEIGHT * loss_normal)

                val_loss_mask += loss_mask.item()
                val_loss_normal += loss_normal.item()
                val_loss_total += total_loss.item()
                pbar_val.set_postfix(val_loss=total_loss.item())

        # --- Logging Epoch Results ---
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train | Total Loss: {train_loss_total/len(train_loader):.4f}, Mask Loss: {train_loss_mask/len(train_loader):.4f}, Normal Loss: {train_loss_normal/len(train_loader):.4f}")
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Val   | Total Loss: {avg_val_loss:.4f}, Mask Loss: {val_loss_mask/len(val_loader):.4f}, Normal Loss: {val_loss_normal/len(val_loader):.4f}")
        print("-" * 30)

        # --- Save Best Model Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = CHECKPOINT_DIR / "best_decoder.pth"
            torch.save(decoder.state_dict(), best_model_path)
            print(f"🎉 New best model saved to {best_model_path}")

    # =====================================================================
    # >> 5. SAVE FINAL MODEL
    # =====================================================================
    final_model_path = CHECKPOINT_DIR / "final_decoder.pth"
    torch.save(decoder.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
