import os
from pathlib import Path

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- UMD AFFORDANCE CLASSES ---
# Canonical mapping from Myers et al. 2015 (UMD Part Affordance Dataset).
# Each ID corresponds to a per-pixel functional region in `_label.mat`.
# Class 0 is background and is NOT a predicted channel; the model produces
# N_AFFORDANCE_CLASSES = 7 sigmoid logits, one per affordance below, in the
# order they appear in this tuple (index 0 → grasp, index 6 → wrap-grasp).
AFFORDANCE_CLASSES = (
    "grasp",        # 1 — region you grip to pick the object up (handles, shafts).
    "cut",          # 2 — sharp edge that severs material (knife / scissor blades).
    "scoop",        # 3 — concave surface that lifts loose material (spoon bowl).
    "contain",      # 4 — interior cavity that holds material (cup / mug / bowl interior).
    "pound",        # 5 — heavy striking face that delivers impact (mallet head).
    "support",      # 6 — broad flat region that supports another object (turner blade, trowel face).
    "wrap-grasp",   # 7 — graspable region wrapped by the hand (mug handle, cylindrical grips).
)
N_AFFORDANCE_CLASSES = len(AFFORDANCE_CLASSES)
# Raw label IDs in the .mat files start at 1; 0 is background.
AFFORDANCE_LABEL_IDS = tuple(range(1, N_AFFORDANCE_CLASSES + 1))

# --- DATA PATHS ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_TOOLS = DATA_DIR / "raw" / "part-affordance-dataset" / "tools"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TOOLS = PROCESSED_DIR / "normals"

# --- TRAINING DEFAULTS ---
IMAGE_SIZE = (480, 640)
BATCH_SIZE = 4

# --- CAMERA INTRINSICS ---
# Per-sensor calibration. Replace realsense / femto values with your actual
# calibrated numbers before deploying on those cameras.
CAMERA_INTRINSICS = {
    'kinect_v1':      dict(fx=525.0, fy=525.0, cx=320.0, cy=240.0),
    'realsense_d435': dict(fx=615.0, fy=615.0, cx=320.0, cy=240.0),
    'femto_bolt':     dict(fx=605.0, fy=605.0, cx=320.0, cy=240.0),
}
TRAIN_INTRINSICS     = CAMERA_INTRINSICS['kinect_v1']
INFERENCE_INTRINSICS = CAMERA_INTRINSICS['kinect_v1']  # override per deployment
