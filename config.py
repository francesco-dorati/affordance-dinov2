import os
from pathlib import Path

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parent

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
