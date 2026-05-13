import os
from pathlib import Path

# --- PROJECT STRUCTURE ---
# Gets the absolute path to the directory containing this config.py file
PROJECT_ROOT = Path(__file__).resolve().parent

# --- DATA PATHS ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_TOOLS = DATA_DIR / "raw" / "part-affordance-dataset" / "tools"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TOOLS = PROCESSED_DIR / "normals"

# --- MODEL HYPERPARAMETERS (For Phase 2) ---
# We will populate these later, but setting the skeleton up now!
IMAGE_SIZE = (480, 640)
BATCH_SIZE = 4