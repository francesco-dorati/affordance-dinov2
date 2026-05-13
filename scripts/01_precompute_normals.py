import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from utils.geometry import batch_precompute_normals

if __name__ == "__main__":
    # Use the path from our central config
    batch_precompute_normals(str(config.RAW_TOOLS), str(config.PROCESSED_TOOLS))