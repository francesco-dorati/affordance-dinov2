# utils/geometry.py
import os
import cv2
import numpy as np
from tqdm import tqdm

def center_crop(img, dim):
    """Crops the center of an image to a square (dim x dim)."""
    h, w = img.shape[:2]
    top = (h - dim) // 2
    left = (w - dim) // 2
    return img[top:top+dim, left:left+dim]

def compute_normals(depth_map, fx=525.0, fy=525.0, cx=320.0, cy=240.0):
    """
    Computes accurate surface normals by back-projecting the depth map 
    into a 3D point cloud and calculating cross products.
    """
    Z = depth_map.astype(np.float32) / 1000.0
    valid_mask = Z > 0

    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    dXdv, dXdu = np.gradient(X)
    dYdv, dYdu = np.gradient(Y)
    dZdv, dZdu = np.gradient(Z)

    Tu = np.stack([dXdu, dYdu, dZdu], axis=-1)
    Tv = np.stack([dXdv, dYdv, dZdv], axis=-1)

    normals = np.cross(Tu, Tv)

    norm_magnitude = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_normalized = -(normals / (norm_magnitude + 1e-6))

    normals_vis = (normals_normalized + 1.0) / 2.0
    
    normals_vis[~valid_mask] = [0, 0, 0]
    normals_normalized[~valid_mask] = 0
    
    return normals_normalized, normals_vis

def batch_precompute_normals(base_dir, output_dir, crop_size=448):
    """
    Reads from base_dir (raw), saves to output_dir (processed).
    """
    tools = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for tool in tools:
        # Create the corresponding subfolder in processed
        tool_out_path = os.path.join(output_dir, tool)
        os.makedirs(tool_out_path, exist_ok=True)
        
        tool_in_path = os.path.join(base_dir, tool)
        files = [f for f in os.listdir(tool_in_path) if f.endswith("_depth.png")]
        
        for f in tqdm(files, desc=f"Processing {tool}"):
            # Load from RAW
            depth = cv2.imread(os.path.join(tool_in_path, f), cv2.IMREAD_ANYDEPTH)
            if depth is None: continue
            
            # Math
            depth_cropped = center_crop(depth, crop_size)
            normals_raw, _ = compute_normals(depth_cropped)
            
            # Save to PROCESSED
            save_name = f.replace("_depth.png", "_normal.npy")
            np.save(os.path.join(tool_out_path, save_name), normals_raw.astype(np.float32))