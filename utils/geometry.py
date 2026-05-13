# utils/geometry.py
import numpy as np

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