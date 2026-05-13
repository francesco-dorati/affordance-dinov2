# utils/visualization.py
import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.geometry import compute_normals

def visualize_sample(base_dir, tool_instance, frame_idx):
    """Loads a UMD sample, computes normals, and plots the results."""
    prefix = os.path.join(base_dir, tool_instance, f"{tool_instance}_{frame_idx:08d}")
    rgb_path = f"{prefix}_rgb.jpg"
    depth_path = f"{prefix}_depth.png"
    label_path = f"{prefix}_label.mat"
    
    for path in [rgb_path, depth_path, label_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing file: {path}")
            return

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    labels = sio.loadmat(label_path)['gt_label']
    
    affordance_mask = np.isin(labels, [1, 7]).astype(np.uint8)
    _, normals_vis = compute_normals(depth)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("RGB Input")
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Depth Map")
    depth_vis = depth.copy()
    depth_vis[depth == 0] = np.max(depth) 
    plt.imshow(depth_vis, cmap='plasma')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Grasp Mask")
    overlay = rgb.copy()
    overlay[affordance_mask == 1] = [0, 255, 0]
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Surface Normals")
    plt.imshow(normals_vis)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()