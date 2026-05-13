import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.geometry import compute_normals

class UMDAffordanceDataset(Dataset):
    def __init__(self, base_dir):
        """
        Args:
            base_dir: Path to 'data/raw/part-affordance-dataset/tools'
        """
        self.base_dir = base_dir
        self.samples = []
        self.to_tensor = transforms.ToTensor()

        # Dynamically find all valid 8-digit frames in the dataset
        for tool in os.listdir(base_dir):
            tool_path = os.path.join(base_dir, tool)
            if not os.path.isdir(tool_path): 
                continue
            
            for file in os.listdir(tool_path):
                if file.endswith("_label.mat"):
                    frame_idx_str = file.split('_')[-2] 
                    self.samples.append((tool, frame_idx_str))

    def __len__(self):
        """Tells PyTorch exactly how many valid samples we have."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetches exactly one sample and translates it into Tensors."""
        tool, frame_idx_str = self.samples[idx]
        
        # 1. File Paths
        prefix = os.path.join(self.base_dir, tool, f"{tool}_{frame_idx_str}")
        rgb_path = f"{prefix}_rgb.jpg"
        depth_path = f"{prefix}_depth.png"
        label_path = f"{prefix}_label.mat"
        
        # 2. Load Raw Data
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        labels = sio.loadmat(label_path)['gt_label']
        
        # 3. Create Targets
        # Mask: 1 for Grasp/Wrap-Grasp, 0 for Background
        mask = np.isin(labels, [1, 7]).astype(np.float32)
        
        # Normals: We compute them dynamically using our validated math
        normals_raw, _ = compute_normals(depth)
        
        # 4. Convert to PyTorch Tensors [Channels, Height, Width]
        # RGB becomes [3, 480, 640] and scaled to [0.0, 1.0]
        rgb_tensor = self.to_tensor(rgb) 
        
        # Depth becomes [1, 480, 640] scaled to meters
        depth_tensor = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 1000.0 
        
        # Mask becomes [1, 480, 640]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # Normals become [3, 480, 640]
        normals_tensor = torch.from_numpy(normals_raw.astype(np.float32)).permute(2, 0, 1)

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'normals': normals_tensor,
            'tool_name': tool
        }