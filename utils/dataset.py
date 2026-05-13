import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UMDAffordanceDataset(Dataset):
    def __init__(self, raw_dir, processed_dir, crop_size=448):
        """
        Args:
            raw_dir: Path to 'data/raw/part-affordance-dataset/tools'
            processed_dir: Path to 'data/processed/umd_normals'
            crop_size: The square size for DINOv2 (default 448)
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.crop_size = crop_size
        self.samples = []
        self.to_tensor = transforms.ToTensor()

        # We look in raw_dir to find all the valid frame indices
        for tool in os.listdir(raw_dir):
            tool_path = os.path.join(raw_dir, tool)
            if not os.path.isdir(tool_path): 
                continue
            
            for file in os.listdir(tool_path):
                if file.endswith("_label.mat"):
                    frame_idx_str = file.split('_')[-2] 
                    self.samples.append((tool, frame_idx_str))

    def center_crop(self, img):
        """Helper to crop images to the center square."""
        h, w = img.shape[:2]
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return img[top:top+self.crop_size, left:left+self.crop_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tool, frame_idx_str = self.samples[idx]
        
        # 1. Construct Paths
        raw_prefix = os.path.join(self.raw_dir, tool, f"{tool}_{frame_idx_str}")
        proc_prefix = os.path.join(self.processed_dir, tool, f"{tool}_{frame_idx_str}")
        
        # 2. Load Raw Data (RGB and Labels)
        rgb = cv2.cvtColor(cv2.imread(f"{raw_prefix}_rgb.jpg"), cv2.COLOR_BGR2RGB)
        labels = sio.loadmat(f"{raw_prefix}_label.mat")['gt_label']
        
        # 3. Load Processed Data (Precomputed Normals)
        normals_path = f"{proc_prefix}_normal.npy"
        # We use np.load for speed
        normals_raw = np.load(normals_path) 
        
        # 4. Apply Center Crop to RGB and Labels
        rgb_cropped = self.center_crop(rgb)
        labels_cropped = self.center_crop(labels)
        
        # 5. Create Mask from cropped labels
        # 1 = grasp, 7 = wrap-grasp
        mask = np.isin(labels_cropped, [1, 7]).astype(np.float32)
        
        # 6. Convert to PyTorch Tensors
        # RGB: [3, 448, 448]
        rgb_tensor = self.to_tensor(rgb_cropped) 
        
        # Mask: [1, 448, 448]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # Normals: [3, 448, 448]
        normals_tensor = torch.from_numpy(normals_raw).permute(2, 0, 1)

        return {
            'rgb': rgb_tensor,
            'mask': mask_tensor,
            'normals': normals_tensor,
            'tool_name': tool
        }