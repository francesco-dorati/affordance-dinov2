import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.geometry import compute_normals

class UMDAffordanceDataset(Dataset):
    def __init__(self, raw_dir, crop_size=448):
        self.raw_dir = raw_dir
        self.crop_size = crop_size
        self.samples = []
        self.to_tensor = transforms.ToTensor()

        for tool in os.listdir(raw_dir):
            tool_path = os.path.join(raw_dir, tool)
            if not os.path.isdir(tool_path): 
                continue
            
            for file in os.listdir(tool_path):
                if file.endswith("_label.mat"):
                    frame_idx_str = file.split('_')[-2] 
                    self.samples.append((tool, frame_idx_str))

    def center_crop(self, img):
        h, w = img.shape[:2]
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return img[top:top+self.crop_size, left:left+self.crop_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tool, frame_idx_str = self.samples[idx]
        prefix = os.path.join(self.raw_dir, tool, f"{tool}_{frame_idx_str}")
        
        # 1. Load Raw Data
        rgb = cv2.cvtColor(cv2.imread(f"{prefix}_rgb.jpg"), cv2.COLOR_BGR2RGB)
        labels = sio.loadmat(f"{prefix}_label.mat")['gt_label']
        depth = cv2.imread(f"{prefix}_depth.png", cv2.IMREAD_ANYDEPTH)
        
        # 2. Crop EVERYTHING first (saves CPU time on the math)
        rgb_cropped = self.center_crop(rgb)
        labels_cropped = self.center_crop(labels)
        depth_cropped = self.center_crop(depth)
        
        # 3. Compute Normals dynamically
        normals_raw, _ = compute_normals(depth_cropped)
        
        # 4. Create Mask (1 for grasp, 7 for wrap-grasp)
        mask = np.isin(labels_cropped, [1, 7]).astype(np.float32)
        
        # 5. Convert to Tensors
        rgb_tensor = self.to_tensor(rgb_cropped) 
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        normals_tensor = torch.from_numpy(normals_raw).permute(2, 0, 1)

        return {
            'rgb': rgb_tensor,
            'mask': mask_tensor,
            'normals': normals_tensor,
            'tool_name': tool
        }