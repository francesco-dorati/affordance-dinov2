"""
dataset.py — UMD affordance dataset with:
  - configurable camera intrinsics (passed from config.py)
  - principal-point correction after center crop
  - optional joint augmentation (utils/augmentations.py)
  - optional depth tensor output for an RGB-D variant

Returns a dict with keys 'rgb', 'mask', 'normals', 'tool_name'
(and 'depth' if return_depth=True).
"""

import os
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.geometry import compute_normals
from utils.augmentations import JointTrainTransform


class UMDAffordanceDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        crop_size: int = 448,
        intrinsics: dict = None,
        augment: bool = False,
        return_depth: bool = False,
    ):
        self.raw_dir = raw_dir
        self.crop_size = crop_size
        self.intrinsics = intrinsics or dict(fx=525.0, fy=525.0, cx=320.0, cy=240.0)
        self.augment_fn = JointTrainTransform() if augment else None
        self.return_depth = return_depth

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.samples = []
        for tool in os.listdir(raw_dir):
            tp = os.path.join(raw_dir, tool)
            if not os.path.isdir(tp):
                continue
            for f in os.listdir(tp):
                if f.endswith("_label.mat"):
                    idx = f.split('_')[-2]
                    self.samples.append((tool, idx))

    # --- helpers ---
    def _center_crop(self, img):
        h, w = img.shape[:2]
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return img[top:top + self.crop_size, left:left + self.crop_size], top, left

    def _shifted_intrinsics(self, top, left):
        intr = dict(self.intrinsics)
        intr['cx'] = intr['cx'] - left
        intr['cy'] = intr['cy'] - top
        return intr

    # --- protocol ---
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tool, fid = self.samples[idx]
        prefix = os.path.join(self.raw_dir, tool, f"{tool}_{fid}")

        rgb = cv2.cvtColor(cv2.imread(f"{prefix}_rgb.jpg"), cv2.COLOR_BGR2RGB)
        labels = sio.loadmat(f"{prefix}_label.mat")['gt_label']
        depth = cv2.imread(f"{prefix}_depth.png", cv2.IMREAD_ANYDEPTH)

        rgb_c, top, left = self._center_crop(rgb)
        labels_c, _, _ = self._center_crop(labels)
        depth_c, _, _ = self._center_crop(depth)

        # Normals are computed with the principal point shifted to the cropped frame.
        intr = self._shifted_intrinsics(top, left)
        normals_raw, _ = compute_normals(depth_c, **intr)

        # UMD label IDs: 1 = grasp, 7 = wrap-grasp
        mask = np.isin(labels_c, [1, 7]).astype(np.float32)

        if self.augment_fn is not None:
            rgb_c, mask, normals_raw = self.augment_fn(rgb_c, mask, normals_raw)

        rgb_t = self.normalize(self.to_tensor(rgb_c))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        normals_t = torch.from_numpy(np.ascontiguousarray(normals_raw)).permute(2, 0, 1).float()

        out = {
            'rgb': rgb_t,
            'mask': mask_t,
            'normals': normals_t,
            'tool_name': tool,
        }
        if self.return_depth:
            depth_t = torch.from_numpy(depth_c.astype(np.float32) / 1000.0).unsqueeze(0)
            out['depth'] = depth_t
        return out


def instance_split(dataset, seed: int = 42, val_frac: float = 0.2):
    """Deterministic split by tool name. Returns (train_indices, val_indices).

    Uses a local RandomState so the global numpy RNG is not perturbed. Tools
    are shuffled once with the given seed; the first (1 - val_frac) fraction
    by tool name go into the train set, the remainder into val. This is an
    instance-split: a tool seen at training is NEVER seen at validation.
    """
    all_tools = sorted({s[0] for s in dataset.samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(all_tools)
    n_train = int((1 - val_frac) * len(all_tools))
    train_tools = set(all_tools[:n_train])
    train_idx = [i for i, s in enumerate(dataset.samples) if s[0] in train_tools]
    val_idx   = [i for i, s in enumerate(dataset.samples) if s[0] not in train_tools]
    return train_idx, val_idx
