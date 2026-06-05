"""
dataset.py — UMD affordance dataset with:
  - configurable camera intrinsics (passed from config.py)
  - principal-point correction after center crop
  - optional joint augmentation (utils/augmentations.py)
  - optional depth tensor output for an RGB-D variant
  - MULTI-CLASS affordance mask: returns a (N_AFFORDANCE_CLASSES, H, W)
    multi-hot tensor, one channel per UMD affordance class. The class order
    is defined by config.AFFORDANCE_CLASSES.

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
from config import AFFORDANCE_LABEL_IDS, N_AFFORDANCE_CLASSES


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

        # Index only samples where ALL three files (label.mat, rgb.jpg,
        # depth.png) exist AND have non-trivial size. Zero-byte files (from
        # interrupted copies or cloud-sync stubs) pass `os.path.exists` but
        # crash cv2.imread later with an opaque `_src.empty()` error.
        MIN_BYTES = 1024  # 1 KB — real UMD frames are tens of KB minimum.
        self.samples = []
        skipped_missing = 0
        skipped_empty   = 0
        for tool in sorted(os.listdir(raw_dir)):
            tp = os.path.join(raw_dir, tool)
            if not os.path.isdir(tp):
                continue
            for f in os.listdir(tp):
                if not f.endswith("_label.mat"):
                    continue
                idx = f.split('_')[-2]
                rgb_p   = os.path.join(tp, f"{tool}_{idx}_rgb.jpg")
                depth_p = os.path.join(tp, f"{tool}_{idx}_depth.png")
                if not (os.path.exists(rgb_p) and os.path.exists(depth_p)):
                    skipped_missing += 1
                    continue
                if (os.path.getsize(rgb_p) < MIN_BYTES
                        or os.path.getsize(depth_p) < MIN_BYTES):
                    skipped_empty += 1
                    continue
                self.samples.append((tool, idx))
        if skipped_missing or skipped_empty:
            print(f"[UMDAffordanceDataset] indexed {len(self.samples)} samples; "
                  f"skipped {skipped_missing} (missing rgb/depth) + "
                  f"{skipped_empty} (zero-byte / stub rgb or depth)")

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

        # Read raw bytes in Python, then let cv2 decode the buffer. This
        # bypasses cv2's internal file I/O which can silently fail on Mac
        # when files have extended attributes (quarantine flags, metadata)
        # or unusual path encodings, even when the content is fine.
        def _imread_via_buffer(path: str, flags: int) -> np.ndarray:
            with open(path, "rb") as fh:
                buf = np.frombuffer(fh.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, flags)
            if img is None:
                raise RuntimeError(
                    f"cv2.imdecode failed for {path} — file bytes are not a "
                    "valid image. Re-copy this file from the source."
                )
            return img

        rgb_bgr = _imread_via_buffer(f"{prefix}_rgb.jpg", cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        labels = sio.loadmat(f"{prefix}_label.mat")['gt_label']
        depth = _imread_via_buffer(f"{prefix}_depth.png", cv2.IMREAD_ANYDEPTH)

        rgb_c, top, left = self._center_crop(rgb)
        labels_c, _, _ = self._center_crop(labels)
        depth_c, _, _ = self._center_crop(depth)

        # Normals are computed with the principal point shifted to the cropped frame.
        intr = self._shifted_intrinsics(top, left)
        normals_raw, _ = compute_normals(depth_c, **intr)

        # Pass the multi-class label image through augmentation as a float32
        # array; cv2.warpAffine with INTER_NEAREST preserves the discrete
        # class IDs (0 background, 1..7 affordance classes).
        label_img = labels_c.astype(np.float32)

        if self.augment_fn is not None:
            rgb_c, label_img, normals_raw = self.augment_fn(rgb_c, label_img, normals_raw)

        # Expand to a (N_AFFORDANCE_CLASSES, H, W) multi-hot tensor: one
        # channel per affordance class, in config.AFFORDANCE_CLASSES order.
        label_int = label_img.astype(np.int64)
        masks = np.stack(
            [(label_int == cid).astype(np.float32) for cid in AFFORDANCE_LABEL_IDS],
            axis=0,
        )  # shape: (N_AFFORDANCE_CLASSES, H, W)

        rgb_t = self.normalize(self.to_tensor(rgb_c))
        mask_t = torch.from_numpy(np.ascontiguousarray(masks)).float()
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


def compute_class_pixel_counts(
    dataset: "UMDAffordanceDataset",
    indices=None,
    label_ids=AFFORDANCE_LABEL_IDS,
    verbose: bool = True,
):
    """Scan label files and count positive pixels per affordance class.

    Loads the raw `_label.mat` files directly (skipping the RGB / depth /
    augmentation pipeline) so the scan is fast — about 5–10 minutes for the
    full UMD training set on a typical SSD.

    Returns:
        counts:        np.ndarray of shape (len(label_ids),) — total positive
                       pixel count per class across the indexed samples.
        total_pixels:  int — total pixels examined (n_samples * H * W).
    """
    if indices is None:
        indices = range(len(dataset.samples))
    indices = list(indices)
    counts = np.zeros(len(label_ids), dtype=np.int64)
    total_pixels = 0
    iterator = indices
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(indices, desc="counting class pixels")
        except ImportError:
            pass
    for i in iterator:
        tool, fid = dataset.samples[i]
        path = os.path.join(dataset.raw_dir, tool, f"{tool}_{fid}_label.mat")
        labels = sio.loadmat(path)["gt_label"]
        # Match the dataset's center-crop so the counts correspond to the
        # same spatial extent the model actually sees.
        h, w = labels.shape[:2]
        cs = dataset.crop_size
        top = (h - cs) // 2
        left = (w - cs) // 2
        labels = labels[top:top + cs, left:left + cs]
        total_pixels += labels.size
        for c_idx, lid in enumerate(label_ids):
            counts[c_idx] += int((labels == lid).sum())
    return counts, total_pixels


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
