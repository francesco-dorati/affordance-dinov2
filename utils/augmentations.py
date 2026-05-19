"""
augmentations.py — Joint train-time augmentations for RGB + mask + normals.

WHY:
  Your current dataset has zero augmentation, and UMD always centers tools.
  The model never sees off-center / rotated / dim / occluded inputs, so it
  generalizes poorly to your in-the-wild office captures.

WHAT IT DOES:
  Applies geometric augmentations CONSISTENTLY across RGB, mask, AND normals.
  Critically, when we rotate the image, we ALSO rotate the normal vectors
  themselves (their direction in image coordinates must follow the rotation).
  When we horizontally flip, we negate the x-component of the normals.

  Photometric augmentations and Gaussian noise are RGB-only.

USAGE:
  aug = JointTrainTransform()
  rgb, mask, normals = aug(rgb_uint8, mask_f32, normals_f32_HxWx3)

REVERT: delete this file.
"""

import math
import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF


# =====================================================================
# 1. Helpers — keeping normals geometrically consistent
# =====================================================================
def _rotate_normals(normals: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the 3D normal vectors in the image plane (x,y components)."""
    theta = math.radians(angle_deg)
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,              0.0,             1.0],
    ], dtype=np.float32)
    flat = normals.reshape(-1, 3)
    return (flat @ R.T).reshape(normals.shape)


def _flip_normals_horizontal(normals: np.ndarray) -> np.ndarray:
    n = normals.copy()
    n[..., 0] *= -1.0
    return n


# =====================================================================
# 2. Joint train transform
# =====================================================================
class JointTrainTransform:
    """
    Default settings are conservative — tuned for the UMD dataset where objects
    are already roughly centered. Increase max_rot / scale_range if you want to
    simulate more aggressive viewpoint variation.
    """
    def __init__(
        self,
        p_hflip: float = 0.5,
        max_rot: float = 15.0,
        scale_range=(0.85, 1.15),
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
        gauss_noise_std: float = 0.01,
        p_erase: float = 0.25,
    ):
        self.p_hflip = p_hflip
        self.max_rot = max_rot
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gauss_noise_std = gauss_noise_std
        self.p_erase = p_erase

    def __call__(self, rgb: np.ndarray, mask: np.ndarray, normals: np.ndarray):
        """
        Args:
            rgb:     HxWx3 uint8
            mask:    HxW   float32 (0/1)
            normals: HxWx3 float32 (unit vectors, 0 in invalid regions)
        Returns:
            same shapes/dtypes after augmentation.
        """
        H, W = rgb.shape[:2]

        # -------- Geometric: rotation + scale --------
        angle = random.uniform(-self.max_rot, self.max_rot)
        scale = random.uniform(*self.scale_range)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)

        rgb = cv2.warpAffine(
            rgb, M, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        mask = cv2.warpAffine(
            mask, M, (W, H),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        normals = cv2.warpAffine(
            normals, M, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        normals = _rotate_normals(normals, angle)

        # -------- Geometric: horizontal flip --------
        if random.random() < self.p_hflip:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            normals = np.ascontiguousarray(normals[:, ::-1, :])
            normals = _flip_normals_horizontal(normals)

        # -------- Photometric (RGB only) --------
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if self.brightness > 0:
            rgb_t = TF.adjust_brightness(rgb_t, 1 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            rgb_t = TF.adjust_contrast(rgb_t,   1 + random.uniform(-self.contrast, self.contrast))
        if self.saturation > 0:
            rgb_t = TF.adjust_saturation(rgb_t, 1 + random.uniform(-self.saturation, self.saturation))
        if self.hue > 0:
            rgb_t = TF.adjust_hue(rgb_t, random.uniform(-self.hue, self.hue))
        rgb_t = rgb_t.clamp(0, 1)

        # -------- Sensor noise --------
        if self.gauss_noise_std > 0:
            rgb_t = (rgb_t + torch.randn_like(rgb_t) * self.gauss_noise_std).clamp(0, 1)

        # -------- Random erasing (occlusion robustness) --------
        if random.random() < self.p_erase:
            eh = random.randint(H // 16, H // 6)
            ew = random.randint(W // 16, W // 6)
            ey = random.randint(0, H - eh)
            ex = random.randint(0, W - ew)
            rgb_t[:, ey:ey + eh, ex:ex + ew] = torch.rand(3, 1, 1)

        rgb = (rgb_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return rgb, mask, normals
