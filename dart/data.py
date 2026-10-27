"""Data pipeline for DART pretraining.

Includes:
  - MaskGenerator: block-wise iBOT mask sampler over the global-crop patch grid.
  - RGBMultiCropAugmentation: DINOv2-style multi-crop on RGB, with aligned
    pseudo-depth targets carried alongside the two global crops.
  - WebDataset loading utilities reading paired (rgb.png, depth.npy) shards.
  - collate_dinov2: stacks crops, depth targets, and sampled masks.

Depth is the only modality used as a supervision target; it never enters
the encoder. Locals therefore carry no depth target.
"""

from __future__ import annotations

import glob
import io
import math
import os
import random
from functools import partial

import albumentations as A
import cv2
import numpy as np
import torch
import webdataset as wds

from .constants import IMG_MEAN, IMG_STD, DEPTH_MEAN, DEPTH_STD
from .ddp import is_main


# ════════════════════════════════════════════════════════════════════════
# iBOT block-wise mask generator
# ════════════════════════════════════════════════════════════════════════
class MaskGenerator:
    """Block-wise mask generator over a 2D patch grid (iBOT-style)."""

    def __init__(
        self,
        input_size,
        mask_ratio_range=(0.1, 0.5),
        min_num_patches=4,
        min_aspect=0.3,
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.H, self.W = input_size
        self.num_patches = self.H * self.W
        self.mask_ratio_range = mask_ratio_range
        self.min_num_patches = min_num_patches
        self.log_aspect = (math.log(min_aspect), math.log(1 / min_aspect))

    def __call__(self):
        mask = np.zeros((self.H, self.W), dtype=bool)
        mask_ratio = random.uniform(*self.mask_ratio_range)
        num_to_mask = int(self.num_patches * mask_ratio)
        num_masked = 0
        for _ in range(100):
            target_area = random.uniform(
                self.min_num_patches,
                max(self.min_num_patches + 1, num_to_mask - num_masked),
            )
            aspect = math.exp(random.uniform(*self.log_aspect))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if h < self.H and w < self.W:
                top = random.randint(0, self.H - h)
                left = random.randint(0, self.W - w)
                new_masked = (~mask[top:top + h, left:left + w]).sum()
                if new_masked > 0:
                    mask[top:top + h, left:left + w] = True
                    num_masked += new_masked
                    if num_masked >= num_to_mask:
                        break
        return mask.flatten()


# ════════════════════════════════════════════════════════════════════════
# Multi-crop augmentation with aligned depth targets
# ════════════════════════════════════════════════════════════════════════
class RGBMultiCropAugmentation:
    """DINOv2-style multi-crop on RGB plus aligned depth targets for globals.

    RGB is the only thing that flows through the encoder. Depth is used only
    as a supervision target for the depth decoder at global-crop masked
    positions. Locals don't carry depth targets (no iBOT mask, no decoder
    applied to them).

    Globals: joint spatial aug on RGB + depth (keeps them pixel-aligned
             after random crop/flip), then RGB-only color aug. Returned as
             separate 3-channel RGB and 1-channel standardized depth tensors.
    Locals:  spatial + color aug on RGB only.
    """

    def __init__(
        self,
        global_size=224,
        local_size=96,
        n_local=8,
        global_scale=(0.32, 1.0),
        local_scale=(0.05, 0.32),
        rgb_mean=IMG_MEAN,
        rgb_std=IMG_STD,
        depth_mean=DEPTH_MEAN,
        depth_std=DEPTH_STD,
    ):
        self.n_local = n_local
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.depth_mean = depth_mean
        self.depth_std = depth_std

        # Globals: joint spatial aug to keep RGB and depth aligned
        self.global_spatial = A.Compose(
            [
                A.RandomResizedCrop(
                    (global_size, global_size),
                    scale=global_scale,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets={"depth": "image"},
        )

        # Locals: RGB-only spatial aug
        self.local_spatial = A.Compose(
            [
                A.RandomResizedCrop(
                    (local_size, local_size),
                    scale=local_scale,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        self.color_global1 = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.2),
                A.GaussianBlur(
                    blur_limit=(3, 23), sigma_limit=(0.1, 2.0), p=1.0
                ),
            ]
        )
        self.color_global2 = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.2),
                A.GaussianBlur(
                    blur_limit=(3, 23), sigma_limit=(0.1, 2.0), p=0.1
                ),
                A.Solarize(threshold_range=(0.5, 0.5), p=0.2),
            ]
        )
        self.color_local = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.2),
                A.GaussianBlur(
                    blur_limit=(3, 23), sigma_limit=(0.1, 2.0), p=0.5
                ),
            ]
        )

    def _normalize_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_f = (rgb_f - self.rgb_mean) / self.rgb_std
        return torch.from_numpy(rgb_f.transpose(2, 0, 1))  # (3, H, W)

    def _normalize_depth(self, depth: np.ndarray) -> torch.Tensor:
        if depth.ndim == 2:
            depth = depth[:, :, np.newaxis]
        depth_f = (depth.astype(np.float32) - self.depth_mean) / self.depth_std
        return torch.from_numpy(depth_f.transpose(2, 0, 1))  # (1, H, W)

    def __call__(self, rgb, depth):
        if depth.ndim == 2:
            depth_3d = depth[:, :, np.newaxis]
        else:
            depth_3d = depth

        global_crops = []
        global_depth_targets = []
        local_crops = []

        # 2 global crops with matched depth targets
        for i in range(2):
            out = self.global_spatial(image=rgb, depth=depth_3d)
            rgb_colored = (
                self.color_global1 if i == 0 else self.color_global2
            )(image=out["image"])["image"]
            global_crops.append(self._normalize_rgb(rgb_colored))
            global_depth_targets.append(self._normalize_depth(out["depth"]))

        # n_local local crops (RGB only)
        for _ in range(self.n_local):
            rgb_cropped = self.local_spatial(image=rgb)["image"]
            rgb_colored = self.color_local(image=rgb_cropped)["image"]
            local_crops.append(self._normalize_rgb(rgb_colored))

        return {
            "global_crops": global_crops,
            "global_depth_targets": global_depth_targets,
            "local_crops": local_crops,
        }


# ════════════════════════════════════════════════════════════════════════
# WebDataset loading
# ════════════════════════════════════════════════════════════════════════
def _rgb_decoder(key, data):
    if not key.endswith(".png"):
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    except Exception:
        pass
    return None


def _npy_decoder(key, data):
    if not key.endswith(".npy"):
        return None
    try:
        return np.load(io.BytesIO(data), allow_pickle=False).copy()
    except Exception:
        return None


def collate_dinov2(samples, mask_generator, mask_sample_prob=0.5):
    """Stack crops, stack depth targets for globals, generate iBOT masks.

    Output dict:
      - global_crops:          (2*B, 3, Hg, Wg)
      - local_crops:           (n_local*B, 3, Hl, Wl)
      - global_depth_targets:  (2*B, 1, Hg, Wg)
      - masks:                 (2*B, N)  bool  (N = patch grid of global crop)

    Interleaving: [all samples' crop 0, all samples' crop 1, ...]. Depth
    targets follow global_crops ordering so masks align.
    """
    n_global = 2
    n_local = len(samples[0]["local_crops"])
    B = len(samples)

    global_list = [s["global_crops"][i] for i in range(n_global) for s in samples]
    global_crops = torch.stack(global_list)

    depth_list = [
        s["global_depth_targets"][i] for i in range(n_global) for s in samples
    ]
    global_depth_targets = torch.stack(depth_list)

    local_list = [s["local_crops"][i] for i in range(n_local) for s in samples]
    local_crops = torch.stack(local_list)

    total_global = n_global * B
    masks = []
    for _ in range(total_global):
        if random.random() < mask_sample_prob:
            masks.append(torch.from_numpy(mask_generator()))
        else:
            masks.append(torch.zeros(mask_generator.num_patches, dtype=torch.bool))
    masks = torch.stack(masks)

    return {
        "global_crops": global_crops,
        "local_crops": local_crops,
        "global_depth_targets": global_depth_targets,
        "masks": masks,
    }


def make_loader(
    data_path,
    shard_pattern,
    transform,
    batch_size,
    num_workers,
    mask_generator,
    mask_sample_prob=0.5,
    steps_per_epoch=None,
    shuffle_buffer=5000,
):
    """WebDataset loader: decodes paired rgb.png and depth.npy from shards."""
    full_pattern = os.path.join(data_path, shard_pattern)
    shard_urls = sorted(glob.glob(full_pattern))
    if not shard_urls:
        raise FileNotFoundError(f"No shards found at {full_pattern}")

    if is_main():
        print(f"[data] Found {len(shard_urls)} shards matching {full_pattern}")

    collate_fn = partial(
        collate_dinov2,
        mask_generator=mask_generator,
        mask_sample_prob=mask_sample_prob,
    )

    dataset = (
        wds.WebDataset(
            shard_urls, shardshuffle=200, nodesplitter=wds.split_by_node
        )
        .shuffle(shuffle_buffer)
        .decode(_rgb_decoder, _npy_decoder)
        .to_tuple("rgb.png", "depth.npy")
        .map(lambda pair: transform(pair[0], pair[1]))
        .batched(batch_size, partial=False, collation_fn=collate_fn)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    if steps_per_epoch is not None:
        loader = loader.with_epoch(steps_per_epoch)
    return loader
