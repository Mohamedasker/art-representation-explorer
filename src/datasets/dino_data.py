"""Data for DINO training (scripts/29_train_dino.py): the uint8 256 px caches of scripts/29_cache_images.py and the DINO
multi-crop augmentation (Caron et al. 2021, main_dino.py DataAugmentationDINO)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache256"
BICUBIC = transforms.InterpolationMode.BICUBIC


class CachedImages(Dataset):
    """256x256 uint8 cache of one dataset split; ``transform(PIL image)`` -> item, plus the style label."""

    def __init__(self, dataset: str, split: str, transform=None, cache_dir: str | Path = CACHE_DIR) -> None:
        cache_dir = Path(cache_dir)
        self.arr = np.load(cache_dir / f"{dataset}_{split}_256.npy", mmap_mode="r")
        meta = json.loads((cache_dir / f"{dataset}_{split}_meta.json").read_text())
        self.paths: list[str] = meta["paths"]
        self.labels = np.asarray(meta["labels"], dtype=np.int64)
        self.class_names: list[str] = meta["class_names"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int):
        img = Image.fromarray(np.ascontiguousarray(self.arr[idx]))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p, self.rmin, self.rmax = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.rmin, self.rmax)))


class Solarization:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


def to_uint8_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> uint8 (C, H, W); normalisation happens on the GPU (4x less data through the worker queues)."""
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


class DINOAugmentation:
    """2 global crops (224 px, scale 0.4-1; blur p=1.0 / blur p=0.1 + solarize p=0.2) and ``n_local`` local crops (96 px,
    scale 0.05-0.4; blur p=0.5); flip, colour jitter (0.4, 0.4, 0.2, 0.1) with p=0.8, grayscale p=0.2 as in DINO."""

    def __init__(self, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4), n_local: int = 6, global_size: int = 224,
                 local_size: int = 96) -> None:
        flip_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=BICUBIC), flip_color,
            GaussianBlur(1.0), to_uint8_tensor])
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=BICUBIC), flip_color,
            GaussianBlur(0.1), Solarization(0.2), to_uint8_tensor])
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale, interpolation=BICUBIC), flip_color,
            GaussianBlur(0.5), to_uint8_tensor])
        self.n_local = n_local

    def __call__(self, img):
        return [self.global1(img), self.global2(img)] + [self.local(img) for _ in range(self.n_local)]


def eval_transform_uint8(size: int = 224):
    """Centre crop of the cache (= the evaluation view Resize 256 + CenterCrop 224), uint8 (C, H, W)."""
    return transforms.Compose([transforms.CenterCrop(size), to_uint8_tensor])
