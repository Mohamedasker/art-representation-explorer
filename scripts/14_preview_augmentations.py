#!/usr/bin/env python3
"""Write a grid per augmentation set: rows = images, columns = original | 3 sampled views.
    python scripts/14_preview_augmentations.py [--dataset wikiart] [--sets baseline color_heavy] -> visualizations/aug_sets/"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_two_view, list_aug_sets
from src.datasets.simclr_data import SplitImages

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="artbench")
p.add_argument("--sets", nargs="+", default=None)
p.add_argument("--n", type=int, default=6)
p.add_argument("--out-dir", default="visualizations/aug_sets")
a = p.parse_args()
ds = SplitImages(a.dataset, "train")
ids = np.random.default_rng(3).choice(len(ds), a.n, replace=False)
mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
orig = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
for name in a.sets or list_aug_sets():
    tv = build_two_view(name)
    torch.manual_seed(0)
    import random; random.seed(0)
    rows = []
    for i in ids:
        img = ds.load(int(i))
        v = [tv.view(img) for _ in range(3)]
        rows += [orig(img)] + [(x * std + mean).clamp(0, 1) for x in v]
    save_image(make_grid(torch.stack(rows), nrow=4, padding=3), f"{a.out_dir}/{a.dataset}_{name}.png")
    print("wrote", f"{a.out_dir}/{a.dataset}_{name}.png")
