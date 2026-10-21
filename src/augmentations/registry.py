"""
Config-driven augmentation sets: ``configs/aug_sets.yaml`` -> :class:`TwoViewTransform`.

    from src.augmentations import build_two_view, list_aug_sets
    tf = build_two_view("color_heavy", crop_size=224)   # PIL -> (view1, view2) normalised tensors
"""

from __future__ import annotations

from pathlib import Path

import yaml
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .ops import OPS, Op

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "aug_sets.yaml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_aug_sets(path: str | Path = CONFIG_PATH) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text())


def list_aug_sets(path: str | Path = CONFIG_PATH) -> list[str]:
    return list(load_aug_sets(path))


def _build_op(spec: dict, crop_size: int) -> Op:
    spec = dict(spec)
    name = spec.pop("op")
    if name not in OPS:
        raise ValueError(f"unknown augmentation op {name!r}; known: {sorted(OPS)}")
    if name in ("random_resized_crop", "rotated_resized_crop"):
        spec.setdefault("size", crop_size)
    if name == "one_of":
        spec["ops"] = [_build_op(s, crop_size) for s in spec["ops"]]
    return OPS[name](**spec)


class TwoViewTransform:
    """PIL image -> two independently augmented, normalised tensors ``(v1, v2)``."""

    def __init__(self, ops: list[Op]) -> None:
        self.ops = ops
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def view(self, img):
        kind = "pil"
        for op in self.ops:
            if op.kind != kind:
                img = TF.to_tensor(img) if op.kind == "tensor" else TF.to_pil_image(img)
                kind = op.kind
            img = op(img)
        if kind == "pil":
            img = TF.to_tensor(img)
        return self.norm(img)

    def __call__(self, img):
        return self.view(img), self.view(img)

    def __repr__(self) -> str:
        return "TwoViewTransform(\n  " + "\n  ".join(map(repr, self.ops)) + "\n)"


def build_ops(name: str, crop_size: int = 224, path: str | Path = CONFIG_PATH) -> list[Op]:
    sets = load_aug_sets(path)
    if name not in sets:
        raise ValueError(f"unknown aug set {name!r}; available: {list(sets)}")
    return [_build_op(s, crop_size) for s in sets[name]["ops"]]


def build_two_view(name: str, crop_size: int = 224, path: str | Path = CONFIG_PATH) -> TwoViewTransform:
    return TwoViewTransform(build_ops(name, crop_size, path))
