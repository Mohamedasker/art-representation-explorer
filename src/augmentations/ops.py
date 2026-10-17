"""
Individual SimCLR augmentation operations.

Every op is a small callable class that works on either a PIL image (``kind = "pil"``) or a float tensor
``(3, H, W)`` in [0, 1] (``kind = "tensor"``); :class:`~src.augmentations.registry.TwoViewTransform` inserts the
PIL <-> tensor conversions where the kinds change.  An op with probability ``p`` is applied to a view with that
probability (independently for each of the two views).

``OPS`` maps the op name used in ``configs/aug_sets.yaml`` to its class.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import functional as TF


class Op:
    kind = "pil"          # "pil" or "tensor": the type this op consumes and returns

    def __init__(self, p: float = 1.0) -> None:
        self.p = float(p)

    def __call__(self, x):
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply(x)
        return x

    def apply(self, x):                      # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


# ── crops (always the first op of a pipeline) ─────────────────────────────────

class RandomResizedCrop(Op):
    """Standard SimCLR crop: random area fraction in ``scale`` and aspect ratio in ``ratio``, resized to ``size``."""

    def __init__(self, size: int = 224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), p: float = 1.0) -> None:
        super().__init__(1.0)
        self.size, self.scale, self.ratio = int(size), tuple(scale), tuple(ratio)
        self._t = T.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio,
                                      interpolation=IM.BICUBIC, antialias=True)

    def apply(self, x):
        return self._t(x)


class RotatedResizedCrop(Op):
    """Random resized crop followed by a rotation by a uniform angle in ``degrees`` WITHOUT empty corners.

    A crop of side ``size * sqrt(2)`` is rotated and its central ``size x size`` square is kept, so the whole output
    is real image content for every angle (no black corners that would reveal the rotation angle).
    """

    def __init__(self, size: int = 224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), degrees=(0, 360), p: float = 1.0) -> None:
        super().__init__(1.0)
        self.size, self.scale, self.ratio, self.degrees = int(size), tuple(scale), tuple(ratio), tuple(degrees)
        self._big = int(math.ceil(self.size * math.sqrt(2)))
        self._t = T.RandomResizedCrop(self._big, scale=self.scale, ratio=self.ratio,
                                      interpolation=IM.BICUBIC, antialias=True)

    def apply(self, x):
        x = self._t(x)
        a = random.uniform(*self.degrees)
        x = x.rotate(a, resample=Image.BICUBIC)
        return TF.center_crop(x, [self.size, self.size])


# ── geometric ─────────────────────────────────────────────────────────────────

class HFlip(Op):
    def apply(self, x):
        return ImageOps.mirror(x) if self.kind == "pil" else x.flip(-1)


class VFlip(Op):
    def apply(self, x):
        return ImageOps.flip(x)


class Rot90(Op):
    """Rotate by a random multiple of 90 degrees (lossless)."""

    def apply(self, x):
        return x.rotate(90 * random.randint(0, 3))


class Perspective(Op):
    def __init__(self, distortion: float = 0.5, p: float = 0.5) -> None:
        super().__init__(p)
        self.distortion = distortion
        self._t = T.RandomPerspective(distortion_scale=distortion, p=1.0, interpolation=IM.BICUBIC,
                                      fill=tuple(int(255 * m) for m in (0.485, 0.456, 0.406)))

    def apply(self, x):
        return self._t(x)


class Shear(Op):
    def __init__(self, degrees: float = 20.0, p: float = 0.5) -> None:
        super().__init__(p)
        self.degrees = degrees
        self._t = T.RandomAffine(0, shear=(-degrees, degrees, -degrees, degrees), interpolation=IM.BICUBIC,
                                 fill=tuple(int(255 * m) for m in (0.485, 0.456, 0.406)))

    def apply(self, x):
        return self._t(x)


class Rotate(Op):
    """Small/medium rotation of the (already cropped) view; fills the corners with the mean colour."""

    def __init__(self, degrees: float = 30.0, p: float = 0.5) -> None:
        super().__init__(p)
        self.degrees = degrees

    def apply(self, x):
        return x.rotate(random.uniform(-self.degrees, self.degrees), resample=Image.BICUBIC,
                        fillcolor=tuple(int(255 * m) for m in (0.485, 0.456, 0.406)))


# ── colour ────────────────────────────────────────────────────────────────────

class ColorJitter(Op):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p: float = 0.8) -> None:
        super().__init__(p)
        self.brightness, self.contrast, self.saturation, self.hue = brightness, contrast, saturation, hue
        self._t = T.ColorJitter(brightness, contrast, saturation, hue)

    def apply(self, x):
        return self._t(x)


class Grayscale(Op):
    def apply(self, x):
        return ImageOps.grayscale(x).convert("RGB")


class Solarize(Op):
    def __init__(self, threshold=(64, 192), p: float = 0.2) -> None:
        super().__init__(p)
        self.threshold = tuple(threshold) if isinstance(threshold, (list, tuple)) else (threshold, threshold)

    def apply(self, x):
        return ImageOps.solarize(x, random.randint(*self.threshold))


class Posterize(Op):
    def __init__(self, bits=(2, 5), p: float = 0.2) -> None:
        super().__init__(p)
        self.bits = tuple(bits)

    def apply(self, x):
        return ImageOps.posterize(x, random.randint(*self.bits))


class Invert(Op):
    def apply(self, x):
        return ImageOps.invert(x)


class ChannelShuffle(Op):
    """Random permutation of the R, G, B channels (changes hues, keeps luminance structure approximately)."""

    def apply(self, x):
        ch = list(x.split())
        random.shuffle(ch)
        return Image.merge("RGB", ch)


class ChannelDrop(Op):
    """Replace one random channel (or two, with ``two_p``) by its image-mean grey level."""

    def __init__(self, two_p: float = 0.3, p: float = 0.3) -> None:
        super().__init__(p)
        self.two_p = two_p

    def apply(self, x):
        ch = list(x.split())
        k = 2 if random.random() < self.two_p else 1
        for i in random.sample(range(3), k):
            m = int(np.asarray(ch[i]).mean())
            ch[i] = Image.new("L", x.size, m)
        return Image.merge("RGB", ch)


# ── blur / sharpen / noise ────────────────────────────────────────────────────

class GaussianBlur(Op):
    """Gaussian blur with sigma ~ U(sigma); sigma is given for a 224 px view and scaled with the view size."""

    def __init__(self, sigma=(0.1, 2.0), p: float = 0.5) -> None:
        super().__init__(p)
        self.sigma = tuple(sigma)

    def apply(self, x):
        s = random.uniform(*self.sigma) * x.size[0] / 224.0
        return x.filter(ImageFilter.GaussianBlur(radius=s))


class Sharpen(Op):
    """Unsharp masking (boosts edges and fine texture)."""

    def __init__(self, radius=(1.0, 3.0), percent=(150, 500), p: float = 0.5) -> None:
        super().__init__(p)
        self.radius, self.percent = tuple(radius), tuple(percent)

    def apply(self, x):
        return x.filter(ImageFilter.UnsharpMask(radius=random.uniform(*self.radius),
                                                percent=int(random.uniform(*self.percent)), threshold=0))


class GaussianNoise(Op):
    kind = "tensor"

    def __init__(self, std=(0.02, 0.1), p: float = 0.3) -> None:
        super().__init__(p)
        self.std = tuple(std)

    def apply(self, x):
        return (x + torch.randn_like(x) * random.uniform(*self.std)).clamp_(0, 1)


# ── edge / texture views (tensor ops) ─────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def _gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None, None]            # (1, 1, H, W)


def _conv(g: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), k[None, None])


def _norm01(e: torch.Tensor) -> torch.Tensor:
    e = e - e.amin()
    return e / e.amax().clamp_min(1e-6)


class SobelEdges(Op):
    """Replace the view by its (grey-level) Sobel gradient-magnitude map, a pure edge image."""

    kind = "tensor"

    def apply(self, x):
        g = _gray(x)
        gx, gy = _conv(g, _SOBEL_X), _conv(g, _SOBEL_X.t())
        m = (gx ** 2 + gy ** 2).sqrt()
        m = (m / m.flatten().kthvalue(int(0.99 * m.numel())).values.clamp_min(1e-6)).clamp(0, 1)   # robust max
        return m[0].expand(3, -1, -1).contiguous()


class LaplacianEdges(Op):
    """Replace the view by its absolute Laplacian response (fine edges and texture)."""

    kind = "tensor"

    def apply(self, x):
        m = _conv(_gray(x), _LAPLACE).abs()
        m = (m / m.flatten().kthvalue(int(0.99 * m.numel())).values.clamp_min(1e-6)).clamp(0, 1)
        return m[0].expand(3, -1, -1).contiguous()


class HighPass(Op):
    """Colour high-pass view: image - gaussian-blurred image, re-centred at 0.5 (keeps texture, removes layout/colour)."""

    kind = "tensor"

    def __init__(self, sigma: float = 3.0, gain: float = 3.0, p: float = 0.5) -> None:
        super().__init__(p)
        self.sigma, self.gain = sigma, gain

    def apply(self, x):
        blur = TF.gaussian_blur(x, kernel_size=int(2 * math.ceil(3 * self.sigma) + 1), sigma=self.sigma)
        return (0.5 + self.gain * (x - blur)).clamp(0, 1)


class EdgeEnhance(Op):
    """Add ``alpha`` x the Laplacian edge response to the image (edges exaggerated, colour kept)."""

    kind = "tensor"

    def __init__(self, alpha=(1.0, 3.0), p: float = 0.5) -> None:
        super().__init__(p)
        self.alpha = tuple(alpha)

    def apply(self, x):
        k = _LAPLACE
        e = torch.stack([_conv(x[c][None, None], k)[0, 0] for c in range(3)])
        return (x - random.uniform(*self.alpha) * e).clamp(0, 1)


class PatchShuffle(Op):
    """Cut the view into a ``g x g`` grid (g ~ U{grids}) and shuffle the tiles: destroys global shape/layout,
    keeps local texture, brush strokes and colour statistics."""

    kind = "tensor"

    def __init__(self, grids=(2, 4), p: float = 0.5) -> None:
        super().__init__(p)
        self.grids = tuple(grids)

    def apply(self, x):
        g = random.randint(*self.grids)
        _, H, W = x.shape
        h, w = H // g, W // g
        x = x[:, : h * g, : w * g]
        tiles = x.unfold(1, h, h).unfold(2, w, w).reshape(3, g * g, h, w)          # (3, g*g, h, w)
        tiles = tiles[:, torch.randperm(g * g)]
        out = tiles.reshape(3, g, g, h, w).permute(0, 1, 3, 2, 4).reshape(3, g * h, g * w)
        return F.interpolate(out[None], size=(H, W), mode="bilinear", align_corners=False)[0] if out.shape[-1] != W else out


class RandomErasing(Op):
    kind = "tensor"

    def __init__(self, scale=(0.02, 0.2), p: float = 0.3) -> None:
        super().__init__(p)
        self.scale = tuple(scale)
        self._t = T.RandomErasing(p=1.0, scale=self.scale, value="random")

    def apply(self, x):
        return self._t(x)


class OneOf(Op):
    """Apply exactly one op of a list (chosen with ``weights``) with probability ``p``.  The sub-ops must share a kind."""

    def __init__(self, ops: list, weights=None, p: float = 1.0) -> None:
        super().__init__(p)
        self.ops = ops
        self.weights = list(weights) if weights else [1.0] * len(ops)
        kinds = {o.kind for o in ops}
        if len(kinds) != 1:
            raise ValueError(f"one_of needs ops of a single kind, got {kinds}")
        self.kind = kinds.pop()

    def apply(self, x):
        return random.choices(self.ops, self.weights)[0].apply(x)


OPS: dict[str, type] = {
    "random_resized_crop": RandomResizedCrop, "rotated_resized_crop": RotatedResizedCrop,
    "hflip": HFlip, "vflip": VFlip, "rot90": Rot90, "rotate": Rotate, "perspective": Perspective, "shear": Shear,
    "color_jitter": ColorJitter, "grayscale": Grayscale, "solarize": Solarize, "posterize": Posterize, "invert": Invert,
    "channel_shuffle": ChannelShuffle, "channel_drop": ChannelDrop,
    "gaussian_blur": GaussianBlur, "sharpen": Sharpen, "gaussian_noise": GaussianNoise,
    "sobel": SobelEdges, "laplacian": LaplacianEdges, "high_pass": HighPass, "edge_enhance": EdgeEnhance,
    "patch_shuffle": PatchShuffle, "random_erasing": RandomErasing,
    "one_of": OneOf,
}
