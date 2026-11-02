"""
StyleGAN2 in pure PyTorch (no custom CUDA ops) + adaptive discriminator augmentation + an inversion encoder.

Follows Karras et al. 2020 (StyleGAN2) / 2020 (ADA) with the pieces that need custom kernels replaced by plain ops:

* modulated convolution with weight demodulation uses the "non-fused" form (scale the input activations by the style,
  shared-weight ``conv2d``, scale the output by the demodulation coefficients) -> ordinary cuDNN convolutions;
* up / down-sampling = zero-insertion / stride-2 depthwise ``conv_transpose2d`` / ``conv2d`` with the fixed [1,3,3,1]
  FIR filter (what ``upfirdn2d`` does), written with ``F.conv*``; the generator upsamples then convolves (instead of a
  transposed modulated convolution);
* the fused bias + leaky-ReLU (x sqrt(2)) is two ordinary ops.

Generator: mapping network (z -> w, 2 layers, lr multiplier 0.01), 4x4 constant, one modulated 3x3 conv + noise per
resolution step (two from 8x8 on), skip architecture (ToRGB at every resolution), style mixing (done in the training
script).  Discriminator: residual architecture, minibatch-stddev layer, 4x4 conv, fc(512) = the **feature** vector that
``gan_disc`` embeddings use, fc -> logit.  The minibatch-stddev scalar makes a single image's output depend on the rest
of its batch; here an EMA of that scalar is kept in a buffer and used in eval mode, so features are per-image.

``AugmentPipe`` is a compact differentiable ADA: per-sample, with probability ``p``: translation, isotropic scale,
rotation, anisotropic scale (one affine ``grid_sample``, reflection padding), brightness, contrast, saturation.

``GanEncoder`` inverts the (frozen) generator: image -> w (512-D), a ResNet-34 trunk (ImageNet init).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)
CHANNELS_LAST = False            # set by the training script (--channels-last): NHWC activations, same numerics, faster on A100
_FIR = [1.0, 3.0, 3.0, 1.0]


@dataclass
class GanConfig:
    img_size: int = 128
    z_dim: int = 512
    w_dim: int = 512
    channel_base: int = 16384        # channels at resolution r: min(channel_base // r, channel_max)
    channel_max: int = 512
    map_layers: int = 2
    mbstd_group: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GanConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def channels(self, res: int) -> int:
        return min(self.channel_base // res, self.channel_max)

    @property
    def resolutions(self) -> list[int]:
        return [2 ** i for i in range(2, int(math.log2(self.img_size)) + 1)]      # 4, 8, ..., img_size

    @property
    def num_ws(self) -> int:
        return 2 + 3 * (len(self.resolutions) - 1)      # 4x4: conv + rgb; every further res: conv, conv, rgb


# ── resampling ────────────────────────────────────────────────────────────────

def upsample2(x: torch.Tensor) -> torch.Tensor:
    """x2 upsampling with the [1,3,3,1] filter of StyleGAN2 (zero-insertion + low-pass == bilinear interpolation with
    weights 3/4, 1/4; the image border replicates instead of fading to zero).  ``F.interpolate`` is far cheaper than a
    depthwise ``conv_transpose2d`` and keeps double back-prop (R1, path length) fast."""
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def _box(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    return F.avg_pool2d(x, 2, stride)


def blur(x: torch.Tensor) -> torch.Tensor:
    """[1,3,3,1]/8 low-pass at constant resolution (reflect padding): [1,3,3,1] = [1,1] * [1,1] * [1,1], so three box
    filters (avg-pooling, native kernels, unlike a depthwise conv)."""
    return _box(_box(_box(F.pad(x, (1, 2, 1, 2), mode="reflect"))))


def downsample2(x: torch.Tensor) -> torch.Tensor:
    """[1,3,3,1]/8 low-pass + stride 2 (zero padding): [1,2,1] box cascade, then a stride-2 box."""
    return _box(_box(_box(F.pad(x, (1, 1, 1, 1)))), 2)


# ── equalised-lr layers ───────────────────────────────────────────────────────

class EqualLinear(nn.Module):
    def __init__(self, i: int, o: int, bias: bool = True, bias_init: float = 0.0, lr_mul: float = 1.0,
                 act: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.bias = nn.Parameter(torch.full([o], float(bias_init))) if bias else None
        self.w_gain, self.b_gain, self.act = lr_mul / math.sqrt(i), lr_mul, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight * self.w_gain, None if self.bias is None else self.bias * self.b_gain)
        return F.leaky_relu(x, 0.2) * SQRT2 if self.act else x


class EqualConv(nn.Module):
    """3x3 / 1x1 conv with runtime weight scaling; optional blur + stride-2 downsampling and fused activation."""

    def __init__(self, i: int, o: int, k: int, bias: bool = True, act: bool = True, down: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, k, k))
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.gain, self.act, self.down, self.pad = 1 / math.sqrt(i * k * k), act, down, k // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.gain
        if self.down:        # blur, then stride-2 conv (== StyleGAN2's FIR-filtered downsampling conv)
            x = F.conv2d(blur(x), w, stride=2, padding=self.pad)
        else:
            x = F.conv2d(x, w, padding=self.pad)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)[None, :, None, None]
        return F.leaky_relu(x, 0.2) * SQRT2 if self.act else x


# ── generator ─────────────────────────────────────────────────────────────────

class ModConv(nn.Module):
    def __init__(self, i: int, o: int, k: int, w_dim: int, demod: bool = True, up: bool = False) -> None:
        super().__init__()
        self.affine = EqualLinear(w_dim, i, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(o, i, k, k))
        self.gain, self.demod, self.up, self.pad = 1 / math.sqrt(i * k * k), demod, up, k // 2

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):                  # styles / demodulation always in fp32
            s = self.affine(w.float())                               # (B, in)
        wt = self.weight * self.gain
        if self.up:
            x = upsample2(x)
        x = F.conv2d(x * s[:, :, None, None].to(x.dtype), wt.to(x.dtype), padding=self.pad)
        if self.demod:
            with torch.autocast("cuda", enabled=False):
                d = (s.square() @ wt.float().square().sum([2, 3]).t() + 1e-8).rsqrt()
            x = x * d[:, :, None, None].to(x.dtype)
        return x


class SynthLayer(nn.Module):
    def __init__(self, i: int, o: int, res: int, w_dim: int, up: bool = False) -> None:
        super().__init__()
        self.conv = ModConv(i, o, 3, w_dim, up=up)
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros(o))
        self.res = res

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: bool = True) -> torch.Tensor:
        x = self.conv(x, w)
        if noise:
            x = x + torch.randn(x.shape[0], 1, self.res, self.res, device=x.device, dtype=x.dtype) * self.noise_strength
        return F.leaky_relu(x + self.bias.to(x.dtype)[None, :, None, None], 0.2) * SQRT2


class ToRGB(nn.Module):
    def __init__(self, i: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModConv(i, 3, 1, w_dim, demod=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.conv(x, w) + self.bias.to(x.dtype)[None, :, None, None]


class Mapping(nn.Module):
    def __init__(self, cfg: GanConfig) -> None:
        super().__init__()
        dims = [cfg.z_dim] + [cfg.w_dim] * cfg.map_layers
        self.layers = nn.ModuleList(EqualLinear(dims[i], dims[i + 1], lr_mul=0.01, act=True)
                                    for i in range(cfg.map_layers))
        self.num_ws = cfg.num_ws
        self.register_buffer("w_avg", torch.zeros(cfg.w_dim))

    def forward(self, z: torch.Tensor, truncation: float = 1.0) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            x = z.float() * (z.float().square().mean(1, keepdim=True) + 1e-8).rsqrt()
            for layer in self.layers:
                x = layer(x)
        if truncation != 1.0:
            x = self.w_avg + (x - self.w_avg) * truncation
        return x


class Generator(nn.Module):
    def __init__(self, cfg: GanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mapping = Mapping(cfg)
        res = cfg.resolutions
        c4 = cfg.channels(4)
        self.const = nn.Parameter(torch.randn(c4, 4, 4))
        self.conv4 = SynthLayer(c4, c4, 4, cfg.w_dim)
        self.rgb4 = ToRGB(c4, cfg.w_dim)
        self.blocks = nn.ModuleList()
        for r in res[1:]:
            i, o = cfg.channels(r // 2), cfg.channels(r)
            self.blocks.append(nn.ModuleDict({
                "conv0": SynthLayer(i, o, r, cfg.w_dim, up=True), "conv1": SynthLayer(o, o, r, cfg.w_dim),
                "rgb": ToRGB(o, cfg.w_dim)}))
        self.num_ws = cfg.num_ws

    def synthesis(self, ws: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """ws: (B, num_ws, w_dim) -> images in [-1, 1]-ish (not clamped), (B, 3, S, S)."""
        x = self.const[None].expand(ws.shape[0], -1, -1, -1)
        if CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv4(x, ws[:, 0], noise)
        img = self.rgb4(x, ws[:, 1])
        i = 2
        for b in self.blocks:
            x = b["conv0"](x, ws[:, i], noise)
            x = b["conv1"](x, ws[:, i + 1], noise)
            img = upsample2(img) + b["rgb"](x, ws[:, i + 2])
            i += 3
        return img

    def forward(self, z: torch.Tensor, truncation: float = 1.0, noise: bool = True) -> torch.Tensor:
        w = self.mapping(z, truncation)
        return self.synthesis(w[:, None].repeat(1, self.num_ws, 1), noise)


# ── discriminator ─────────────────────────────────────────────────────────────

class DiscBlock(nn.Module):
    def __init__(self, i: int, o: int) -> None:
        super().__init__()
        self.conv0 = EqualConv(i, i, 3)
        self.conv1 = EqualConv(i, o, 3, down=True)
        self.skip = EqualConv(i, o, 1, bias=False, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.skip(downsample2(x))
        x = self.conv1(self.conv0(x))
        return (x + y) * math.sqrt(0.5)


class Discriminator(nn.Module):
    def __init__(self, cfg: GanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        res = cfg.resolutions
        self.fromrgb = EqualConv(3, cfg.channels(res[-1]), 1)
        self.blocks = nn.Sequential(*[DiscBlock(cfg.channels(r), cfg.channels(r // 2)) for r in res[::-1][:-1]])
        c = cfg.channels(4)
        self.conv = EqualConv(c + 1, c, 3)
        self.fc = EqualLinear(c * 16, c, act=True)
        self.out = EqualLinear(c, 1)
        self.feature_dim = c
        self.register_buffer("mbstd_ema", torch.ones([]))

    def _mbstd(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        if not self.training:        # eval: constant channel (EMA of the training statistic) -> per-image features
            return torch.cat([x, self.mbstd_ema.to(x.dtype).expand(n, 1, h, w)], 1)
        g = min(self.cfg.mbstd_group, n)
        while n % g:
            g -= 1
        y = x.reshape(g, -1, 1, c, h, w)
        y = (y - y.mean(0)).square().mean(0).add(1e-8).sqrt().mean([2, 3, 4])     # (n/g, 1)
        with torch.no_grad():
            self.mbstd_ema.mul_(0.99).add_(0.01 * y.mean().detach())
        return torch.cat([x, y.reshape(-1, 1, 1, 1).repeat(g, 1, h, w)], 1)

    def features(self, img: torch.Tensor) -> torch.Tensor:
        """Penultimate vector (B, 512): output of the fc layer after the activation (input of the logit layer)."""
        if CHANNELS_LAST:
            img = img.contiguous(memory_format=torch.channels_last)
        x = self.blocks(self.fromrgb(img))
        with torch.autocast("cuda", enabled=False):                  # 4x4 epilogue in fp32
            x = self.conv(self._mbstd(x.float()))
            return self.fc(x.flatten(1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(img)).squeeze(1)


# ── adaptive discriminator augmentation ───────────────────────────────────────

class AugmentPipe(nn.Module):
    """Differentiable per-sample augmentation, each op applied with probability ``p`` (set ``self.p``)."""

    def __init__(self, p: float = 0.0, translate_std: float = 0.125, scale_std: float = 0.2, rotate_max: float = 1.0,
                 aniso_std: float = 0.2, bright_std: float = 0.2, contrast_std: float = 0.5, sat_std: float = 1.0):
        super().__init__()
        self.p = float(p)
        self.t, self.s, self.r, self.a, self.b, self.c, self.sat = (
            translate_std, scale_std, rotate_max, aniso_std, bright_std, contrast_std, sat_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        B, dev = x.shape[0], x.device

        def gate() -> torch.Tensor:
            return (torch.rand(B, device=dev) < self.p).float()

        def rnd() -> torch.Tensor:
            return torch.randn(B, device=dev)

        # one 3x3 affine per sample (coordinates in [-1, 1]); the distributions are symmetric, so no inversion
        m = torch.eye(3, device=dev).repeat(B, 1, 1)
        tx, ty = rnd() * self.t * 2 * gate(), rnd() * self.t * 2 * gate()
        m = m @ _mat(1, 0, tx, 0, 1, ty)
        s = torch.exp2(rnd() * self.s * gate())
        m = m @ _mat(s, 0, 0, 0, s, 0)
        th = (torch.rand(B, device=dev) * 2 - 1) * math.pi * self.r * gate()
        m = m @ _mat(th.cos(), -th.sin(), 0, th.sin(), th.cos(), 0)
        sa = torch.exp2(rnd() * self.a * gate())
        m = m @ _mat(sa, 0, 0, 0, 1 / sa, 0)
        if (m[:, :2] - torch.eye(3, device=dev)[:2]).abs().max() > 0:
            grid = F.affine_grid(m[:, :2], list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        # colour
        x = x + (rnd() * self.b * gate()).view(B, 1, 1, 1)
        mean = x.mean([1, 2, 3], keepdim=True)
        x = (x - mean) * torch.exp2(rnd() * self.c * gate()).view(B, 1, 1, 1) + mean
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * torch.exp2(rnd() * self.sat * gate()).view(B, 1, 1, 1)
        return x


def _mat(a, b, c, d, e, f, dev=None) -> torch.Tensor:
    """Batch of [[a,b,c],[d,e,f],[0,0,1]] from scalars / (B,) tensors."""
    ref = next(v for v in (a, b, c, d, e, f) if torch.is_tensor(v))
    B, dev = ref.shape[0], ref.device
    vals = [(v if torch.is_tensor(v) else torch.full((B,), float(v), device=dev)) for v in (a, b, c, d, e, f)]
    m = torch.zeros(B, 3, 3, device=dev)
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = vals
    m[:, 2, 2] = 1
    return m


# ── inversion encoder (stretch goal) ──────────────────────────────────────────

class GanEncoder(nn.Module):
    """image in [-1, 1] -> w (B, w_dim); ResNet-34 trunk (ImageNet init), output = w_avg + fc(pooled)."""

    def __init__(self, w_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision
        net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.trunk = net
        self.head = nn.Linear(512, w_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("w_avg", torch.zeros(w_dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = ((img + 1) / 2 - self.mean) / self.std
        return self.w_avg + self.head(self.trunk(x))
