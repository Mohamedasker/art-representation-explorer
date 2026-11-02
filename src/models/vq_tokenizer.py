"""
Discrete image tokenizer (VQ-VAE in the style of the VQGAN / ViT-VQGAN autoencoder, plain PyTorch, no custom ops).

128 x 128 x 3 image -> conv encoder (3 stride-2 stages, one self-attention block at the bottleneck) -> 16 x 16 grid of
``code_dim``-D vectors -> l2-normalised, cosine nearest neighbour in a codebook of ``codebook_size`` entries (factorised,
low-dimensional codes as in ViT-VQGAN, which keeps the codebook well used) -> 256 integer tokens -> conv decoder.

Quantiser loss = codebook loss + ``beta`` x commitment loss, straight-through gradient to the encoder.  Codes that are
not used for a while can be re-initialised from encoder outputs (:meth:`VQTokenizer.revive_dead_codes`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQConfig:
    img_size: int = 128
    base_ch: int = 64
    ch_mult: tuple = (1, 2, 4, 8)        # len-1 = number of stride-2 downsamplings (3 -> 16 x 16 tokens at 128 px)
    num_res_blocks: int = 2
    codebook_size: int = 4096
    code_dim: int = 32
    beta: float = 0.25

    @property
    def grid(self) -> int:
        return self.img_size // 2 ** (len(self.ch_mult) - 1)

    @property
    def n_tokens(self) -> int:
        return self.grid ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ch_mult"] = list(self.ch_mult)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VQConfig":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "ch_mult" in d:
            d["ch_mult"] = tuple(d["ch_mult"])
        return cls(**d)


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, c, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.n1, self.c1 = _norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.norm, self.qkv, self.proj = _norm(c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, 1, C, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)                      # (B, 1, HW, C)
        return x + self.proj(h.transpose(-1, -2).reshape(B, C, H, W))


class Encoder(nn.Module):
    def __init__(self, cfg: VQConfig) -> None:
        super().__init__()
        ch = cfg.base_ch
        layers: list[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1)]
        for i, m in enumerate(cfg.ch_mult):
            cout = cfg.base_ch * m
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(ch, cout)); ch = cout
            if i < len(cfg.ch_mult) - 1:
                layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        layers += [ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch), _norm(ch), nn.SiLU(), nn.Conv2d(ch, cfg.code_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: VQConfig) -> None:
        super().__init__()
        ch = cfg.base_ch * cfg.ch_mult[-1]
        layers: list[nn.Module] = [nn.Conv2d(cfg.code_dim, ch, 3, padding=1), ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch)]
        for i, m in reversed(list(enumerate(cfg.ch_mult))):
            cout = cfg.base_ch * m
            for _ in range(cfg.num_res_blocks + 1):
                layers.append(ResBlock(ch, cout)); ch = cout
            if i > 0:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)]
        layers += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VQTokenizer(nn.Module):
    """``forward(x)``: x in [-1, 1], (B, 3, H, W) -> (reconstruction, quantiser loss, token ids (B, grid, grid))."""

    def __init__(self, cfg: VQConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder, self.decoder = Encoder(cfg), Decoder(cfg)
        self.codebook = nn.Parameter(F.normalize(torch.randn(cfg.codebook_size, cfg.code_dim), dim=-1))
        self.register_buffer("usage", torch.ones(cfg.codebook_size))      # EMA of how often each code is chosen per step

    # -- quantisation ------------------------------------------------------------------------------------------
    def _cb(self) -> torch.Tensor:
        return F.normalize(self.codebook.float(), dim=-1)

    def quantize(self, z: torch.Tensor):
        """z: (B, D, h, w) -> (z_q straight-through (B, D, h, w), loss, ids (B, h, w), z_flat (B*h*w, D) normalised)."""
        B, D, h, w = z.shape
        zf = F.normalize(z.float().permute(0, 2, 3, 1).reshape(-1, D), dim=-1)
        cb = self._cb()
        ids = (zf @ cb.t()).argmax(dim=-1)
        zq = cb[ids]
        loss = F.mse_loss(zq, zf.detach()) + self.cfg.beta * F.mse_loss(zf, zq.detach())
        zq_st = zf + (zq - zf).detach()
        return zq_st.reshape(B, h, w, D).permute(0, 3, 1, 2), loss, ids.reshape(B, h, w), zf

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Token ids (B, grid, grid)."""
        return self.quantize(self.encoder(x))[2]

    def quantized_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Codebook vectors of the tokens (B, code_dim, grid, grid) (unit norm)."""
        return self.quantize(self.encoder(x))[0]

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (B, grid, grid) or (B, grid*grid) -> images in [-1, 1]."""
        if ids.dim() == 2:
            g = self.cfg.grid
            ids = ids.reshape(-1, g, g)
        return self.decoder(self._cb()[ids].permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor):
        zq, qloss, ids, zf = self.quantize(self.encoder(x))
        self.last_z = zf.detach()                   # normalised encoder outputs of this batch (for reviving codes)
        return self.decoder(zq), qloss, ids

    # -- codebook maintenance ----------------------------------------------------------------------------------
    @torch.no_grad()
    def update_usage(self, ids: torch.Tensor, decay: float = 0.99) -> None:
        counts = torch.bincount(ids.reshape(-1), minlength=self.cfg.codebook_size).float()
        self.usage.mul_(decay).add_(counts, alpha=1 - decay)

    @torch.no_grad()
    def revive_dead_codes(self, z_pool: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
        """Re-initialise codes whose usage EMA is below ``threshold`` (expected hits per step) with random vectors of
        ``z_pool`` (N, D, normalised encoder outputs).  Returns the indices of the replaced codes."""
        dead = (self.usage < threshold).nonzero().flatten()
        if len(dead) == 0 or len(z_pool) == 0:
            return dead
        pick = torch.randint(0, len(z_pool), (len(dead),), device=z_pool.device)
        self.codebook.data[dead] = z_pool[pick].to(self.codebook.dtype)
        self.usage[dead] = threshold * 10          # grace period
        return dead


class VGGPerceptual(nn.Module):
    """Perceptual distance on frozen torchvision VGG16 features (relu1_2, 2_2, 3_3, 4_3, 5_3), channel-normalised as in
    LPIPS but with unit layer weights, scaled by 0.1 so that typical values are ~0.1-1 like LPIPS.  Inputs in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23], vgg[23:30]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = ((x + 1) / 2 - self.mean) / self.std
        y = ((y + 1) / 2 - self.mean) / self.std
        loss = x.new_zeros(())
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + (F.normalize(x, dim=1) - F.normalize(y, dim=1)).pow(2).sum(1).mean()
        return 0.1 * loss
