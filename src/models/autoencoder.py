"""
Convolutional image autoencoder (reconstruction objective).

Encoder: strided conv stages with residual GroupNorm/SiLU blocks, down to a
spatial latent of shape ``(latent_channels, H/2^n, W/2^n)``.  Decoder mirrors it
with nearest-neighbour upsampling + conv.  For 256x256 inputs and the default
config the latent is 64x8x8 = 4096 numbers (48x compression).

Inputs/outputs are in [-1, 1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, ch // 4), num_channels=ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.n1 = _norm(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = _norm(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 8, 8)  # one downsample (x2) per entry
    blocks_per_stage: int = 1
    latent_channels: int = 64

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel_mults"] = list(d["channel_mults"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AutoencoderConfig":
        d = dict(d)
        d["channel_mults"] = tuple(d["channel_mults"])
        return cls(**d)


class Autoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AutoencoderConfig()
        c = self.cfg
        chs = [c.base_channels * m for m in c.channel_mults]

        # encoder
        enc: list[nn.Module] = [nn.Conv2d(c.in_channels, chs[0], 3, padding=1)]
        prev = chs[0]
        for ch in chs:
            for _ in range(c.blocks_per_stage):
                enc.append(ResBlock(prev, ch))
                prev = ch
            enc.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))  # downsample
        enc += [ResBlock(prev, prev), _norm(prev), nn.SiLU(),
                nn.Conv2d(prev, c.latent_channels, 1)]
        self.encoder = nn.Sequential(*enc)

        # decoder
        dec: list[nn.Module] = [nn.Conv2d(c.latent_channels, chs[-1], 3, padding=1),
                                ResBlock(chs[-1], chs[-1])]
        prev = chs[-1]
        for ch in reversed(chs):
            dec.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec.append(nn.Conv2d(prev, ch, 3, padding=1))
            prev = ch
            for _ in range(c.blocks_per_stage):
                dec.append(ResBlock(prev, ch))
        dec += [_norm(prev), nn.SiLU(), nn.Conv2d(prev, c.in_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    @property
    def downsample_factor(self) -> int:
        return 2 ** len(self.cfg.channel_mults)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
