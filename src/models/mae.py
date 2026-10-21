"""
Masked Autoencoder (He et al., 2022, "Masked Autoencoders Are Scalable Vision Learners"), ViT-B/16.

A re-implementation of ``facebookresearch/mae/models_mae.py`` on top of timm's ``Block``/``PatchEmbed`` whose
parameter names match Meta's checkpoints, so the official ImageNet-1k weights load without conversion:

    https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth   (encoder + decoder, 1600 epochs)

(the plain ``mae_pretrain_vit_base.pth`` is the encoder only; the ``_full`` file also carries the decoder, which is
needed to *continue* MAE pre-training.)

Training objective: 75 % of the 16x16 patches are dropped at random, the encoder sees only the visible 25 %, a light
decoder reconstructs the pixels of the masked patches, loss = MSE on the masked patches only.  Targets are
per-patch normalised pixels (``norm_pix_loss``) exactly as in the official ViT-B recipe.

Embeddings (see :class:`src.models.mae_extractor.MAEExtractor`) use the encoder on the full, UNMASKED image.
"""

from __future__ import annotations

import urllib.request
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block

MAE_IMAGENET_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class MAEConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    norm_pix_loss: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── fixed 2-D sin-cos position embeddings (as in the official code) ───────────

def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> np.ndarray:
    gh, gw = np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32))
    grid = np.stack([gw, gh], axis=0).reshape(2, 1, grid_size, grid_size)  # w first, like the reference
    emb = np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, cfg: MAEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MAEConfig()
        self.cfg = cfg
        ln = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        n = self.patch_embed.num_patches
        self.grid = cfg.img_size // cfg.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, cfg.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, qkv_bias=True, norm_layer=ln)
                                     for _ in range(cfg.depth)])
        self.norm = ln(cfg.embed_dim)

        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n + 1, cfg.decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.mlp_ratio,
                                                   qkv_bias=True, norm_layer=ln) for _ in range(cfg.decoder_depth)])
        self.decoder_norm = ln(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_embed_dim, cfg.patch_size ** 2 * 3, bias=True)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.pos_embed.data.copy_(torch.from_numpy(sincos_2d_pos_embed(self.cfg.embed_dim, self.grid)).float()[None])
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(sincos_2d_pos_embed(self.cfg.decoder_embed_dim, self.grid)).float()[None])
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── patches ───────────────────────────────────────────────────────────────

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, _, h, w = imgs.shape
        x = imgs.reshape(b, 3, h // p, p, w // p, p)
        return torch.einsum("nchpwq->nhwpqc", x).reshape(b, (h // p) * (w // p), p * p * 3)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, g = self.cfg.patch_size, int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], g, g, p, p, 3)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], 3, g * p, g * p)

    # ── encoder / decoder ─────────────────────────────────────────────────────

    def random_masking(self, x: torch.Tensor, mask_ratio: float, noise: torch.Tensor | None = None):
        n, length, d = x.shape
        keep = int(length * (1 - mask_ratio))
        if noise is None:
            noise = torch.rand(n, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(n, length, device=x.device)
        mask[:, :keep] = 0
        return x_masked, torch.gather(mask, 1, ids_restore), ids_restore

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float, noise: torch.Tensor | None = None):
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)
        else:
            mask = ids_restore = None
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        n_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.to(x.dtype).repeat(x.shape[0], n_mask, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.decoder_pos_embed.to(x.dtype)
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]

    def target(self, imgs: torch.Tensor) -> torch.Tensor:
        t = self.patchify(imgs)
        if self.cfg.norm_pix_loss:
            mean, var = t.mean(dim=-1, keepdim=True), t.var(dim=-1, keepdim=True)
            t = (t - mean) / (var + 1e-6) ** 0.5
        return t

    def forward_loss(self, imgs, pred, mask) -> torch.Tensor:
        loss = ((pred.float() - self.target(imgs)) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75, noise: torch.Tensor | None = None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, noise)
        pred = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(imgs, pred, mask), pred, mask

    @torch.no_grad()
    def embed(self, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Encoder on the full unmasked image. ``mean`` = average of the 196 patch tokens, ``cls`` = CLS token
        (both after the encoder's final LayerNorm), ``both`` = their concatenation."""
        x, _, _ = self.forward_encoder(imgs, 0.0)
        cls, mean = x[:, 0], x[:, 1:].mean(dim=1)
        return {"mean": mean, "cls": cls, "both": torch.cat([mean, cls], dim=1)}[pool]

    def param_groups(self, weight_decay: float):
        """AdamW groups: no weight decay on biases / norms / tokens (official recipe)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith(".bias") or name in ("cls_token", "mask_token") else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ── pretrained weights ────────────────────────────────────────────────────────

def imagenet_checkpoint_path() -> Path:
    """Local path of Meta's ViT-B MAE checkpoint, downloading it into the torch hub cache if missing
    (needs internet: do the first download on a login node)."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(MAE_IMAGENET_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(MAE_IMAGENET_URL, tmp)
        tmp.rename(path)
    return path


def load_imagenet_mae(strict: bool = True) -> MaskedAutoencoderViT:
    model = MaskedAutoencoderViT(MAEConfig())
    sd = torch.load(imagenet_checkpoint_path(), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=strict)
    return model
