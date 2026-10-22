"""
Decoder-only (GPT-style) transformer over the discrete image tokens of ``src/models/vq_tokenizer.py``.

Sequence (length 2 + 256): ``[dataset token, style token, image token 0 ... image token 255]`` in raster order.  The two
prefix tokens are learned label embeddings with an extra "no label" entry, used for classifier-free-style label dropout
in training; unlabeled images (and all embedding extraction) use "no label" for both.  Position p predicts the token at
position p + 1 (cross-entropy over the image vocabulary only; the prefix tokens are never predicted).

Pre-LayerNorm blocks, learned absolute positions, GELU MLP, causal flash attention (``scaled_dot_product_attention``).

Embeddings (iGPT-style): run the real image tokens through the transformer (eval mode, no sampling) and average the hidden
states of one intermediate block over the 256 image positions (:meth:`ImageGPT.pooled_hidden`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ar_data import STYLE_NAMES

N_PREFIX = 2


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    n_tokens: int = 256
    n_layer: int = 12
    n_head: int = 16
    d_model: int = 1024
    dropout: float = 0.1
    n_datasets: int = 2
    n_styles: int = len(STYLE_NAMES)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.n_head, self.p = cfg.n_head, cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, C)), self.p, self.training)
        return x + F.dropout(self.fc2(F.gelu(self.fc(self.ln2(x)))), self.p, self.training)


class ImageGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.ds_emb = nn.Embedding(cfg.n_datasets + 1, d)        # last index = "no label"
        self.style_emb = nn.Embedding(cfg.n_styles + 1, d)       # last index = "no label"
        self.pos_emb = nn.Parameter(torch.zeros(1, N_PREFIX + cfg.n_tokens, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():                      # GPT-2 residual scaling
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        nn.init.normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def null_ds(self) -> int:
        return self.cfg.n_datasets

    @property
    def null_style(self) -> int:
        return self.cfg.n_styles

    def param_groups(self, weight_decay: float):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if p.ndim >= 2 and "emb" not in n else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def _embed(self, tokens: torch.Tensor, ds: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.ds_emb(ds)[:, None], self.style_emb(style)[:, None], self.tok_emb(tokens)], dim=1)
        return self.drop(x + self.pos_emb[:, : x.shape[1]])

    def forward(self, tokens: torch.Tensor, ds: torch.Tensor, style: torch.Tensor, return_logits: bool = False):
        """Training / validation: ``tokens`` (B, 256) -> mean cross-entropy (nats per image token) of predicting every
        token from the prefix and the tokens before it (the last token is only a target, not an input)."""
        x = self._embed(tokens[:, :-1], ds, style)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x[:, N_PREFIX - 1:]))                    # positions 1..256 predict tokens 0..255
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens.reshape(-1))
        return (loss, logits) if return_logits else loss

    @torch.no_grad()
    def pooled_hidden(self, tokens: torch.Tensor, layers: list[int], ds=None, style=None) -> dict[int, torch.Tensor]:
        """Mean over the 256 image-token positions of the (non-affine LayerNormed) output of block ``l`` for every ``l`` in
        ``layers`` (1-based; ``n_layer`` = last block, before the final LayerNorm).  All 256 tokens are inputs here and
        the prefix is "no label" unless given.  Returns {layer: (B, d_model) float32}."""
        B = tokens.shape[0]
        ds = torch.full((B,), self.null_ds, device=tokens.device) if ds is None else ds
        style = torch.full((B,), self.null_style, device=tokens.device) if style is None else style
        x = self._embed(tokens, ds, style)
        out = {}
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i in layers:
                out[i] = F.layer_norm(x[:, N_PREFIX:].float(), (x.shape[-1],)).mean(1)
        return out

    @torch.no_grad()
    def generate(self, n: int, ds: int | torch.Tensor | None = None, style: int | torch.Tensor | None = None,
                 temperature: float = 1.0, top_k: int = 0, cfg_scale: float = 1.0, generator=None,
                 device=None, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """Sample ``n`` token grids (n, 256) in raster order.  ``ds`` / ``style`` = label ids (None = no label);
        ``cfg_scale`` > 1 applies classifier-free guidance (logits = u + s (c - u)) against the unlabeled model.
        ``prompt`` (n, k) = the first k tokens of every sequence (e.g. the upper half of a real image): they are kept and
        the rest is sampled (completion)."""
        device = device or next(self.parameters()).device

        def lab(v, null):
            if v is None:
                return torch.full((n,), null, device=device, dtype=torch.long)
            return torch.as_tensor(v, device=device).long().expand(n).clone()

        dsv, stv = lab(ds, self.null_ds), lab(style, self.null_style)
        nulls = (torch.full_like(dsv, self.null_ds), torch.full_like(stv, self.null_style))
        guided = cfg_scale != 1.0 and (ds is not None or style is not None)
        toks = torch.zeros(n, 0, dtype=torch.long, device=device) if prompt is None else prompt.to(device).long()
        was_training = self.training
        self.eval()
        for _ in range(self.cfg.n_tokens - toks.shape[1]):
            def last_logits(d_, s_):
                x = self._embed(toks, d_, s_)
                for blk in self.blocks:
                    x = blk(x)
                return self.head(self.ln_f(x[:, -1])).float()
            logits = last_logits(dsv, stv)
            if guided:
                u = last_logits(*nulls)
                logits = u + cfg_scale * (logits - u)
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            toks = torch.cat([toks, nxt], dim=1)
        self.train(was_training)
        return toks
