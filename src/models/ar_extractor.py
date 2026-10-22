"""
Embeddings from the autoregressive model (VQ tokenizer + GPT transformer, ``src/models/vq_tokenizer.py`` /
``image_gpt.py``), 128 px input (resize shorter side to 128, centre crop; ``src/models/ar_data.py``).

* ``--model ar_gpt --checkpoint runs/<run>/ar_final.pt`` (slug ``ar_gpt``): the image is tokenised into 16 x 16 = 256
  VQ tokens, the real tokens are fed through the transformer (eval mode, no sampling, prefix = "no label"), and the hidden
  states of ONE intermediate block are averaged over the 256 positions after a non-affine LayerNorm -> 1024-D (iGPT-style
  intermediate-layer features).  The block is the ``best_layer`` of the run's ``layer_probe.json``
  (scripts/24_probe_ar_layers.py: kNN / logistic-regression style accuracy per layer), or ``--ar-layer N``.
* ``--model ar_igpt --checkpoint runs/<run>/ar_final.pt`` (slug ``ar_igpt``, the primary model): iGPT-style pixel model
  (``scripts/27_train_igpt.py``): the 128 px centre crop is average-pooled to 32 x 32, every pixel replaced by its nearest colour
  of the run's 512-colour palette (1024 tokens), the real tokens go through the transformer (eval mode, no sampling, prefix =
  "no label") and the hidden states of ONE intermediate block are averaged over the 1024 positions after a non-affine LayerNorm
  -> 512-D.  The block is probed the same way as for ``ar_gpt``.
* ``--model ar_vq --checkpoint runs/<run>/vq_final.pt`` (slug ``ar_vq``): tokenizer-only baseline -- the quantised
  latents (codebook vectors of the 256 tokens, 32-D, unit norm) average-pooled to a 4 x 4 grid -> 512-D (the same
  spatial pooling as the autoencoder baseline).

Vectors are not normalised.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .ar_data import eval_transform
from .base_extractor import BaseExtractor
from .igpt_data import quantize, shrink
from .image_gpt import GPTConfig, ImageGPT
from .vq_tokenizer import VQConfig, VQTokenizer


def run_dir_of(path: str | Path) -> Path:
    r = Path(path).resolve().parent
    return r.parent if r.name == "checkpoints" else r


def load_vq(path: str | Path, device="cpu") -> VQTokenizer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = VQTokenizer(VQConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def load_gpt(path: str | Path, device="cpu") -> tuple[ImageGPT, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ImageGPT(GPTConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def probed_layer(run_dir: Path) -> int | None:
    f = run_dir / "layer_probe.json"
    return int(json.loads(f.read_text())["best_layer"]) if f.exists() else None


class AutoregressiveExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path, kind: str = "gpt", layer: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if kind not in ("gpt", "vq", "igpt"):
            raise ValueError(f"kind must be gpt|vq|igpt, got {kind!r}")
        if checkpoint is None:
            raise ValueError("--checkpoint is required (ar_final.pt of scripts/23_train_ar.py / vq_final.pt of scripts/21_train_vq.py)")
        self.kind, self.checkpoint = kind, Path(checkpoint)
        self._run = run_dir_of(self.checkpoint)
        self.layer = layer
        self._step = None
        self.vq = None

    def _build_model(self) -> torch.nn.Module:
        if self.kind == "vq":
            m = load_vq(self.checkpoint)
            self._step = torch.load(self.checkpoint, map_location="cpu", weights_only=False).get("step")
            return m
        gpt, ck = load_gpt(self.checkpoint)
        self._step = ck.get("step")
        if self.layer is None:
            self.layer = probed_layer(self._run) or gpt.cfg.n_layer // 2
        if self.kind == "igpt":
            self.palette = torch.tensor(ck["info"]["palette"], device=self.device)
            self.res = int(ck["info"]["res"])
            return gpt
        tok = Path(ck["info"]["tokenizer"])
        self.vq = load_vq(tok if tok.exists() else self._run.parent / tok.parent.name / tok.name, self.device)
        return gpt

    def _get_transform(self):
        return eval_transform()

    @torch.no_grad()
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.kind == "igpt":                  # nearest palette colour in fp32 (bf16 autocast would change the tokens)
            ids = quantize(shrink((batch * 255).round() / 255, self.res), self.palette)
        x = batch * 2 - 1
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if self.kind == "vq":
                z = self.model.quantized_latents(x)                           # (B, 32, 16, 16)
                return F.adaptive_avg_pool2d(z.float(), 4).flatten(1)
            if self.kind == "gpt":
                ids = self.vq.encode(x).flatten(1)
            return self.model.pooled_hidden(ids, [self.layer])[self.layer].float()

    @property
    def file_slug(self) -> str:
        """representations/<dataset>_<slug>_<split>.npz"""
        return f"ar_{self.kind}"

    @property
    def model_name(self) -> str:
        self.setup()
        step = f"step{self._step}" if self._step is not None else "?"
        lay = f"_layer{self.layer}" if self.kind != "vq" else ""
        return f"ar_{self.kind}_{self._run.name}_{step}{lay}"

    @property
    def layer_name(self) -> str:
        if self.kind == "vq":
            return "vq_quantised_latents_pooled_4x4"
        self.setup()
        return f"{'igpt' if self.kind == 'igpt' else 'gpt'}_block{self.layer}_mean_hidden"
