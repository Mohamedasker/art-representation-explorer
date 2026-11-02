"""
Embeddings from a StyleGAN2 trained here (``scripts/20_train_gan.py``, ``src/models/gan.py``), 128 px input.

* ``--model gan_disc --checkpoint runs/<run>/gan_final.pt``: the **discriminator's penultimate features**: the 512-D
  output (after the activation) of its fully-connected layer, the input of the real/fake logit layer.  Eval mode, no
  gradients; the minibatch-stddev channel is the training EMA of that statistic, so a vector never depends on the other
  images of its batch.  Not normalised.
* ``--model gan_enc --checkpoint runs/<run>/gan_enc_final.pt``: the **inversion encoder** (``scripts/25_train_gan_encoder.py``,
  generator frozen, trained to reconstruct the image from its latent): the predicted latent w, 512-D.

Fixed pre-processing for every image (the same as in training): resize the shorter side to 128 px (bicubic), centre
crop 128 x 128, scale to [-1, 1].  Files: ``representations/<dataset>_gan_disc_*`` / ``<dataset>_gan_enc_*``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from .base_extractor import BaseExtractor
from .gan import Discriminator, GanConfig, GanEncoder


class GanExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path, kind: str = "disc", **kwargs) -> None:
        super().__init__(**kwargs)
        if kind not in ("disc", "enc"):
            raise ValueError(f"kind must be disc|enc, got {kind!r}")
        if checkpoint is None:
            raise ValueError("--model gan_disc / gan_enc needs --checkpoint runs/<run>/gan_final.pt (resp. gan_enc_final.pt)")
        self.checkpoint, self.kind = Path(checkpoint), kind
        self._ck: dict = {}
        self._run = self.checkpoint.resolve().parent
        if self._run.name == "checkpoints":
            self._run = self._run.parent

    def _build_model(self) -> torch.nn.Module:
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._ck = {k: ck.get(k) for k in ("nimg", "it", "img_size", "gan_run", "step")}
        if self.kind == "disc":
            cfg = GanConfig.from_dict(ck["cfg"])
            self._ck["img_size"] = cfg.img_size
            net = Discriminator(cfg)
            net.load_state_dict(ck["D"])
            return net
        enc = GanEncoder(ck["w_dim"], pretrained=False)
        enc.load_state_dict(ck["E"])
        return enc

    def _get_transform(self):
        if not self._ck:
            self.setup()
        S = int(self._ck["img_size"] or 128)
        return transforms.Compose([
            transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(S), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return (self.model.features(batch) if self.kind == "disc" else self.model(batch)).float()

    @property
    def file_slug(self) -> str:
        return f"gan_{self.kind}"

    @property
    def model_name(self) -> str:
        self.setup()
        if self.kind == "enc":
            return f"stylegan2_{self._ck.get('gan_run')}_inversion_encoder_{self._run.name}_step{self._ck.get('step')}"
        n = f"{self._ck['nimg'] / 1000:.0f}kimg" if self._ck.get("nimg") else "?"
        return f"stylegan2_{self._run.name}_{n}_disc_fc512"

    @property
    def layer_name(self) -> str:
        return "discriminator_fc_features" if self.kind == "disc" else "encoder_w_latent"
