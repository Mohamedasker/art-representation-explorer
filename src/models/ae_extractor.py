"""
Embeddings from the trained convolutional autoencoder (``src/models/autoencoder.py``).

The embedding is the encoder's latent code.  The full latent of a 256x256 image is
``64 x 8 x 8 = 4096`` numbers; by default it is average-pooled to ``64 x 4 x 4 = 1024``
dimensions (``pool=4``) so that the vectors are comparable in size to the other
representations.  ``pool=0`` keeps the full spatial latent, ``pool=1`` gives a 64-d
global descriptor.

``checkpoint`` is any file written by ``scripts/07_train_autoencoder.py``
(``autoencoder_final.pt``, ``checkpoints/best.pt`` or ``checkpoints/last.pt``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from .autoencoder import Autoencoder, AutoencoderConfig
from .base_extractor import BaseExtractor


class AutoencoderExtractor(BaseExtractor):
    def __init__(self, checkpoint: str | Path, pool: int = 4, img_size: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        if checkpoint is None:
            raise ValueError("--checkpoint <run>/autoencoder_final.pt is required for the autoencoder")
        self.checkpoint = Path(checkpoint)
        self.pool = pool
        self.img_size = img_size
        self._step = None
        self._run = self.checkpoint.resolve().parent
        if self._run.name == "checkpoints":
            self._run = self._run.parent

    def _build_model(self) -> torch.nn.Module:
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._step = ck.get("step")
        model = Autoencoder(AutoencoderConfig.from_dict(ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        return model.to(memory_format=torch.channels_last)

    def _get_transform(self):
        S = self.img_size
        return transforms.Compose([
            transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(S),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            z = self.model.encode(batch)
        z = z.float()
        if self.pool > 0:
            z = F.adaptive_avg_pool2d(z, self.pool)
        return z.flatten(1)

    @property
    def model_name(self) -> str:
        step = f"step{self._step}" if self._step is not None else "?"
        return f"autoencoder_{self._run.name}_{step}"

    @property
    def layer_name(self) -> str:
        return f"latent_pool{self.pool}" if self.pool > 0 else "latent_full"
