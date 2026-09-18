"""
Abstract base class for all feature extractors.

Every extractor must:
  * Accept a ``torch.utils.data.DataLoader`` (or any iterable of batches).
  * Return a ``RepresentationBundle`` — a named container holding the
    feature matrix, labels, paths, and metadata.
  * Save / load that bundle to / from  .npz  files.

The .npz format is chosen because it is dependency-free (only NumPy required
to reload), small, and readable by both Python and MATLAB / R.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


# ── RepresentationBundle ──────────────────────────────────────────────────────

@dataclass
class RepresentationBundle:
    """
    Container for a set of image representations.

    Attributes
    ----------
    vectors:
        Float32 array of shape (N, D) — one row per image.
    labels:
        Int64 array of shape (N,) — class index per image.
    label_names:
        String array of shape (N,) — human-readable class name per image.
    paths:
        String array of shape (N,) — absolute path to source image.
    model_name:
        Identifier of the model that produced these vectors.
    layer_name:
        Name of the layer / head from which features were extracted.
    metadata:
        Arbitrary extra key→value pairs (stored as JSON-encoded strings inside
        the npz).
    """

    vectors: np.ndarray                          # shape (N, D)
    labels: np.ndarray                           # shape (N,)  int64
    label_names: np.ndarray                      # shape (N,)  str
    paths: np.ndarray                            # shape (N,)  str
    model_name: str = ""
    layer_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self.vectors)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def class_names(self) -> list[str]:
        """Sorted unique class names."""
        return sorted(set(self.label_names.tolist()))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Save bundle to a compressed ``.npz`` file.

        Parameters
        ----------
        path:
            Destination path.  The ``.npz`` extension is added if absent.

        Returns
        -------
        Path
            Resolved path of the saved file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        import json

        np.savez_compressed(
            path,
            vectors=self.vectors.astype(np.float32),
            labels=self.labels.astype(np.int64),
            label_names=self.label_names.astype(str),
            paths=self.paths.astype(str),
            model_name=np.array(self.model_name, dtype=str),
            layer_name=np.array(self.layer_name, dtype=str),
            metadata_json=np.array(json.dumps(self.metadata), dtype=str),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RepresentationBundle":
        """
        Load a bundle previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file.

        Returns
        -------
        RepresentationBundle
        """
        import json

        path = Path(path)
        data = np.load(path, allow_pickle=False)
        return cls(
            vectors=data["vectors"],
            labels=data["labels"],
            label_names=data["label_names"],
            paths=data["paths"],
            model_name=str(data["model_name"]),
            layer_name=str(data["layer_name"]),
            metadata=json.loads(str(data["metadata_json"])),
        )

    def __repr__(self) -> str:
        return (
            f"RepresentationBundle("
            f"n={self.n_samples}, dim={self.dim}, "
            f"model={self.model_name!r}, layer={self.layer_name!r})"
        )


# ── BaseExtractor ─────────────────────────────────────────────────────────────

class BaseExtractor(ABC):
    """
    Abstract extractor.  Subclasses implement :meth:`_build_model` and
    :meth:`_forward`.

    Parameters
    ----------
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Defaults to best available.
    batch_size:
        Images per forward pass.
    num_workers:
        DataLoader worker processes.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model: Optional[torch.nn.Module] = None

    # ── to be implemented by subclasses ───────────────────────────────────────

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Instantiate and return the backbone (already on correct device)."""

    @abstractmethod
    def _get_transform(self):
        """Return the torchvision transform to apply to raw PIL images."""

    @abstractmethod
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run one batch through the model.

        Parameters
        ----------
        batch:
            Float tensor of shape (B, C, H, W) already on ``self.device``.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape (B, D).
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``"resnet50_imagenet"``)."""

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Name of the layer being extracted (e.g. ``"avgpool"``)."""

    # ── public API ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Build and move the model to the device; call once before extract."""
        if self.model is None:
            self.model = self._build_model().to(self.device).eval()

    def extract(
        self,
        dataset,
        *,
        save_path: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> RepresentationBundle:
        """
        Extract features for an entire dataset.

        Parameters
        ----------
        dataset:
            A ``torch.utils.data.Dataset`` that returns ``(image, label)``
            or ``(image, label, path)`` tuples.
        save_path:
            If given, the bundle is saved to this path as ``.npz``.
        show_progress:
            Show a tqdm progress bar.

        Returns
        -------
        RepresentationBundle
        """
        from tqdm import tqdm

        self.setup()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        all_vectors: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_paths: list[list[str]] = []
        return_path = hasattr(dataset, "return_path") and dataset.return_path

        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=self.model_name, disable=not show_progress):
                if return_path or len(batch) == 3:
                    imgs, labels, paths = batch
                    all_paths.extend(paths)
                else:
                    imgs, labels = batch

                imgs = imgs.to(self.device, non_blocking=True)
                feats = self._forward(imgs)
                all_vectors.append(feats.cpu().float().numpy())
                all_labels.append(labels.numpy())

        elapsed = time.perf_counter() - t0
        vectors = np.concatenate(all_vectors, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        # Build string label array
        if hasattr(dataset, "get_style_names"):
            label_names = np.array(dataset.get_style_names(), dtype=str)
        else:
            label_names = labels_arr.astype(str)

        paths_arr = (
            np.array(all_paths, dtype=str)
            if all_paths
            else np.array(dataset.get_paths() if hasattr(dataset, "get_paths") else [""] * len(labels_arr), dtype=str)
        )

        bundle = RepresentationBundle(
            vectors=vectors,
            labels=labels_arr,
            label_names=label_names,
            paths=paths_arr,
            model_name=self.model_name,
            layer_name=self.layer_name,
            metadata={
                "n_samples": int(len(vectors)),
                "dim": int(vectors.shape[1]),
                "device": str(self.device),
                "extraction_time_s": round(elapsed, 2),
            },
        )

        if save_path is not None:
            out = bundle.save(save_path)
            print(f"  Saved → {out}")

        return bundle
