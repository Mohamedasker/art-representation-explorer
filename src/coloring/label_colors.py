"""
Color values derived from dataset labels (art style class index / name).
"""

from __future__ import annotations

import numpy as np

from src.models.base_extractor import RepresentationBundle


def from_labels(bundle: RepresentationBundle) -> dict:
    """
    Assign an integer color code to each sample based on its class label.

    Returns
    -------
    dict with keys:
        ``values``   — int array (N,) with label indices.
        ``names``    — str array (N,) with human-readable style names.
        ``colormap`` — suggested matplotlib colormap name.
        ``label``    — axis / legend label string.
        ``is_categorical`` — True (downstream code uses this flag).
    """
    return {
        "values": bundle.labels,
        "names": bundle.label_names,
        "colormap": "tab10",
        "label": "Art style",
        "is_categorical": True,
    }
