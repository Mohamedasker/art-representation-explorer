"""Variant families: representations that share ONE overview card and one page, with a drop-down to switch between them.

Currently: the SimCLR models trained here with different augmentation sets.  Page stems look like
``<eval dataset>_simclr_<train dataset>_<aug set>_<all|train|test>`` (e.g. ``artbench_simclr_artbench_color_heavy_all``);
``baseline`` is the default member (its page and card represent the family, the others are reached from the drop-down
"Augmentation set" on every member's page).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_AUG_SETS = Path(__file__).resolve().parents[2] / "configs" / "aug_sets.yaml"
DEFAULT_AUG = "baseline"


def _aug_names() -> list[str]:
    try:
        return list(yaml.safe_load(_AUG_SETS.read_text()))
    except OSError:
        return [DEFAULT_AUG]


def variant_info(stem: str) -> dict | None:
    """``{"family", "label", "default", "order"}`` for a variant page stem, else None."""
    names = _aug_names()
    m = re.fullmatch(r"(artbench|wikiart)_simclr_(artbench|wikiart)_(%s)_(all|train|test)"
                     % "|".join(map(re.escape, sorted(names, key=len, reverse=True))), stem)
    if not m:
        return None
    ev, tr, aug, split = m.groups()
    return {"family": f"{ev}_simclr_{tr}_{split}", "label": aug, "default": aug == DEFAULT_AUG,
            "order": names.index(aug)}
