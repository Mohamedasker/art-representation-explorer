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


# ── model types and their weights ─────────────────────────────────────────────
# The overview shows ONE card per model type ("family") and a dataset; inside every page a "Weights" drop-down switches
# between the weights of that model type (e.g. MAE: ImageNet / adapted on ArtBench / from scratch on WikiArt ...).
# (family key, title, [(member key, weights label)]).  A member key is the model part of a page stem
# (``<dataset>_<model part>_<all|train|test>``) or its prefix (SimCLR: ``simclr_artbench_<augmentation set>``).
FAMILIES = [
    ("resnet50", "ResNet-50", [("cnn_supervised", "trained here (supervised)"), ("resnet50_imagenet", "ImageNet")]),
    ("autoencoder", "Convolutional autoencoder", [("autoencoder", "trained here")]),
    ("mae", "MAE ViT-B/16", [("mae_imagenet", "ImageNet (frozen)"),
                             ("mae_adapt_artbench", "ImageNet, adapted on ArtBench"),
                             ("mae_adapt_wikiart", "ImageNet, adapted on WikiArt"),
                             ("mae_scratch_artbench", "from scratch on ArtBench"),
                             ("mae_scratch_wikiart", "from scratch on WikiArt")]),
    ("simclr", "SimCLR ResNet-50", [("simclr_imagenet", "ImageNet (frozen)"),
                                    ("simclr_artbench", "trained on ArtBench"),
                                    ("simclr_wikiart", "trained on WikiArt")]),
    ("dinov2", "DINOv2 ViT-B/14", [("dinov2", "Meta (LVD-142M)")]),
    ("clip", "CLIP ViT-B/32", [("clip", "OpenAI")]),
]
_STEM = re.compile(r"^(artbench|wikiart)_(.+)_(all|train|test)$")


def group_of(stem: str) -> str:
    """Dataset + split kind, e.g. ``artbench_all``: pages of one group show the same images and are navigated together."""
    return f"{stem.split('_')[0]}_{stem.rsplit('_', 1)[-1]}"


def member_of(stem: str) -> tuple[str, str] | None:
    """``(family key, member key)`` of a page stem, or None for a representation that is not in FAMILIES."""
    m = _STEM.match(stem)
    if not m:
        return None
    model = m.group(2)
    for fam, _title, members in FAMILIES:
        for key, _label in members:
            if model == key or model.startswith(key + "_"):
                return fam, key
    return None


# ── where the pages live ──────────────────────────────────────────────────────
# visualizations/explorer/
#   index.html, models.html          overview (root)
#   shared/thumbs, shared/meta       per-image files used by every model page
#   <model dir>/<stem>[__key].html   the pages of one model type (all its weights, both datasets); one directory per
#                                    model so that a course can publish them week by week
#   lecture1/, featuremaps/          stand-alone viewers;  cache/  build intermediates (not needed to view the pages)
# A model page links to the shared files and to other pages with paths relative to itself (it sits one level below
# the root), so the root can be uploaded or served from anywhere.
_DIR_OF_FIRST_WORD = {"cnn": "resnet50", "dinov2": "dino"}       # supervised ResNet-50 weights, DINO v1 + v2 share a family


def model_dir(stem: str) -> str:
    """Directory of the page stem's model type = the family its weights belong to in the explorer's menus (``mae``,
    ``simclr``, ``clip``, ``dino`` ...): the first word of the model name, so it does not depend on which model types
    this file lists in FAMILIES (``ar_vq`` -> ``ar``, ``gan_enc`` -> ``gan``, ``dinov2_vitb14`` -> ``dino``)."""
    m = _STEM.match(stem)
    if not m:
        return "other"
    word = m.group(2).split("_")[0]
    return _DIR_OF_FIRST_WORD.get(word, word)


def page_name(stem: str, key: str = "s1") -> str:
    """File name of view ``key`` of a representation (``s1`` = first sample, the default page)."""
    return f"{stem}.html" if key == "s1" else f"{stem}__{key}.html"


def page_path(stem: str, key: str = "s1") -> str:
    """Path of the page relative to the explorer root."""
    return f"{model_dir(stem)}/{page_name(stem, key)}"


def page_href(from_stem: str, to_stem: str, key: str = "s1") -> str:
    """Link from a page of ``from_stem`` to a page of ``to_stem`` (same file name when both are in one directory)."""
    return page_name(to_stem, key) if model_dir(from_stem) == model_dir(to_stem) else f"../{page_path(to_stem, key)}"


def family_title(fam: str) -> str:
    return next((t for k, t, _m in FAMILIES if k == fam), fam)


def family_order(stem: str) -> tuple[int, int]:
    """Sort key: position of the page's model type, then of its weights."""
    mo = member_of(stem)
    if mo is None:
        return (len(FAMILIES), 0)
    fi = next(i for i, (k, *_r) in enumerate(FAMILIES) if k == mo[0])
    members = [k for k, _l in FAMILIES[fi][2]]
    return (fi, members.index(mo[1]))


def nav_lists(stem: str, key: str, stems, available, href) -> dict:
    """The navigation data embedded in a page.

    ``stems``      all known representation stems; ``available(stem)`` says whether that representation has (or will have)
                   a page for view ``key``; ``href(stem)`` is its page file.
    Returns ``siblings`` (one entry per MODEL TYPE of this dataset; the current type points at the current page, the
    others at their first weights), ``weights`` (the weights of the current model type, for the "Weights" drop-down) and
    ``variants`` (SimCLR augmentation sets of the current weights, for the "Augmentation set" drop-down)."""
    grp, cur = group_of(stem), member_of(stem)
    vi = variant_info(stem)
    reps: dict = {}                              # (family, member) -> page shown for those weights (default variant)
    others = []                                  # representations that are not in FAMILIES
    for o in stems:
        if group_of(o) != grp or not available(o):
            continue
        ov = variant_info(o)
        if ov is not None and not ov["default"]:
            continue
        mo = member_of(o)
        if mo is None:
            others.append(o)
        else:
            reps.setdefault(mo, o)
    siblings, weights = [], []
    for fam, title, members in FAMILIES:
        present = [(k, lb) for k, lb in members if (fam, k) in reps]
        if not present:
            continue
        is_cur = cur is not None and cur[0] == fam
        target = reps[(fam, cur[1])] if is_cur and (fam, cur[1]) in reps else reps[(fam, present[0][0])]
        siblings.append({"name": title, "href": href(target), "current": is_cur})
        if is_cur and len(present) > 1:
            weights = [{"label": lb, "href": href(reps[(fam, k)]), "current": k == cur[1]} for k, lb in present]
    for o in sorted(others):
        siblings.append({"name": o, "href": href(o), "current": o == stem})
    variants = []
    if vi is not None:
        fam_pages = sorted((o for o in stems if (variant_info(o) or {}).get("family") == vi["family"] and available(o)),
                           key=lambda o: variant_info(o)["order"])
        variants = [{"label": variant_info(o)["label"], "href": href(o), "current": o == stem} for o in fam_pages]
    return {"siblings": siblings, "weights": weights, "variants": variants}
