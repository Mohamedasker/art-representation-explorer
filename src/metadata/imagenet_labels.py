"""
ImageNet classes as an explorer colour map, assigned to paintings by CLIP zero-shot classification.

Question behind it: do models pretrained on natural images organise their embedding space by the ImageNet classes
(dogs, vehicles, buildings, ...), even though paintings are far from ImageNet photos?  A painting gets, from CLIP
ViT-L/14 and the standard ImageNet prompt templates, a probability for each of the 1000 ImageNet classes.  From that:

  * ``imagenet_group``  the coarse WordNet group (dogs, other mammals, birds, ..., buildings, vehicles, ...) with the
                        largest total probability; images where no group reaches ``MIN_GROUP_PROB`` are "unknown",
  * ``imagenet_class``  the top-1 class (the most frequent ``TOP_CLASSES`` are named, the rest are "(other)"),
  * ``imagenet_conf``   CLIP's top-1 probability (how ImageNet-like the painting is).

The 1000 class names, wnids and the class -> group table are in ``imagenet_groups.json`` (built once from WordNet with
``build_groups``; nltk is only needed for that).  The CLIP label scores per image are computed by
``scripts/17_label_imagenet.py`` from the CLIP ViT-L/14 image features saved by scripts/15_flag_nudity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUPS_FILE = Path(__file__).with_name("imagenet_groups.json")
LABELS_DIR = Path(__file__).resolve().parents[2] / "data" / "imagenet"
MIN_GROUP_PROB = 0.35        # a painting counts as showing a group of ImageNet classes when CLIP puts this much mass on it
MIN_CLASS_PROB = 0.25        # ... and as showing one class when the top-1 probability is at least this
TOP_CLASSES = 40
GROUP = "ImageNet classes (CLIP zero-shot)"

# (group name, WordNet synsets whose descendants belong to it); the first rule that matches wins
GROUP_RULES = [
    ("dogs", ["dog.n.01"]),
    ("other mammals", ["mammal.n.01"]),
    ("birds", ["bird.n.01"]),
    ("reptiles & amphibians", ["reptile.n.01", "amphibian.n.03"]),
    ("fish", ["fish.n.01"]),
    ("insects, spiders & crustaceans", ["insect.n.01", "arachnid.n.01", "arthropod.n.01"]),
    ("other animals", ["animal.n.01"]),
    ("food & drink", ["food.n.01", "food.n.02", "edible_fruit.n.01", "vegetable.n.01", "beverage.n.01", "dish.n.02"]),
    ("plants & fungi", ["plant.n.02", "fungus.n.01"]),
    ("vehicles", ["vehicle.n.01", "craft.n.02", "conveyance.n.03", "public_transport.n.01"]),
    ("clothing & accessories", ["clothing.n.01", "footwear.n.01", "headdress.n.01", "jewelry.n.01", "neckwear.n.01"]),
    ("buildings & structures", ["structure.n.01"]),
    ("musical instruments", ["musical_instrument.n.01"]),
    ("furniture, household & containers", ["furniture.n.01", "home_appliance.n.01", "container.n.01", "tableware.n.01",
                                           "kitchen_utensil.n.01", "furnishing.n.02", "covering.n.01"]),
    ("tools, machines & devices", ["device.n.01", "implement.n.01", "machine.n.01", "equipment.n.01", "instrument.n.01",
                                   "tool.n.01"]),
    ("landscapes & natural scenes", ["geological_formation.n.01", "natural_object.n.01", "body_of_water.n.01"]),
    ("people", ["person.n.01"]),
    ("other objects", ["entity.n.01"]),
]
GROUP_NAMES = [g for g, _ in GROUP_RULES]


def build_groups(out: Path = GROUPS_FILE) -> dict:
    """Map the 1000 ImageNet classes to GROUP_RULES through WordNet (needs nltk and its 'wordnet' corpus) and save."""
    import nltk
    from nltk.corpus import wordnet as wn
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    from timm.data import ImageNetInfo
    nltk.download("wordnet", quiet=True)
    wnids = ImageNetInfo().label_names()
    rules = [(g, [wn.synset(s) for s in syns]) for g, syns in GROUP_RULES]
    class_group = []
    for wnid in wnids:
        syn = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
        ancestors = {a for path in syn.hypernym_paths() for a in path}
        class_group.append(next(i for i, (_g, syns) in enumerate(rules) if any(s in ancestors for s in syns)))
    data = {"groups": GROUP_NAMES, "classnames": list(IMAGENET_CLASSNAMES), "wnids": wnids, "class_group": class_group}
    out.write_text(json.dumps(data))
    return data


def load_groups() -> dict:
    return json.loads(GROUPS_FILE.read_text())


def labels_file(dataset: str) -> Path:
    return LABELS_DIR / f"labels_{dataset}.npz"


def imagenet_attributes(dataset: str, paths: list[str]):
    """``(defs, arrays)`` of the ImageNet attributes for ``paths`` (image order of the attribute file), or None when the
    label scores of this dataset have not been computed (scripts/17_label_imagenet.py)."""
    from .nudity import image_key
    f = labels_file(dataset)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    row = {str(k): i for i, k in enumerate(z["keys"])}
    sel = np.array([row.get(image_key(p), -1) for p in paths])
    if (sel < 0).any():
        print(f"  imagenet labels: {int((sel < 0).sum())} images missing in {f.name}; skipped")
        return None
    meta = load_groups()
    gp, top1, p1 = z["group_prob"][sel].astype(np.float32), z["top1"][sel].astype(np.int64), z["prob1"][sel].astype(np.float32)
    group = np.where(gp.max(1) >= MIN_GROUP_PROB, gp.argmax(1), -1).astype(np.int16)

    cls = np.where(p1 >= MIN_CLASS_PROB, top1, -1)
    counts = np.bincount(cls[cls >= 0], minlength=1000)
    top = [int(c) for c in np.argsort(-counts)[:TOP_CLASSES] if counts[c] > 0]
    where = {c: i for i, c in enumerate(top)}
    klass = np.array([-1 if c < 0 else where.get(int(c), len(top)) for c in cls], dtype=np.int16)
    cnames = [meta["classnames"][c].split(",")[0] for c in top] + ["(other class)"]

    defs = [
        {"key": "imagenet_group", "label": "ImageNet group (WordNet)", "group": GROUP, "kind": "cat", "names": meta["groups"]},
        {"key": "imagenet_class", "label": f"ImageNet class (top {len(top)})", "group": GROUP, "kind": "cat", "names": cnames},
        {"key": "imagenet_conf", "label": "ImageNet-likeness (CLIP top-1 probability)", "group": GROUP, "kind": "num"},
    ]
    arrays = {"imagenet_group": group, "imagenet_class": klass, "imagenet_conf": p1}
    print(f"  imagenet labels: {int((group >= 0).sum()):,} of {len(group):,} images get a group (>= {MIN_GROUP_PROB} of the "
          f"probability), {int((cls >= 0).sum()):,} a class (top-1 >= {MIN_CLASS_PROB})")
    return defs, arrays


def add_to_attribute_file(attr_file: Path, dataset: str) -> bool:
    """Append (or replace) the ImageNet attributes in an existing attribute file; False if there are no label scores."""
    from .attributes import load_attributes, save_attributes
    paths, defs, arrays = load_attributes(attr_file)
    new = imagenet_attributes(dataset, paths)
    if new is None:
        return False
    keep = [d for d in defs if not d["key"].startswith("imagenet_")]
    ndefs, narrays = new
    save_attributes(attr_file, paths, keep + ndefs, {**{d["key"]: arrays[d["key"]] for d in keep}, **narrays})
    return True
