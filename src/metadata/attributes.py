"""
Per-image attributes that can be shown as colour maps in the explorer.

An attribute file (``data/attributes/<dataset>_<split>.npz``, written by
``scripts/09_build_attributes.py``) holds ``paths`` (``data/<dataset>/<split>/<style>/<file>``,
the same strings as in the representation bundles), a JSON ``defs`` list describing the
attributes, and one array per attribute:

  kind "cat"  int16 index into ``names`` (``-1`` = unknown)
  kind "num"  float32 (continuous score)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

# ── WikiArt: genre / artist from the cached HuggingFace arrow files ───────────

_HF_ID = "huggan/wikiart"


def wikiart_labels(data_dir: Path) -> dict[str, dict]:
    """``{file_stem: {"genre": str, "artist": str}}`` for every image written by
    ``01_download_wikiart.py`` (file stem = md5("<hf-id>:<split>:<row>")[:16])."""
    import pyarrow as pa

    cache = next(d for d in (data_dir / "_hf_cache" / "huggan___wikiart").glob("default/*/*") if d.is_dir())
    info = json.loads((cache / "dataset_info.json").read_text())
    names = {k: info["features"][k]["names"] for k in ("genre", "artist")}
    shards = sorted(cache.glob("wikiart-train-*.arrow"))
    out: dict[str, dict] = {}
    row = 0
    for shard in shards:
        with pa.memory_map(str(shard)) as src:
            reader = pa.ipc.open_stream(src)
            for batch in reader:       # only the two label columns are touched
                genre = batch.column(batch.schema.get_field_index("genre")).to_pylist()
                artist = batch.column(batch.schema.get_field_index("artist")).to_pylist()
                for g, a in zip(genre, artist):
                    key = hashlib.md5(f"{_HF_ID}:train:{row}".encode()).hexdigest()[:16]
                    out[key] = {"genre": names["genre"][g] if g is not None and g >= 0 else "Unknown Genre",
                                "artist": names["artist"][a] if a is not None and a >= 0 else "Unknown Artist"}
                    row += 1
    return out


# ── ArtBench: artist + keyword tags from the file name ────────────────────────

_ARTIST_RE = re.compile(r"^([a-z0-9][a-z0-9\-\.]*)_")      # "claude-monet_water-lilies-1916"

# tag -> title words (matched as whole words, hyphen/underscore separated)
TITLE_TAGS: dict[str, list[str]] = {
    "portrait": ["portrait", "portraits", "head", "bust", "self-portrait"],
    "woman": ["woman", "women", "girl", "girls", "lady", "ladies", "mother", "maiden", "daughter", "wife",
              "queen", "countess", "madame", "mrs", "miss", "princess", "bride"],
    "man": ["man", "men", "boy", "boys", "gentleman", "father", "king", "sir", "monsieur", "mr", "son",
            "peasant", "soldier", "sailor", "fisherman", "priest"],
    "child": ["child", "children", "baby", "infant", "kids"],
    "nude": ["nude", "nudes", "naked", "bather", "bathers", "bathing", "venus", "odalisque"],
    "religious": ["christ", "madonna", "virgin", "saint", "saints", "st", "annunciation", "crucifixion",
                  "nativity", "mary", "angel", "angels", "apostle", "apostles", "pieta", "holy", "adoration",
                  "lamentation", "deposition", "resurrection", "god", "jesus", "bible", "biblical", "church",
                  "last-supper", "magi", "baptism", "temple", "prophet", "disciples", "cross"],
    "mythology": ["apollo", "venus", "diana", "zeus", "jupiter", "hercules", "mars", "minerva", "bacchus",
                  "dionysus", "cupid", "satyr", "nymph", "nymphs", "mythology", "allegory", "triumph", "muse",
                  "orpheus", "narcissus", "perseus", "juno", "neptune", "psyche"],
    "landscape": ["landscape", "landscapes", "mountain", "mountains", "valley", "hill", "hills", "forest",
                  "field", "fields", "meadow", "tree", "trees", "countryside", "lake", "river", "pond", "path",
                  "road", "sunset", "sunrise", "morning", "evening", "moonlight", "autumn", "winter", "spring",
                  "summer", "snow"],
    "sea / boats": ["sea", "seascape", "boat", "boats", "ship", "ships", "harbour", "harbor", "coast", "beach",
                    "shore", "marine", "port", "sailing", "canal", "bridge", "fishing"],
    "city / buildings": ["city", "town", "street", "streets", "village", "square", "view-of", "cityscape",
                         "houses", "house", "building", "buildings", "palace", "castle", "cathedral", "tower",
                         "market", "paris", "venice", "london", "rome", "florence", "amsterdam", "bridge"],
    "interior": ["interior", "room", "kitchen", "studio", "window", "salon", "bedroom", "table"],
    "still life": ["still-life", "life", "still", "fruit", "fruits", "bottle", "vase", "bouquet"],
    "flowers / garden": ["flower", "flowers", "garden", "gardens", "rose", "roses", "lilies", "lily",
                         "sunflowers", "bouquet", "poppies", "blossom", "blossoms", "iris", "irises"],
    "animals": ["horse", "horses", "dog", "dogs", "cat", "cats", "bird", "birds", "cow", "cows", "sheep",
                "lion", "tiger", "fish", "deer", "animals", "cattle", "swan"],
    "scene / people": ["scene", "party", "dance", "dancers", "dancing", "concert", "feast", "banquet", "family",
                       "group", "procession", "festival", "meeting", "game", "ball", "carnival", "conversation"],
    "battle / war": ["battle", "war", "siege", "soldiers", "army", "victory", "execution", "massacre"],
    "study / sketch": ["study", "sketch", "sketches", "drawing", "design", "illustration", "studies",
                       "figure", "figures", "detail", "fragment", "cartoon"],
    "untitled": ["untitled", "composition", "abstract", "improvisation", "no"],
}


def artbench_artist(stem: str) -> str | None:
    m = _ARTIST_RE.match(stem)
    return m.group(1) if m else None


def artbench_title(stem: str) -> str:
    m = _ARTIST_RE.match(stem)
    return stem[m.end():] if m else stem


def title_tags(stem: str) -> dict[str, bool]:
    """Keyword flags from an ArtBench file stem (``artist_title-words-year``)."""
    title = "-" + re.sub(r"[_ ]+", "-", artbench_title(stem).lower()) + "-"
    flags = {}
    for tag, words in TITLE_TAGS.items():
        flags[tag] = any(f"-{w}-" in title for w in words)
    return flags


# ── helpers for categorical attributes ────────────────────────────────────────

def top_categories(values: list[str | None], n: int, other: str = "(other)", unknown: str = "(unknown)"):
    """Map raw strings to indices: the ``n`` most frequent values, then ``other``;
    None / 'Unknown …' -> -1.  Returns (indices, names)."""
    vals = [None if (v is None or v.lower().startswith("unknown")) else v for v in values]
    counts: dict[str, int] = {}
    for v in vals:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]
    names = top + ([other] if len(counts) > n else [])
    idx = {k: i for i, k in enumerate(top)}
    out = np.array([-1 if v is None else idx.get(v, len(top)) for v in vals], dtype=np.int16)
    return out, names


# ── save / load ───────────────────────────────────────────────────────────────

def save_attributes(path: Path, paths: list[str], defs: list[dict], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")        # write-then-rename: running builds never see a partial file
    np.savez_compressed(tmp, paths=np.array(paths, dtype=str), defs=np.array(json.dumps(defs)),
                        **{f"a__{k}": v for k, v in arrays.items()})
    tmp.replace(path)


def load_attributes(path: Path):
    z = np.load(path, allow_pickle=False)
    defs = json.loads(str(z["defs"]))
    return [str(p) for p in z["paths"]], defs, {d["key"]: z[f"a__{d['key']}"] for d in defs}
