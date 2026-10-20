#!/usr/bin/env python3
"""Re-link the explorer pages: rewrite the navigation data embedded in every page so it covers ALL representations that
have a page now, not only the ones of the build run that wrote the page:

  * "siblings"  one entry per MODEL TYPE of the page's dataset (header navigation),
  * "weights"   the weights of the page's model type (the "Weights" drop-down: ImageNet / adapted / from scratch ...),
  * "variants"  the augmentation sets of the page's SimCLR weights (the "Augmentation set" drop-down).

Run after building new pages (cheap, no recomputation), and after scripts/rerender_explorer.py:
    python scripts/relink_explorer.py [--dir visualizations/explorer]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.visualization.variants import family_order, nav_lists

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="visualizations/explorer")
d = Path(ap.parse_args().dir)

PAGE = re.compile(r"^(?P<stem>(?:artbench|wikiart)_.+?_(?:all|train|test))(?:__(?P<key>[a-z0-9]+))?\.html$")
pages: dict[tuple[str, str], Path] = {}              # (stem, key) -> file
for f in d.glob("*.html"):
    m = PAGE.match(f.name)
    if m:
        pages[(m["stem"], m["key"] or "s1")] = f


def href(stem: str, key: str) -> str:
    return f"{stem}.html" if key == "s1" else f"{stem}__{key}.html"


def put(text: str, name: str, value: list) -> str:
    """Set the list ``"name":[...]`` of the page's data block, adding it if the page does not have it yet."""
    js = json.dumps(value, separators=(",", ":"))
    if f'"{name}":[' in text:
        return re.sub(rf'"{name}":\[[^\]]*\]', lambda _: f'"{name}":{js}', text, count=1)
    return re.sub(r"(const D = (?:hydrate\()?\{)", lambda m: f'{m.group(1)}"{name}":{js},', text, count=1)


stems = sorted({s for s, _ in pages}, key=lambda s: (family_order(s), s))
n = 0
for (stem, key), f in pages.items():
    nl = nav_lists(stem, key, stems, lambda o: (o, key) in pages, lambda o: href(o, key))
    text = f.read_text()
    new = text
    if '"siblings":[' in text:
        new = put(new, "siblings", nl["siblings"])
    for name in ("weights", "variants"):
        if nl[name] or f'"{name}":[' in new:
            new = put(new, name, nl[name])
    if new != text:
        f.write_text(new)
        n += 1
print(f"relinked {n} of {len(pages)} pages ({len(stems)} representations)")
