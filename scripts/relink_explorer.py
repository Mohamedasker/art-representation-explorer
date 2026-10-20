#!/usr/bin/env python3
"""Re-link the explorer pages: rewrite the "siblings" (model navigation) and "variants" (augmentation-set drop-down)
lists embedded in every page so they cover ALL representations that have a page now, not only the ones of the build run
that wrote the page.  Run after building new pages (cheap, no recomputation):
    python scripts/relink_explorer.py [--dir visualizations/explorer]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.visualization.landing import MODEL_INFO
from src.visualization.variants import variant_info

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="visualizations/explorer")
d = Path(ap.parse_args().dir)

PAGE = re.compile(r"^(?P<stem>(?:artbench|wikiart)_.+?_(?:all|train|test))(?:__(?P<key>[a-z0-9]+))?\.html$")
pages: dict[tuple[str, str], Path] = {}              # (stem, key) -> file
for f in d.glob("*.html"):
    m = PAGE.match(f.name)
    if m:
        pages[(m["stem"], m["key"] or "s1")] = f


def group(stem: str) -> str:
    return f"{stem.split('_')[0]}_{stem.rsplit('_', 1)[-1]}"


def order(stem: str):
    i = next((i for i, (k, *_r) in enumerate(MODEL_INFO) if k in stem), 99)
    return (i, stem)


def href(stem: str, key: str) -> str:
    return f"{stem}.html" if key == "s1" else f"{stem}__{key}.html"


stems = sorted({s for s, _ in pages}, key=order)
n = 0
for (stem, key), f in pages.items():
    vi = variant_info(stem)
    sib = [{"name": o, "href": href(o, key), "current": o == stem or (
        vi is not None and (variant_info(o) or {}).get("family") == vi["family"])}
        for o in stems if group(o) == group(stem) and (o, key) in pages and (variant_info(o) is None or variant_info(o)["default"])]
    text = f.read_text()
    new = text
    if '"siblings":[' in text:
        new = re.sub(r'"siblings":\[[^\]]*\]', lambda _: '"siblings":' + json.dumps(sib, separators=(",", ":")), new, count=1)
    if vi is not None:
        fam = sorted((o for o in stems if (variant_info(o) or {}).get("family") == vi["family"] and (o, key) in pages),
                     key=lambda o: variant_info(o)["order"])
        var = [{"label": variant_info(o)["label"], "href": href(o, key), "current": o == stem} for o in fam]
        if '"variants":[' in new:
            new = re.sub(r'"variants":\[[^\]]*\]', lambda _: '"variants":' + json.dumps(var, separators=(",", ":")), new, count=1)
    if new != text:
        f.write_text(new)
        n += 1
print(f"relinked {n} of {len(pages)} pages ({len(stems)} representations)")
