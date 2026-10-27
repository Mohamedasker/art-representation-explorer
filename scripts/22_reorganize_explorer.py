#!/usr/bin/env python3
"""
One-off migration of an existing, flat ``visualizations/explorer/`` to the per-model layout of
src/visualization/variants.py (see "Layout of visualizations/explorer/" in the README):

    <stem>[__key].html   ->  <model>/<stem>[__key].html
    thumbs/, meta/       ->  shared/thumbs/, shared/meta/
    lecture1.html        ->  lecture1/index.html

The pages are patched in place (their template links to the shared files and the overview, and the thumbnail paths
inside them), the navigation lists are re-linked (scripts/relink_explorer.py) and the overview is rewritten from
cache/*/landing.json.  Nothing is recomputed.  Safe to run again: already moved files are left alone.

    python scripts/22_reorganize_explorer.py --dry-run
    python scripts/22_reorganize_explorer.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.variants import model_dir

PAGE = re.compile(r"^(?P<stem>(?:artbench|wikiart)_.+?_(?:all|train|test))(?:__[a-z0-9]+)?\.html$")
# text of the page template / data that changed with the move (each is replaced everywhere it occurs)
PAGE_PATCHES = [
    ('<script src="meta/', '<script src="../shared/meta/'),
    ('<a href="index.html">&#8962; overview</a>', '<a href="../index.html">&#8962; overview</a>'),
    ('`meta/${d.meta}.js: keep the meta/ folder next to this page.</p>`',
     '`shared/meta/${d.meta}.js: keep the shared/ folder next to the model folders.</p>`'),
    ('throw new Error("missing meta/" + d.meta + ".js");', 'throw new Error("missing shared/meta/" + d.meta + ".js");'),
    ('"thumbs/', '"../shared/thumbs/'),
]
META_PATCHES = [('"thumbs/', '"../shared/thumbs/')]


def patch(path: Path, patches: list[tuple[str, str]], dry: bool) -> bool:
    text = path.read_text()
    new = text
    for old, rep in patches:
        new = new.replace(old, rep)
    if new == text:
        return False
    if not dry:
        path.write_text(new)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="visualizations/explorer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, dry = Path(a.dir), a.dry_run

    flat = sorted(f for f in root.glob("*.html") if PAGE.match(f.name))
    moved_pages = [f for f in root.glob("*/*.html") if PAGE.match(f.name)]
    print(f"{len(flat)} flat pages to move, {len(moved_pages)} already in model directories")
    misplaced = [f for f in moved_pages if f.parent.name != model_dir(PAGE.match(f.name)["stem"])]
    print(f"{len(misplaced)} pages in the wrong model directory")
    for f in misplaced:                                     # e.g. after model_dir changed
        if not dry:
            (root / model_dir(PAGE.match(f.name)["stem"])).mkdir(exist_ok=True)
            f.rename(root / model_dir(PAGE.match(f.name)["stem"]) / f.name)

    # 1. shared files: whole-directory renames (same file system, instant)
    if not dry:
        (root / "shared").mkdir(exist_ok=True)
    for name in ("thumbs", "meta"):
        src, dst = root / name, root / "shared" / name
        if src.is_dir() and not dst.exists():
            print(f"move {src} -> {dst}")
            if not dry:
                src.rename(dst)
    meta_dir = root / "shared" / "meta"
    for f in sorted(meta_dir.glob("*.js")) if meta_dir.is_dir() else []:
        if patch(f, META_PATCHES, dry):
            print(f"patched thumbnail paths in {f.relative_to(root)}")

    # 2. pages: move, then patch the template links
    dirs: dict[str, int] = {}
    for f in flat:
        d = model_dir(PAGE.match(f.name)["stem"])
        dirs[d] = dirs.get(d, 0) + 1
        if not dry:
            (root / d).mkdir(exist_ok=True)
            f.rename(root / d / f.name)
    print("pages per model directory:", dict(sorted(dirs.items())))
    if not dry:
        n = sum(patch(root / model_dir(PAGE.match(f.name)["stem"]) / f.name, PAGE_PATCHES, dry) for f in flat)   # misplaced pages are patched already
        print(f"patched {n} pages")

    # 3. lecture 1
    old = root / "lecture1.html"
    if old.exists():
        print(f"move {old.name} -> lecture1/index.html")
        if not dry:
            (root / "lecture1").mkdir(exist_ok=True)
            old.rename(root / "lecture1" / "index.html")
            patch(root / "lecture1" / "index.html", [('class="home" href="index.html"', 'class="home" href="../index.html"')], dry)

    if dry:
        return
    # 4. overview pages: patch the links in place (the overview may contain cards that this checkout's landing.py does not
    #    know, so it is NOT regenerated)
    HREF = re.compile(r'"href":"((?:artbench|wikiart)_[^"/]+)\.html"')
    def to_dir(m: re.Match) -> str:
        stem = re.sub(r"__[a-z0-9]+$", "", m[1])
        return f'"href":"{model_dir(stem)}/{m[1]}.html"'
    for name in ("index.html", "models.html"):
        f = root / name
        if f.exists():
            text = f.read_text()
            new = HREF.sub(to_dir, text).replace('href="lecture1.html"', 'href="lecture1/index.html"')
            if new != text:
                f.write_text(new)
                print(f"patched links in {name}")
    for d in sorted(root.iterdir()):                        # empty leftovers
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            print(f"removed empty {d.name}/")
    print("now run scripts/relink_explorer.py from the checkout that built the pages (its variants.py knows all model types) "
          "to rewrite the navigation links inside the pages")

if __name__ == "__main__":
    main()
