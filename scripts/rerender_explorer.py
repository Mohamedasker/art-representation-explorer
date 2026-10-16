#!/usr/bin/env python3
"""
Re-render the explorer pages with the current HTML/JS templates (src/visualization/explorer.py and
landing.py) WITHOUT recomputing anything: the data block embedded in each existing page is reused.

Use this after changing only the page design / interaction code.

    python scripts/rerender_explorer.py                       # rewrite visualizations/explorer/*.html in place
    python scripts/rerender_explorer.py --out-dir /tmp/test   # write the re-rendered pages elsewhere
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.explorer import write_explorer
from src.visualization import landing


def current_attr_keys(page_stem: str, attr_dir: Path):
    """Attribute keys defined in the current attribute file for this page's dataset and split (None if unknown)."""
    import numpy as np
    m = re.match(r"(artbench|wikiart)_.*_(train|test|all)(?:__\w+)?$", page_stem)
    f = attr_dir / f"{m.group(1)}_{m.group(2)}.npz" if m else None
    if f is None or not f.exists():
        return None
    return {d["key"] for d in json.loads(str(np.load(f, allow_pickle=False)["defs"]))}


_B08 = None


def refresh_attrs(page_stem: str, data: dict, attr_dir: Path):
    """Re-encode the page's attributes from the current attribute files (None if they cannot be aligned)."""
    global _B08
    import importlib.util
    import numpy as np
    base = page_stem.split("__")[0]
    rep = REPO_ROOT / "representations" / f"{base}.npz"
    if not rep.exists() or "idx" not in data:
        return None
    if _B08 is None:
        spec = importlib.util.spec_from_file_location("b08", REPO_ROOT / "scripts" / "08_build_explorer.py")
        _B08 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_B08)
    paths = np.load(rep, allow_pickle=False)["paths"][np.asarray(data["idx"])]
    out = _B08.load_attrs(base, paths, attr_dir)
    return out if out else None


def refresh_landing_json(src: Path, page_stem: str, attrs: list) -> None:
    if "__" in page_stem:
        return
    f = src / "cache" / page_stem / "landing.json"
    if f.exists():
        d = json.loads(f.read_text())
        groups: dict[str, list] = {}
        for a in attrs:
            groups.setdefault(a["group"], []).append(a["label"])
        d["attr_groups"] = groups
        f.write_text(json.dumps(d))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="visualizations/explorer")
    ap.add_argument("--out-dir", default=None, help="default: overwrite in place")
    ap.add_argument("--attr-dir", default="data/attributes", help="attributes that pages may still contain")
    ap.add_argument("--no-index", action="store_true", help="leave index.html alone")
    args = ap.parse_args()
    src, dst = Path(args.dir), Path(args.out_dir or args.dir)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for page in sorted(src.glob("*.html")):
        text = page.read_text()
        if page.name == "index.html":
            if args.no_index:
                continue
            ns = re.search(r"a random sample of ([\d,]+) images", text)
            entries = [json.loads(f.read_text()) for f in sorted((src / "cache").glob("*/landing.json"))
                       if (src / f"{f.parent.name}.html").exists()]
            if not entries:                       # older pages without landing.json: keep the existing data block
                m = re.search(r"const D = (\{.*\});\nconst \$ = id", text, re.S)
                if not m:
                    continue
                html = landing.TEMPLATE.replace("__NS__", ns.group(1) if ns else "10,000").replace("__DATA__", m.group(1))
                (dst / page.name).write_text(html)
            else:
                landing.write_index(dst / page.name, entries, int(ns.group(1).replace(",", "")) if ns else 10000)
        else:
            m = re.search(r"const D = (\{.*?\});\nconst N", text, re.S)
            title = re.search(r"<title>(.*?)</title>", text)
            if not m:
                continue
            data = json.loads(m.group(1))
            for st in data.get("sets", []):                  # labels: first sample = set 1, "__setN" = set N+1
                mm = re.search(r"__set(\d+)\.html$", st["href"])
                if mm:
                    st["label"] = re.sub(r"\(set \d+\)", f"(set {int(mm.group(1)) + 1})", st["label"])
            refreshed = refresh_attrs(page.stem, data, Path(args.attr_dir))
            if refreshed is not None:
                if len(refreshed) != len(data.get("attrs", [])) or \
                        {a["key"] for a in refreshed} != {a["key"] for a in data.get("attrs", [])}:
                    print(f"   {page.name}: attributes updated ({len(data.get('attrs', []))} -> {len(refreshed)})")
                data["attrs"] = refreshed
                refresh_landing_json(src, page.stem, refreshed)
            write_explorer(dst / page.name, data, title.group(1) if title else page.stem)
        n += 1
        print("re-rendered", page.name)
    print(f"{n} pages -> {dst}")


if __name__ == "__main__":
    main()
