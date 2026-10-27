#!/usr/bin/env python3
"""
Re-render the explorer pages with the current HTML/JS templates (src/visualization/explorer.py and
landing.py) WITHOUT recomputing anything: the data block embedded in each existing page is reused.

Use this after changing only the page design / interaction code.  Per-image data shared by all models of a dataset
is (re)written once to shared/meta/ (see write_meta in src/visualization/explorer.py).

    python scripts/rerender_explorer.py                       # rewrite visualizations/explorer/<model>/*.html in place
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

from src.metadata.nudity import load_flags
from src.visualization.explorer import META_DIR, hydrate, write_explorer, write_meta
from src.visualization import landing
from src.visualization.variants import page_path


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


def refresh_nsfw(page_stem: str, data: dict):
    """Nudity flags (True = flagged) for the page's samples and for the whole embedding, from data/nudity/flags.npz;
    (None, None) if there are no flags or they do not cover this representation."""
    import numpy as np
    rep = REPO_ROOT / "representations" / f"{page_stem.split('__')[0]}.npz"
    if not rep.exists() or "idx" not in data:
        return None, None
    all_paths = np.load(rep, allow_pickle=False)["paths"]
    flags = load_flags(all_paths)
    if flags is None:
        return None, None
    return flags[np.asarray(data["idx"])], flags


def refresh_landing_nsfw(src: Path, page_stem: str, n_page: int, flags) -> None:
    """Hidden-by-default flags for the landing page's preview dots and its per-dataset flagged counts."""
    import numpy as np
    f = src / "cache" / f"{page_stem}__space" / "landing.json"
    if "__" in page_stem or not f.exists():
        return
    d = json.loads(f.read_text())
    sel = np.random.default_rng(0).choice(n_page, size=min(900, n_page), replace=False)   # same dots as scripts/08
    d["summary"]["preview"]["f"] = flags[1][np.asarray(page_idx_cache[page_stem])][sel].astype(int).tolist()
    d["summary"]["nsfw_flagged"], d["summary"]["nsfw_total"] = int(flags[1].sum()), int(len(flags[1]))
    f.write_text(json.dumps(d))


page_idx_cache: dict = {}


def refresh_landing_json(src: Path, page_stem: str, attrs: list) -> None:
    if "__" in page_stem:
        return
    f = src / "cache" / f"{page_stem}__space" / "landing.json"
    if f.exists():
        d = json.loads(f.read_text())
        groups: dict[str, list] = {}
        for a in attrs:
            groups.setdefault(a["group"], []).append(a["label"])
        d["attr_groups"] = groups
        f.write_text(json.dumps(d))


def refresh_featuremaps(src: Path, dst: Path) -> None:
    """Nudity flags for the feature-map viewer's curated test paintings (featuremaps/data/index.json) + current viewer page."""
    f = src / "featuremaps" / "data" / "index.json"
    if not f.exists():
        return
    from src.visualization.featuremaps import write_viewer
    d = json.loads(f.read_text())
    fl = load_flags([f"data/{im['dataset']}/test/{im['style']}/{im['file']}" for im in d["images"]])
    if fl is None:
        print("   featuremaps: no nudity flags for these paintings - explicit images are NOT hidden there")
    else:
        for im, v in zip(d["images"], fl):
            im["nsfw"] = int(v)
        f.write_text(json.dumps(d))
        print(f"   featuremaps: {int(fl.sum())} of {len(fl)} paintings flagged")
    write_viewer(dst / "featuremaps" / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="visualizations/explorer")
    ap.add_argument("--out-dir", default=None, help="default: overwrite in place")
    ap.add_argument("--attr-dir", default="data/attributes", help="attributes that pages may still contain")
    ap.add_argument("--no-index", action="store_true", help="leave index.html alone")
    ap.add_argument("--index-only", action="store_true", help="only rewrite index.html and models.html")
    args = ap.parse_args()
    src, dst = Path(args.dir), Path(args.out_dir or args.dir)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    metas: dict[str, str] = {}               # dataset + split -> meta/<name>.js, made from its full ("__all") page
    # full pages first (they define the shared per-image meta of their dataset), index last
    pages = [*src.glob("[a-z]*/*_*.html"), src / "index.html"]       # model pages (shared/ lecture1/ featuremaps/ have none)
    pages = [q for q in pages if q.exists() and (q.name == "index.html" or re.match(r"(artbench|wikiart)_", q.name))]
    if args.index_only:
        pages = [q for q in pages if q.name == "index.html"]
    for page in sorted(pages, key=lambda q: (q.name == "index.html", not q.stem.endswith("__all"), q.name)):
        text = page.read_text()
        if page.name == "index.html":
            if args.no_index:
                continue
            ns = re.search(r"a random sample of ([\d,]+) images", text)
            entries = [json.loads(f.read_text()) for f in sorted((src / "cache").glob("*__space/landing.json"))
                       if (src / page_path(f.parent.name.removesuffix('__space'))).exists()]
            if not entries:                       # older pages without landing.json: keep the existing data block
                m = re.search(r"const D = (\{.*\});\nconst \$ = id", text, re.S)
                if not m:
                    continue
                html = landing.TEMPLATE.replace("__NS__", ns.group(1) if ns else "10,000").replace("__DATA__", m.group(1))
                (dst / page.name).write_text(html)
            else:
                landing.write_index(dst / page.name, entries, int(ns.group(1).replace(",", "")) if ns else 10000)
        else:
            m = re.search(r"const D = (?:hydrate\()?(\{.*?\})\)?;\nconst N", text, re.S)
            title = re.search(r"<title>(.*?)</title>", text)
            if not m:
                continue
            data = hydrate(json.loads(m.group(1)), src)       # pages already split: put their meta rows back
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
            fl_page, fl_all = refresh_nsfw(page.stem, data)
            if fl_page is not None:
                data["nsfw"] = fl_page.astype(int).tolist()
                page_idx_cache[page.stem] = data["idx"]
                refresh_landing_nsfw(src, page.stem, len(data["idx"]), (fl_page, fl_all))
            else:
                data.pop("nsfw", None)
                print(f"   {page.name}: no nudity flags - explicit images are NOT hidden here")
            base = page.stem.split("__")[0]
            group = f"{base.split('_')[0]}_{base.rsplit('_', 1)[-1]}"
            if group not in metas and page.stem.endswith("__all") and data["idx"] == list(range(len(data["idx"]))):
                metas[group] = write_meta(dst, group, data)
            write_explorer(dst / page.parent.name / page.name, data, title.group(1) if title else page.stem, meta=metas.get(group))
        n += 1
        print("re-rendered", page.name)
    for f in (dst / META_DIR).glob("*.js"):      # meta of earlier renders that no page uses any more
        if f.stem not in metas.values():
            f.unlink()
            print("removed unused", f.relative_to(dst))
    if (src / "featuremaps").exists() and (dst / "featuremaps").exists():
        refresh_featuremaps(src, dst)
    print(f"{n} pages -> {dst}")


if __name__ == "__main__":
    main()
