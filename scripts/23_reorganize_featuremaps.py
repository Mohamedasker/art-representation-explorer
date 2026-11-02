#!/usr/bin/env python3
"""
One-off migration of the feature-map data from one folder per painting to one folder per model (see the docstring of
scripts/11_build_featuremaps.py for the new layout):

    featuremaps/data/<id>/input.jpg                    ->  featuremaps/inputs/<id>.jpg
    featuremaps/data/<id>/<model>__<layer>__*          ->  <model dir>/featuremaps/<model>/<id>/
    featuremaps/data/<id>/recon.jpg                    ->  autoencoder/featuremaps/autoencoder/<id>/recon.jpg
    featuremaps/data/<id>/stats.json                   ->  split by model, one stats.json per model and painting
    featuremaps/data/index.json                        ->  featuremaps/index.json (paintings, model -> dir) +
                                                           <model dir>/featuremaps/<model>/meta.json (title, layers, frame)

Files are renamed (same file system, instant).  Safe to run again: paintings that are already moved are skipped.

    python scripts/23_reorganize_featuremaps.py --dry-run
    python scripts/23_reorganize_featuremaps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.visualization.variants import model_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="visualizations/explorer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, dry = Path(a.dir), a.dry_run
    fm = root / "featuremaps"
    data = fm / "data"
    if not data.is_dir():
        print(f"{data} does not exist: nothing to do")
        return

    index = json.loads((data / "index.json").read_text())
    models = list(index["models"])
    mroot = {m: root / model_dir(f"artbench_{m}_all") / "featuremaps" / m for m in models}
    prefixes = sorted(((f"{m}__", m) for m in models), key=lambda p: -len(p[0]))     # longest first (mae_adapt_* vs mae_*)
    print("models:", {m: str(p.relative_to(root)) for m, p in mroot.items()})

    ids = [d.name for d in sorted(data.iterdir()) if d.is_dir()]
    moved = 0
    for i, pid in enumerate(ids):
        src = data / pid
        stats = json.loads((src / "stats.json").read_text()) if (src / "stats.json").exists() else {}
        if not dry:
            (fm / "inputs").mkdir(exist_ok=True)
            for m in models:
                (mroot[m] / pid).mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.name == "input.jpg":
                dst = fm / "inputs" / f"{pid}.jpg"
            elif f.name == "recon.jpg":
                dst = mroot["autoencoder"] / pid / f.name
            elif f.name == "stats.json":
                continue
            else:
                m = next((m for pre, m in prefixes if f.name.startswith(pre)), None)
                if m is None:
                    raise SystemExit(f"cannot assign {f} to a model")
                dst = mroot[m] / pid / f.name
            if not dry:
                f.rename(dst)
            moved += 1
        if not dry:
            for m in models:
                part = {k: v for k, v in stats.items() if k.startswith(f"{m}/")}
                (mroot[m] / pid / "stats.json").write_text(json.dumps(part))
            (src / "stats.json").unlink(missing_ok=True)
            src.rmdir()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(ids)} paintings")
    print(f"{moved} files {'would be ' if dry else ''}moved for {len(ids)} paintings")
    if dry:
        return

    for m in models:                                                              # meta.json of every model
        (mroot[m] / "meta.json").write_text(json.dumps(index["models"][m]))
    index["models"] = {m: {"dir": model_dir(f"artbench_{m}_all")} for m in models}
    (fm / "index.json").write_text(json.dumps(index))
    (data / "index.json").unlink()
    data.rmdir()
    from src.visualization.featuremaps import write_viewer
    print("viewer:", write_viewer(fm / "index.html"))


if __name__ == "__main__":
    main()
