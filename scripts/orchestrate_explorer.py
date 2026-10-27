#!/usr/bin/env python3
"""
Keeps the explorer builds running until every page of the chosen datasets exists, then refreshes the landing page.

One build job (scripts/slurm/explorer_build.sbatch -> scripts/08_build_explorer.py) handles a few models; a job that
hits its time limit simply leaves its finished pages behind and the next round continues (--skip-existing).
At most --max-jobs of them are queued at once.  Prints one line per event; the last line is ALL DONE / GAVE UP.

    python scripts/orchestrate_explorer.py --datasets artbench
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.visualization.variants import page_path


def expected_pages(npz: Path, n_samples: int = 10000) -> list[str]:
    """Page files that scripts/08_build_explorer.py writes for one merged bundle (mirrors subset_defs)."""
    z = np.load(npz, allow_pickle=False)
    paths = [str(p) for p in z["paths"]]
    n = len(paths)
    ns = min(n_samples, n)
    stem = npz.stem
    keys = ["s1"]
    if ns < n and n - ns >= ns // 2:
        keys.append("s2")
    sp = [("train" if "/train/" in p else "test" if "/test/" in p else "") for p in paths]
    for name in ("train", "test"):
        if 0 < sp.count(name) < n:
            keys.append(name)
    if ns < n:
        keys.append("all")
    return [page_path(stem, k) for k in keys]


def missing_models(datasets: list[str], out: Path) -> list[Path]:
    todo = []
    for f in sorted((REPO / "representations").glob("*_all.npz")):
        if f.name.split("_")[0] not in datasets:
            continue
        pages = expected_pages(f)
        landing = out / "cache" / f"{f.stem}__space" / "landing.json"
        if not all((out / p).exists() for p in pages) or not landing.exists():
            todo.append(f)
    return todo


def queued_jobs() -> list[str]:
    r = subprocess.run(["squeue", "-u", __import__("os").environ["USER"], "-h", "-n", "explorer", "-o", "%i"],
                       capture_output=True, text=True)
    return r.stdout.split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["artbench", "wikiart"])
    ap.add_argument("--max-jobs", type=int, default=2)
    ap.add_argument("--no-sequential", dest="sequential", action="store_false",
                    help="by default datasets are built one after the other, in the order given")
    ap.add_argument("--models-per-job", type=int, default=3)
    ap.add_argument("--max-submissions", type=int, default=14)
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--out-dir", default="visualizations/explorer")
    args = ap.parse_args()
    out = REPO / args.out_dir
    state = REPO / "logs" / "orchestrator_assigned.json"      # survives a restart of this watcher
    assigned: dict[str, list[str]] = json.loads(state.read_text()) if state.exists() else {}
    submissions = 0
    while True:
        queue = queued_jobs()
        assigned = {j: m for j, m in assigned.items() if j in queue}
        state.write_text(json.dumps(assigned))
        missing = missing_models(args.datasets, out)
        if args.sequential and missing:                      # finish one dataset completely before starting the next
            first = next(d for d in args.datasets if any(f.name.startswith(d + "_") for f in missing))
            missing = [f for f in missing if f.name.startswith(first + "_")]
        if not missing and not queue:
            subprocess.run([sys.executable, str(REPO / "scripts/rerender_explorer.py")], capture_output=True)
            print("ALL DONE: every explorer page of " + ", ".join(args.datasets) + " is built, landing page refreshed", flush=True)
            return
        busy = {m for ms in assigned.values() for m in ms}
        todo = [f for f in missing if f.stem not in busy]
        while todo and len(queue) < args.max_jobs and submissions < args.max_submissions:
            chunk, todo = todo[:args.models_per_job], todo[args.models_per_job:]
            # spread work: when two jobs can start, give each about half of what is left
            if len(queue) + 1 < args.max_jobs and len(chunk) == args.models_per_job and todo == []:
                pass
            env = dict(__import__("os").environ, ARGS="--repr " + " ".join(f"representations/{f.name}" for f in chunk)
                       + " --skip-existing")
            r = subprocess.run(["sbatch", "--parsable", "scripts/slurm/explorer_build.sbatch"], cwd=REPO, env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                break
            job = r.stdout.strip().split(";")[0]
            assigned[job] = [f.stem for f in chunk]
            state.write_text(json.dumps(assigned))
            queue.append(job)
            submissions += 1
            print(f"submitted job {job}: {', '.join(f.stem for f in chunk)} ({len(missing)} models still missing)", flush=True)
        if submissions >= args.max_submissions and not queue and missing:
            print(f"GAVE UP: {len(missing)} models still missing after {submissions} submissions", flush=True)
            sys.exit(1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
