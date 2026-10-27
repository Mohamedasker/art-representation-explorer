"""
Lecture 1 page of the explorer: nearest neighbours, interpolation, PCA, t-SNE and UMAP on a toy data set of fish.

Every fish is an ellipse described by two numbers, its length and its thickness (cm); the 3-D version adds the length of
its tail.  The cloud has a major axis of variation (large <-> small fish: long and thick together) and a minor one
(slender <-> stocky, at the same size), so PCA has something to find.  The page itself is static HTML + vanilla JS
(no CDN, works offline): the fish are generated here, everything else - k-NN, interpolation, PCA, t-SNE and UMAP, including
all of their intermediate steps - is computed in the browser by ``algos.js``.

    python scripts/20_build_lecture1.py            # -> visualizations/explorer/lecture1/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = ("algos.js", "plot.js", "explore.js", "pca.js", "embed.js", "app.js")   # concatenated in this order

# major axis (size) and minor axis (slenderness) of the length / thickness cloud, in cm
MEAN_LT = (26.0, 11.0)
SD_MAJOR, SD_MINOR, TILT_DEG = 7.0, 2.3, 24.0
TAIL_MEAN, TAIL_SLOPE, TAIL_SD = 7.0, 0.10, 1.9
# the second population: a cluster of slender fish with extremely long tails (mean, sd in cm; main cloud: tails of 3-12 cm)
CLUSTER_LENGTH, CLUSTER_THICK, CLUSTER_TAIL = (25.0, 3.0), (3.4, 0.5), (22.0, 2.5)


def make_fish(n: int = 150, seed: int = 11) -> np.ndarray:
    """(n, 3) array of [length, thickness, tail length] in cm, rounded to 0.1 cm.  Length and thickness grow together
    along the major axis (tilted by TILT_DEG from the length axis); the tail is long or short independently of that
    (it only grows slightly with the fish)."""
    rng = np.random.default_rng(seed)
    th = np.deg2rad(TILT_DEG)
    u1, u2 = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    rows: list[list[float]] = []
    while len(rows) < n:
        s, t, w = np.clip(rng.standard_normal(3), -2.2, 2.2)
        length, thick = np.array(MEAN_LT) + SD_MAJOR * s * u1 + SD_MINOR * t * u2
        tail = TAIL_MEAN + TAIL_SLOPE * (length - MEAN_LT[0]) + TAIL_SD * w
        if 0.2 <= thick / length <= 0.65 and tail > 2.0:       # drawable fish only
            rows.append([length, thick, tail])
    return np.round(np.array(rows), 1)


def make_fish_with_cluster(n: int = 150, seed: int = 11, n_cluster: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """The same kind of cloud (n - n_cluster fish) plus ``n_cluster`` slender fish with extremely long tails: they differ from
    the rest in the tail by several standard deviations and are slender, so they form an island in the space.
    Returns the (n, 3) fish (shuffled) and the group of every fish (0 = main cloud, 1 = long-tail cluster)."""
    rng = np.random.default_rng(seed + 1000)
    main = make_fish(n - n_cluster, seed)
    cl = np.stack([np.clip(rng.normal(*CLUSTER_LENGTH, n_cluster), 19, 31), np.clip(rng.normal(*CLUSTER_THICK, n_cluster), 2.4, 4.6),
                   np.clip(rng.normal(*CLUSTER_TAIL, n_cluster), 16, 28)], axis=1)
    fish = np.round(np.concatenate([main, cl]), 1)
    group = np.r_[np.zeros(len(main), int), np.ones(n_cluster, int)]
    order = rng.permutation(n)
    return fish[order], group[order]


# four groups: (name, colour, mean [length, thickness, tail], sd [length, thickness, tail]) in cm.  The two round groups have the
# same body and differ only in the tail, so they overlap completely when fish are described by length and thickness alone.
GROUPS4 = [
    ("small short fish", "#8bc34a", (14.0, 5.0, 4.0), (1.8, 0.9, 1.0)),
    ("tall narrow fish", "#ab47bc", (12.0, 15.0, 6.0), (1.5, 1.4, 1.2)),
    ("round fish, short tail", "#26a69a", (25.0, 21.0, 4.0), (2.0, 1.6, 1.0)),
    ("round fish, long tail", "#ec407a", (25.0, 21.0, 17.0), (2.0, 1.6, 1.8)),
]
SEPARATIONS = [("clusters4", "four clusters, clearly separated", 1.0), ("clusters4_close", "four clusters, close together", 0.6),
               ("clusters4_overlap", "four clusters, overlapping", 0.38)]


def make_fish_clusters(n: int = 150, seed: int = 11, sep: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Four groups of fish (see GROUPS4); ``sep`` scales the distances between the group centres (1 = as defined, < 1 = closer
    together, so that the groups touch and overlap).  Returns the (n, 3) fish (shuffled) and the group (0..3) of every fish."""
    rng = np.random.default_rng(seed + 2000)
    centres = np.array([g[2] for g in GROUPS4])
    centres = centres.mean(0) + sep * (centres - centres.mean(0))
    sizes = [n // 4 + (k < n % 4) for k in range(4)]
    fish, group = [], []
    for k, (size, (_, _, _, sd)) in enumerate(zip(sizes, GROUPS4)):
        x = rng.normal(centres[k], sd, (size, 3))
        fish.append(np.maximum(x, [8.0, 3.0, 2.0])); group += [k] * size
    fish, group = np.round(np.concatenate(fish), 1), np.array(group)
    order = rng.permutation(n)
    return fish[order], group[order]


def build_page(path: str | Path, n: int = 150, seed: int = 11) -> Path:
    """Write the lecture page to ``path``."""
    cl_fish, cl_group = make_fish_with_cluster(n, seed)
    sets = {
        "continuum": {"label": "one continuous cloud", "fish": make_fish(n, seed).tolist(), "group": None},
        "cluster": {"label": "cloud + a cluster of long-tail fish", "fish": cl_fish.tolist(), "group": cl_group.tolist(), "kind": "island",
                    "group_names": ["main cloud", "long-tail cluster"], "group_colors": ["#78909c", "#e040fb"], "title": "Second population",
                    "note": f"this data set contains a cluster of {int(cl_group.sum())} slender fish with <b>extremely long tails</b> (colour by <i>population</i> to see them).",
                    "note2": "With only length and thickness they are separated just by being very slender (their tails are drawn but not measured).",
                    "note3": "In the measurement space they sit far from the main cloud along the tail axis (see the 3-D view or a plane with the tail)."},
    }
    for key, label, sep in SEPARATIONS:
        fish, group = make_fish_clusters(n, seed, sep)
        sets[key] = {"label": label, "fish": fish.tolist(), "group": group.tolist(), "kind": "clusters", "group_names": [g[0] for g in GROUPS4],
                     "group_colors": [g[1] for g in GROUPS4], "title": "Four clusters",
                     "note": "this data set has four groups of fish: <b>small short fish</b>, <b>tall narrow fish</b> (taller than long), <b>round fish with a short tail</b> and <b>round fish with a long tail</b> "
                             f"(colour by <i>population</i>; the groups are {'clearly separated' if sep >= 1 else 'close together' if sep >= 0.5 else 'so close that they overlap'}).",
                     "note2": "With only length and thickness the two round groups have the same body and cannot be told apart: only the tail separates them.",
                     "note3": "The two round groups differ only in the tail length: look at the planes with the tail or at the 3-D view."}
    data = {"names": ["length", "thickness", "tail"], "units": "cm", "seed": seed, "sets": sets}
    html = (HERE / "page.html").read_text(encoding="utf-8")
    html = html.replace("__STYLE__", (HERE / "style.css").read_text(encoding="utf-8"))
    html = html.replace("__SCRIPTS__", "\n".join((HERE / f).read_text(encoding="utf-8") for f in SCRIPTS))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
