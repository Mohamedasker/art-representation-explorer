#!/usr/bin/env python3
"""
Interpolate between images in the latent space of the convolutional autoencoder and DECODE the points on
the line, i.e. generate new images that are not in the dataset.

The stored autoencoder embeddings (``representations/*_autoencoder_*.npz``) are the latent average-pooled to
64x4x4 = 1024 numbers; the decoder needs the full 64x8x8 = 4096-number latent, so images are re-encoded and
the interpolation happens in the full latent:  z(t) = (1-t) z_A + t z_B,  x(t) = decoder(z(t)).

Two figures, both from held-out (test) images:
  pairs.png  one block per pair A->B.  Top row: decoded points on the line.  Bottom row: the nearest real
             test image to each point in the stored 1024-d embedding space (what a retrieval-only view of
             the embedding space would show; the ends are the images themselves).
  grid.png   bilinear interpolation between four corner images (a 2-D slice of the latent space).

    python scripts/21_interpolate_autoencoder.py
    python scripts/21_interpolate_autoencoder.py --seed 3 --steps 11 --pairs impressionism:ukiyo_e wikiart/cubism:wikiart/pointillism
    python scripts/21_interpolate_autoencoder.py --no-nn --grid-classes baroque realism romanticism surrealism

A pair is ``[dataset/]class:[dataset/]class`` (dataset = artbench or wikiart, default artbench); the images
are drawn at random from the test split (seeded).  Images flagged for nudity by scripts/15_flag_nudity.py
are never picked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.metadata.nudity import load_flags
from src.models.autoencoder import Autoencoder, AutoencoderConfig

DEFAULT_RUN = "runs/20261015-124529_ae_union"
DEFAULT_PAIRS = ["impressionism:ukiyo_e", "baroque:surrealism", "realism:expressionism",
                 "wikiart/pointillism:wikiart/cubism", "artbench/romanticism:wikiart/pop_art",
                 "impressionism:impressionism"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=f"{DEFAULT_RUN}/autoencoder_final.pt")
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--grid-classes", nargs=4, default=["impressionism", "ukiyo_e", "baroque", "surrealism"],
                   metavar=("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"))
    p.add_argument("--steps", type=int, default=9, help="Points per line, ends included")
    p.add_argument("--grid-size", type=int, default=7)
    p.add_argument("--no-nn", action="store_true", help="Skip the nearest-real-image row")
    p.add_argument("--seed", type=int, default=0, help="Which random images are drawn")
    p.add_argument("--out-dir", default="visualizations/interpolation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

class Pool:
    """Test-split images of both datasets with their stored (pooled) autoencoder embeddings."""

    def __init__(self, model_name: str) -> None:
        vecs, paths, labels, sources = [], [], [], []
        for ds in ("artbench", "wikiart"):
            d = np.load(REPO / "representations" / f"{ds}_autoencoder_test.npz", allow_pickle=True)
            self.matches = str(d["model_name"]) == model_name
            vecs.append(d["vectors"]); paths += d["paths"].tolist()
            labels += d["label_names"].tolist(); sources += [ds] * len(d["paths"])
        self.vecs = np.concatenate(vecs).astype(np.float32)
        self.paths, self.labels, self.sources = np.array(paths), np.array(labels), np.array(sources)
        nsfw = load_flags(self.paths.tolist())
        if nsfw is None:
            print("WARNING: no nudity flags found (scripts/15_flag_nudity.py) - explicit images may be picked")
            nsfw = np.zeros(len(self.paths), bool)
        self.ok = ~nsfw
        print(f"pool: {len(self.paths)} test images, {int(nsfw.sum())} flagged and excluded")

    def pick(self, spec: str, rng: np.random.Generator, exclude: tuple = ()) -> int:
        ds, _, cls = spec.rpartition("/")
        m = self.ok & (self.labels == cls) & ((self.sources == ds) if ds else (self.sources == "artbench"))
        idx = [i for i in np.flatnonzero(m) if i not in exclude]
        if not idx:
            raise SystemExit(f"no images for '{spec}' (classes: artbench/wikiart test labels)")
        return int(rng.choice(idx))


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Autoencoder(AutoencoderConfig.from_dict(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    run = path.resolve().parent.parent.name if path.parent.name == "checkpoints" else path.resolve().parent.name
    return model.to(device).eval(), f"autoencoder_{run}_step{ck['step']}", ck.get("img_size", 256)


@torch.no_grad()
def encode_paths(model, paths, size: int, device: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    x = torch.stack([tf(Image.open(REPO / p).convert("RGB")) for p in paths]).to(device)
    return model.encode(x)  # (n, 64, 8, 8), full latent


@torch.no_grad()
def decode(model, z: torch.Tensor) -> np.ndarray:
    x = (model.decode(z).float().clamp(-1, 1) + 1) / 2
    return x.permute(0, 2, 3, 1).cpu().numpy()  # (n, H, W, 3) in [0, 1]


def line(za: torch.Tensor, zb: torch.Tensor, steps: int) -> torch.Tensor:
    t = torch.linspace(0, 1, steps, device=za.device).view(-1, 1, 1, 1)
    return (1 - t) * za + t * zb


def nearest_real(pool: Pool, z: torch.Tensor) -> list[int]:
    """Nearest test image (L2) in the stored 1024-d embedding space for each latent in ``z``."""
    q = F.adaptive_avg_pool2d(z, 4).flatten(1).cpu().numpy()
    d = (q ** 2).sum(1)[:, None] - 2 * q @ pool.vecs.T + (pool.vecs ** 2).sum(1)[None]
    d[:, ~pool.ok] = np.inf
    return d.argmin(1).tolist()


def load_img(path: str) -> np.ndarray:
    return np.asarray(Image.open(REPO / path).convert("RGB").resize((256, 256), Image.BICUBIC)) / 255.0


# ── figures ───────────────────────────────────────────────────────────────────

def short(pool: Pool, i: int) -> str:
    return f"{pool.sources[i]}/{pool.labels[i]}"


def draw_pairs(rows: list[dict], steps: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = sum(len(r["rows"]) for r in rows)
    fig, axes = plt.subplots(n_rows, steps, figsize=(1.45 * steps, 1.5 * n_rows + 0.9 * len(rows)),
                             squeeze=False)
    k = 0
    for r in rows:
        for j, (label, imgs) in enumerate(r["rows"]):
            for s in range(steps):
                ax = axes[k, s]
                ax.imshow(imgs[s]); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_title(f"t={s / (steps - 1):.2f}", fontsize=7, pad=2)
            axes[k, 0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
            if j == 0:
                axes[k, 0].annotate(r["title"], xy=(0, 1.3), xycoords="axes fraction", fontsize=9,
                                    fontweight="bold", va="bottom")
            k += 1
    fig.subplots_adjust(left=0.1, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.7)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw_grid(imgs: np.ndarray, corners: list[str], n: int, out: Path) -> None:
    tile = imgs.shape[1]
    canvas = np.ones((n * tile + (n - 1) * 3, n * tile + (n - 1) * 3, 3))
    for i in range(n):
        for j in range(n):
            canvas[i * (tile + 3):i * (tile + 3) + tile, j * (tile + 3):j * (tile + 3) + tile] = imgs[i * n + j]
    Image.fromarray((canvas * 255).astype(np.uint8)).save(out)
    print("corners (TL, TR, BL, BR):", ", ".join(corners))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_name, size = load_model(REPO / args.checkpoint, args.device)
    pool = Pool(model_name)
    use_nn = not args.no_nn
    if use_nn and not pool.matches:
        print("WARNING: stored embeddings come from a different checkpoint -> skipping the nearest-image row")
        use_nn = False
    rng = np.random.default_rng(args.seed)
    meta = {"checkpoint": args.checkpoint, "seed": args.seed, "pairs": [], "grid": None}

    # lines between two images
    blocks = []
    for spec in args.pairs:
        a_spec, b_spec = spec.split(":")
        ia = pool.pick(a_spec, rng)
        ib = pool.pick(b_spec, rng, exclude=(ia,))
        za, zb = encode_paths(model, pool.paths[[ia, ib]], size, args.device)
        z = line(za[None], zb[None], args.steps).squeeze(1)
        rows = [("decoded\ninterpolation", decode(model, z))]
        nn_idx = None
        if use_nn:
            nn_idx = nearest_real(pool, z)
            rows.append(("nearest real\nimage", np.stack([load_img(pool.paths[i]) for i in nn_idx])))
        blocks.append({"title": f"{short(pool, ia)}  →  {short(pool, ib)}", "rows": rows})
        # linear interpolation shrinks vectors between the ends: a hint for why mid-points look washed out
        norms = z.flatten(1).norm(dim=1)
        print(f"{blocks[-1]['title']}:  |z| ends {norms[0]:.1f} / {norms[-1]:.1f}, mid {norms[args.steps // 2]:.1f}")
        meta["pairs"].append({"a": pool.paths[ia].item(), "b": pool.paths[ib].item(),
                              "nearest": [pool.paths[i].item() for i in nn_idx] if nn_idx else None})
    draw_pairs(blocks, args.steps, out_dir / "pairs.png")

    # 2-D bilinear grid between four corner images
    corner_idx = []
    for spec in args.grid_classes:
        corner_idx.append(pool.pick(spec, rng, exclude=tuple(corner_idx)))
    zc = encode_paths(model, pool.paths[corner_idx], size, args.device)
    n = args.grid_size
    u = torch.linspace(0, 1, n, device=args.device)
    uu, vv = torch.meshgrid(u, u, indexing="xy")  # uu: left->right, vv: top->bottom
    w = torch.stack([(1 - uu) * (1 - vv), uu * (1 - vv), (1 - uu) * vv, uu * vv]).flatten(1)  # (4, n*n)
    zg = torch.einsum("kp,kchw->pchw", w, zc)
    draw_grid(decode(model, zg), [short(pool, i) for i in corner_idx], n, out_dir / "grid.png")
    meta["grid"] = [pool.paths[i].item() for i in corner_idx]

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}/pairs.png, grid.png, meta.json")


if __name__ == "__main__":
    main()
