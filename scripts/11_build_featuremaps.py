#!/usr/bin/env python3
"""
Precompute the data of the feature-map viewer (visualizations/explorer/featuremaps/index.html).

For a curated set of held-out (test split) paintings it records the internal feature maps of
  * the supervised ResNet-50 trained here          (stem, layer1-4)
  * the ImageNet ResNet-50                         (stem, layer1-4)
  * the convolutional autoencoder                  (5 encoder stages, latent code, reconstruction)
  * the masked autoencoders (ImageNet MAE ViT-B/16 and the ones trained here; encoder on the unmasked image, 14x14 grid)
  * DINOv2 ViT-B/14 and CLIP ViT-B/32 (image tower) (patch tokens of every second transformer block, reshaped to a
                                                    grid, plus the CLS-token attention map of that block)
and stores per image and layer
  * a sprite with the 64 most varied channels (8 x 8 tiles of 64 px, grey scale; the viewer applies colour maps),
  * a sprite with the 16 strongest principal components of the layer (4 x 4 tiles): usually far easier to read than
    individual channels,
  * the first three principal components of the layer's activations as an RGB image,
  * the mean activation map (overlaid on the image in the viewer),
  * transformers only: the CLS attention of every head (sprite), the head average, and the attention rollout
    (attention flow from the input up to this block, Abnar & Zuidema 2020),
  * a few statistics (shape, sparsity, mean, channel ids).

    python scripts/11_build_featuremaps.py                 # 10 images per ArtBench style, 4 per WikiArt style
    python scripts/11_build_featuremaps.py --per-style-artbench 20 --per-style-wikiart 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.autoencoder import Autoencoder, AutoencoderConfig
from src.models.supervised_cnn import MultiHeadCNN

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TILE, GRID = 64, 8            # sprite: GRID x GRID tiles of TILE px  (-> 512 x 512)


def load_image(path: str, size: int = 256) -> Image.Image:
    im = Image.open(path)
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def pick_images(per_style: dict[str, int], seed: int) -> list[dict]:
    """Held-out (test split) images, evenly across styles, deterministic."""
    out = []
    rng = np.random.default_rng(seed)
    for ds, n_per in per_style.items():
        z = np.load(REPO_ROOT / f"representations/{ds}_clip_vitb32_test.npz", allow_pickle=False)
        paths, styles = z["paths"].astype(str), z["label_names"].astype(str)
        for style in sorted(set(styles.tolist())):
            idx = np.flatnonzero(styles == style)
            for i in rng.choice(idx, size=min(n_per, len(idx)), replace=False):
                p = Path(paths[i])
                out.append({"id": f"{ds}_{p.stem}", "dataset": ds, "style": style, "path": str(p), "file": p.name})
    return out


# ── models ────────────────────────────────────────────────────────────────────

class Taps:
    """Forward hooks that keep the output of chosen modules."""

    def __init__(self, modules: dict[str, nn.Module]):
        self.out: dict[str, torch.Tensor] = {}
        self.handles = [m.register_forward_hook(lambda _m, _i, o, k=k: self.out.__setitem__(k, o.detach()))
                        for k, m in modules.items()]


MAE_MODELS = ["mae_imagenet", "mae_adapt_artbench", "mae_adapt_wikiart", "mae_scratch_artbench", "mae_scratch_wikiart"]
MAE_TITLES = {"mae_imagenet": "MAE ViT-B/16 (ImageNet, frozen)",
              "mae_adapt_artbench": "MAE ViT-B/16 (ImageNet, adapted on ArtBench)",
              "mae_adapt_wikiart": "MAE ViT-B/16 (ImageNet, adapted on WikiArt)",
              "mae_scratch_artbench": "MAE ViT-B/16 (from scratch on ArtBench)",
              "mae_scratch_wikiart": "MAE ViT-B/16 (from scratch on WikiArt)"}


class MAEEncoder(nn.Module):
    """Unmasked MAE encoder (what the embeddings use): (B,3,224,224) -> (B,197,768) tokens, CLS first."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae
        self.blocks, self.num_heads = mae.blocks, mae.cfg.num_heads

    def forward(self, x):
        return self.mae.forward_encoder(x, 0.0)[0]


def latest_mae_ckpt(name: str) -> Path | None:
    """runs/<stamp>_mae_<dataset>_<init>/mae_final.pt of the newest run for a model key like mae_adapt_artbench."""
    _, init, ds = name.split("_")
    init = "imagenet" if init == "adapt" else init
    runs = sorted((REPO_ROOT / "runs").glob(f"*_mae_{ds}_{init}/mae_final.pt"))
    return runs[-1] if runs else None


def build_mae_models(device, names: list[str]) -> dict:
    from src.models.mae import MAEConfig, MaskedAutoencoderViT, load_imagenet_mae
    models = {}
    for name in names:
        if name == "mae_imagenet":
            mae = load_imagenet_mae()
        else:
            ck_path = latest_mae_ckpt(name)
            if ck_path is None:
                print(f"  {name}: no finished run found, skipped")
                continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            mae = MaskedAutoencoderViT(MAEConfig.from_dict(ck["model_cfg"]))
            mae.load_state_dict(ck["model"])
        net = MAEEncoder(mae.to(device).eval()).eval()
        blocks = (2, 4, 6, 8, 10, 12)
        # same input as the MAE embeddings (Resize 256 + CenterCrop 224, ImageNet statistics) -> handled like DINOv2
        models[name] = dict(net=net, kind="vit", vit="dino", norm="imagenet", crop=224, frame=[16 / 256, 16 / 256, 240 / 256, 240 / 256],
                            title=MAE_TITLES[name] + " (patch tokens, 14x14 grid)",
                            layers={f"block{b}": mae.blocks[b - 1] for b in blocks},
                            attn={f"block{b}": mae.blocks[b - 1].attn.qkv for b in range(1, 13)}, heads=mae.blocks[0].attn.num_heads)
    return models


def build_models(device, ae_ckpt: Path, cnn_ckpt: Path, wanted: list[str] | None = None):
    import timm
    models = {}
    ck = torch.load(cnn_ckpt, map_location="cpu", weights_only=False)
    cnn = MultiHeadCNN(ck["arch"], ck["head_sizes"])
    cnn.backbone.load_state_dict(ck["backbone"]); cnn.heads.load_state_dict(ck["heads"])
    imnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
    resnet_layers = lambda b: {"stem": b.act1, "layer1": b.layer1, "layer2": b.layer2, "layer3": b.layer3, "layer4": b.layer4}
    models["cnn_supervised"] = dict(net=cnn.backbone.to(device).eval(), kind="resnet", norm="imagenet",
                                    title="ResNet-50 (trained here: style / genre / artist)", layers=resnet_layers(cnn.backbone))
    models["resnet50_imagenet"] = dict(net=imnet.to(device).eval(), kind="resnet", norm="imagenet",
                                       title="ResNet-50 (ImageNet)", layers=resnet_layers(imnet))
    ck = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    ae = Autoencoder(AutoencoderConfig.from_dict(ck["model_cfg"])); ae.load_state_dict(ck["model"])
    ae = ae.to(device).eval()
    downs = [m for m in ae.encoder if isinstance(m, nn.Conv2d) and m.stride == (2, 2)]
    layers = {f"enc{i + 1}": m for i, m in enumerate(downs)}
    layers["latent"] = ae.encoder
    models["autoencoder"] = dict(net=ae, kind="ae", norm="half", title="Convolutional autoencoder", layers=layers)

    # vision transformers: block outputs (tokens) and the qkv projections that give the CLS attention
    from src.models.ssl_extractor import SSL_MODELS, _DINOV2_REF, _local_dinov2_clone
    cfg = SSL_MODELS["dinov2_vitb14"]
    dino = torch.hub.load(str(_local_dinov2_clone(_DINOV2_REF)), cfg["fn"], source="local", pretrained=True, verbose=False)
    dino = dino.to(device).eval()
    blocks = (2, 4, 6, 8, 10, 12)
    models["dinov2"] = dict(net=dino, kind="vit", vit="dino", norm="imagenet", crop=224, frame=[16 / 256, 16 / 256, 240 / 256, 240 / 256],
                            title="DINOv2 ViT-B/14 (patch tokens, 16x16 grid)",
                            layers={f"block{b}": dino.blocks[b - 1] for b in blocks},
                            attn={f"block{b}": dino.blocks[b - 1].attn.qkv for b in range(1, 13)}, heads=dino.blocks[0].attn.num_heads)
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    clip = clip.eval()
    res = clip.visual.transformer.resblocks
    models["clip"] = dict(net=clip.visual, kind="vit", vit="clip", norm="clip", crop=224, frame=[0, 0, 1, 1],
                          title="CLIP ViT-B/32 (patch tokens, 7x7 grid)",
                          layers={f"block{b}": res[b - 1] for b in blocks}, attn={f"block{b}": res[b - 1].attn for b in range(1, 13)},
                          heads=res[0].attn.num_heads)
    models.update(build_mae_models(device, [m for m in MAE_MODELS if wanted is None or m in wanted]))
    return models


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class AttnTaps:
    """Attention of the chosen attention modules, for DINOv2 (hook on the qkv linear) and CLIP (pre-hook on
    nn.MultiheadAttention, which does not return its weights).  Per module it keeps the CLS -> patch attention of every
    head ``heads[k]`` (B, heads, N-1), its head average ``out[k]`` and the full head-averaged matrix ``full[k]`` (B, N, N)."""

    def __init__(self, mods: dict[str, nn.Module], vit: str, heads: int):
        self.out: dict[str, torch.Tensor] = {}
        self.heads_out: dict[str, torch.Tensor] = {}
        self.full: dict[str, torch.Tensor] = {}
        self.vit, self.heads = vit, heads
        self.handles = []
        for k, m in mods.items():
            if vit == "dino":
                self.handles.append(m.register_forward_hook(lambda _m, _i, o, k=k: self._dino(k, o)))
            else:
                self.handles.append(m.register_forward_pre_hook(lambda mod, args, k=k: self._clip(k, mod, args)))

    def _store(self, k: str, q: torch.Tensor, kk: torch.Tensor) -> None:                 # (B, heads, N, hd) each
        att = ((q @ kk.transpose(-1, -2)) * (q.shape[-1] ** -0.5)).softmax(-1)             # (B, heads, N, N)
        self.heads_out[k] = att[:, :, 0, 1:].detach()
        self.out[k] = self.heads_out[k].mean(1)
        self.full[k] = att.mean(1).detach()

    def _dino(self, k, qkv):
        B, N, _ = qkv.shape
        q, kk, _v = qkv.reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        self._store(k, q, kk)

    def _clip(self, k, mod, args):
        x = args[0]
        if x.shape[0] != self.batch:                       # sequence-first layout (N, B, D) -> (B, N, D)
            x = x.transpose(0, 1)
        qkv = F.linear(x, mod.in_proj_weight, mod.in_proj_bias)
        B, N, _ = qkv.shape
        q, kk, _v = qkv.reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        self._store(k, q, kk)

    batch = 0

    def rollout(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """Attention rollout: CLS -> patch attention flow accumulated from the first block up to each block in ``keys``
        (``keys`` must be in order; all blocks up to the last key must have been recorded)."""
        any_k = next(iter(self.full.values()))
        B, N = any_k.shape[:2]
        eye = torch.eye(N, device=any_k.device)[None]
        R, out = eye.expand(B, -1, -1), {}
        last = max(int(k[5:]) for k in keys)
        for b in range(1, last + 1):
            A = 0.5 * self.full[f"block{b}"] + 0.5 * eye
            A = A / A.sum(-1, keepdim=True)
            R = A @ R
            if f"block{b}" in keys:
                out[f"block{b}"] = R[:, 0, 1:]
        return out


LAYER_TITLES = {"stem": "stem (conv1, stride 2)", "layer1": "layer1 (stride 4)", "layer2": "layer2 (stride 8)",
                "layer3": "layer3 (stride 16)", "layer4": "layer4 (stride 32)", "latent": "latent code (stride 32)"}


def save_gray(arr: np.ndarray, path: Path, **kw) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, **kw)


def percentile_norm(x: torch.Tensor, lo: float = 1, hi: float = 99) -> torch.Tensor:
    a, b = torch.quantile(x.flatten().float(), torch.tensor([lo / 100, hi / 100], device=x.device))
    return ((x - a) / (b - a + 1e-8)).clamp(0, 1)


def layer_products(A: torch.Tensor, out_dir: Path, tag: str) -> dict:
    """A: (C, H, W) activations of one layer for one image -> sprite / pca / mean files + statistics."""
    C, H, W = A.shape
    A = A.float()
    var = A.flatten(1).var(1)
    k = min(GRID * GRID, C)
    top = torch.argsort(var, descending=True)[:k]
    tiles = []
    for c in top:
        t = percentile_norm(A[c])[None, None]
        t = F.interpolate(t, size=(TILE, TILE), mode="nearest" if H <= 16 else "bilinear", **({} if H <= 16 else {"align_corners": False}))
        tiles.append(t[0, 0])
    while len(tiles) < GRID * GRID:
        tiles.append(torch.zeros(TILE, TILE, device=A.device))
    sprite = torch.stack(tiles).view(GRID, GRID, TILE, TILE).permute(0, 2, 1, 3).reshape(GRID * TILE, GRID * TILE)
    save_gray((sprite * 255).byte().cpu().numpy(), out_dir / f"{tag}__sprite.jpg", quality=68)

    # first three principal components of the per-pixel activation vectors -> RGB
    X = A.flatten(1).T                                     # (HW, C)
    X = X - X.mean(0, keepdim=True)
    PT = 96                                                # tile size of the principal-component sprite (4 x 4)
    pcs = torch.zeros(4 * PT, 4 * PT, device=A.device)
    varexp = []
    try:
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        P = X @ Vh[:3].T                                    # (HW, 3)
        n_pc = min(16, Vh.shape[0])
        P16 = (X @ Vh[:n_pc].T).T.reshape(n_pc, 1, H, W)    # the strongest components as maps
        varexp = [round(float(v), 4) for v in (S[:n_pc] ** 2 / (S ** 2).sum())]
        tl = []
        for c in range(16):
            if c < n_pc:
                t = percentile_norm(P16[c, 0])[None, None]
                t = F.interpolate(t, size=(PT, PT), mode="nearest" if H <= 16 else "bilinear", **({} if H <= 16 else {"align_corners": False}))[0, 0]
            else:
                t = torch.zeros(PT, PT, device=A.device)
            tl.append(t)
        pcs = torch.stack(tl).view(4, 4, PT, PT).permute(0, 2, 1, 3).reshape(4 * PT, 4 * PT)
    except Exception:
        P = torch.zeros(H * W, 3, device=A.device)
    save_gray((pcs * 255).byte().cpu().numpy(), out_dir / f"{tag}__pcs.jpg", quality=72)
    rgb = torch.stack([percentile_norm(P[:, j], 2, 98) for j in range(3)], 1).view(H, W, 3).permute(2, 0, 1)[None]
    rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear" if H > 16 else "nearest")[0]
    Image.fromarray((rgb.permute(1, 2, 0) * 255).byte().cpu().numpy()).save(out_dir / f"{tag}__pca.jpg", quality=90)

    mean = percentile_norm(A.abs().mean(0), 1, 99)[None, None]
    mean = F.interpolate(mean, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    save_gray((mean * 255).byte().cpu().numpy(), out_dir / f"{tag}__mean.png")
    return {"shape": [C, H, W], "sparsity": float((A.abs() < 1e-6).float().mean()), "mean": float(A.mean()),
            "std": float(A.std()), "pc_var": varexp, "channels": [int(c) for c in top.tolist()],
            "chmean": [round(float(A[c].mean()), 4) for c in top]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-style-artbench", type=int, default=10)
    ap.add_argument("--per-style-wikiart", type=int, default=4)
    ap.add_argument("--out-dir", default="visualizations/explorer/featuremaps")
    ap.add_argument("--ae-ckpt", default="runs/20261015-124529_ae_union/autoencoder_final.pt")
    ap.add_argument("--cnn-ckpt", default="runs/20261015-175907_cnn_resnet50/cnn_final.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=["cnn_supervised", "resnet50_imagenet", "autoencoder", "dinov2", "clip"],
                    choices=["cnn_supervised", "resnet50_imagenet", "autoencoder", "dinov2", "clip"] + MAE_MODELS,
                    help="models to (re)compute; the others keep their existing data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = REPO_ROOT / args.out_dir
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    images = pick_images({"artbench": args.per_style_artbench, "wikiart": args.per_style_wikiart}, args.seed)
    print(f"{len(images)} images, device {device}, models {args.models}")
    models = {m: v for m, v in build_models(device, REPO_ROOT / args.ae_ckpt, REPO_ROOT / args.cnn_ckpt, args.models).items()
              if m in args.models}
    taps = {m: Taps(v["layers"]) for m, v in models.items()}
    attn_taps = {m: AttnTaps(v["attn"], v["vit"], v["heads"]) for m, v in models.items() if v["kind"] == "vit"}

    def layer_title(k: str) -> str:
        if k.startswith("enc"):
            return f"encoder stage {k[3:]} (stride {2 ** int(k[3:])})"
        if k.startswith("block"):
            return f"block {k[5:]} of 12"
        return LAYER_TITLES.get(k, k)

    meta_models = {m: {"title": v["title"], "kind": v["kind"], "frame": v.get("frame", [0, 0, 1, 1]),
                       "layers": [{"key": k, "title": layer_title(k)} for k in v["layers"]]} for m, v in models.items()}

    for s in range(0, len(images), args.batch_size):
        batch = images[s:s + args.batch_size]
        ims = [load_image(str(REPO_ROOT / b["path"])) for b in batch]
        x01 = torch.stack([torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 for im in ims]).to(device)
        B = len(batch)
        rolls: dict[str, dict] = {}
        for b, im in zip(batch, ims):
            (data / b["id"]).mkdir(exist_ok=True)
            im.save(data / b["id"] / "input.jpg", quality=90)
            sf = data / b["id"] / "stats.json"
            b["stats"] = json.loads(sf.read_text()) if sf.exists() else {}          # keep other models' statistics
        with torch.no_grad():
            for m, v in models.items():
                if v["kind"] == "vit" and v["vit"] == "dino":       # same input as the embeddings: Resize(256) + CenterCrop(224)
                    x = ((x01 - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device))[:, :, 16:240, 16:240]
                elif v["kind"] == "vit":                             # CLIP: the whole square resized to 224 (its preprocessing)
                    x224 = F.interpolate(x01, size=224, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
                    x = (x224 - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
                elif v["norm"] == "imagenet":
                    x = (x01 - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
                else:
                    x = x01 * 2 - 1
                if v["kind"] == "vit":
                    for at in attn_taps.values():
                        at.batch = B
                recon = v["net"](x) if v["kind"] == "ae" else (v["net"](x), None)[1]
                if v["kind"] == "vit":                       # attention rollout up to every tapped block
                    rolls[m] = attn_taps[m].rollout(list(v["layers"]))
                for j, b in enumerate(batch):
                    d = data / b["id"]
                    if recon is not None:
                        r = ((recon[j].float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(r).save(d / "recon.jpg", quality=90)
                    for lk in v["layers"]:
                        T = taps[m].out[lk]
                        if v["kind"] == "vit":
                            if T.shape[0] != B:                           # CLIP blocks are sequence-first (N, B, D)
                                T = T.transpose(0, 1)
                            tok = T[j, 1:]                                # drop the CLS token
                            g = int(round(tok.shape[0] ** 0.5))
                            A = tok.T.reshape(-1, g, g)
                        else:
                            A = T[j]
                        st = layer_products(A, d, f"{m}__{lk}")
                        if v["kind"] == "vit":
                            def small(t):        # (g*g,) -> 64 x 64 map in [0, 1]
                                t = F.interpolate(percentile_norm(t.float().reshape(1, 1, g, g), 1, 99), size=(64, 64),
                                                  mode="bilinear", align_corners=False)
                                return t[0, 0]
                            save_gray((small(attn_taps[m].out[lk][j]) * 255).byte().cpu().numpy(), d / f"{m}__{lk}__attn.png")
                            save_gray((small(rolls[m][lk][j]) * 255).byte().cpu().numpy(), d / f"{m}__{lk}__roll.png")
                            hm = attn_taps[m].heads_out[lk][j]                           # (heads, g*g)
                            nh = hm.shape[0]
                            cols = 6
                            sheet = torch.zeros(((nh + cols - 1) // cols) * 64, cols * 64, device=hm.device)
                            for h in range(nh):
                                r_, c_ = divmod(h, cols)
                                sheet[r_ * 64:(r_ + 1) * 64, c_ * 64:(c_ + 1) * 64] = small(hm[h])
                            save_gray((sheet * 255).byte().cpu().numpy(), d / f"{m}__{lk}__heads.png")
                            st.update(has_attn=True, has_roll=True, heads=int(nh), head_cols=cols)
                        b["stats"][f"{m}/{lk}"] = st
        for b in batch:
            (data / b["id"] / "stats.json").write_text(json.dumps(b.pop("stats")))
        print(f"  {min(s + args.batch_size, len(images))} / {len(images)}", flush=True)

    idx_file = data / "index.json"
    old = json.loads(idx_file.read_text()) if idx_file.exists() else {"models": {}}
    order = ["cnn_supervised", "resnet50_imagenet", "autoencoder", *MAE_MODELS, "dinov2", "clip"]
    merged = {**old.get("models", {}), **meta_models}
    from src.metadata.nudity import load_flags
    fl = load_flags([b["path"] for b in images])        # nudity flags (None if scripts/15_flag_nudity.py was not run)
    index = {"images": [{**{k: b[k] for k in ("id", "dataset", "style", "file")}, **({"nsfw": int(fl[j])} if fl is not None else {})}
                        for j, b in enumerate(images)],
             "models": {m: merged[m] for m in order if m in merged}, "tile": TILE, "grid": GRID,
             "note": "images are from the held-out test split"}
    idx_file.write_text(json.dumps(index))
    from src.visualization.featuremaps import write_viewer
    print("viewer:", write_viewer(out / "index.html"))


if __name__ == "__main__":
    main()
