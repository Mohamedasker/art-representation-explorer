#!/usr/bin/env python3
"""
Train a StyleGAN2 (pure PyTorch, ``src/models/gan.py``, adaptive discriminator augmentation) on the TRAIN splits of
ArtBench-10 and WikiArt jointly (~130k paintings, unconditional, 128 px), with 4-GPU data parallelism (gradients
all-reduced by hand, so the lazy R1 / path-length regularisers, which need double back-prop, stay simple) and wandb.

Per iteration (as in StyleGAN2-ADA): G step (non-saturating loss, style mixing), every 4th: path-length regulariser;
D step on augmented real / fake images, every 16th: lazy R1; ADA adjusts the augmentation probability ``p`` so that
the sign of the real logits stays near ``--ada-target``.  A G_ema (EMA of G) produces the sample grids and the FID-like
metric (torchvision Inception-v3 pool features, 5000 generated vs 5000 held-out real images; NOT comparable to
published TF-Inception FIDs).

Every run creates ``runs/<YYYYmmdd-HHMMSS>_<name>/`` with config.json, checkpoints/last.pt (+ numbered snapshots),
gan_final.pt (G_ema + D, what the embedding extractor loads), samples/ (grids of G_ema with fixed latents), train.log,
metrics.jsonl, status.json, wandb/.  Resumable (``--resume <run dir>``) and time-limit aware (``--time-limit-min``).

    torchrun --standalone --nproc_per_node=4 scripts/20_train_gan.py --name gan_union --kimg 20000
    torchrun --standalone --nproc_per_node=4 scripts/20_train_gan.py --resume runs/<stamp>_gan_union

Embeddings:  python scripts/02_extract_features.py --model gan_disc --checkpoint runs/<run>/gan_final.pt --dataset artbench --split all
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.datasets.union import UnionImageDataset
import src.models.gan as gan_module
from src.models.gan import AugmentPipe, Discriminator, GanConfig, Generator
from src.training.common import init_wandb, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("gan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--name", type=str, default="gan_union")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--artbench-dir", type=str, default="data/artbench")
    p.add_argument("--wikiart-dir", type=str, default="data/wikiart")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--channel-base", type=int, default=16384)
    p.add_argument("--channel-max", type=int, default=512)
    p.add_argument("--map-layers", type=int, default=2)
    # optimisation
    p.add_argument("--kimg", type=float, default=None, help="Train until this many thousand real images were shown (default 20000; on --resume the saved value)")
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many iterations (smoke tests)")
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--r1-gamma", type=float, default=0.5)
    p.add_argument("--pl-weight", type=float, default=2.0)
    p.add_argument("--g-reg-interval", type=int, default=4)
    p.add_argument("--d-reg-interval", type=int, default=16)
    p.add_argument("--style-mixing", type=float, default=0.9)
    p.add_argument("--ema-kimg", type=float, default=None, help="EMA half-life; default global_batch * 10 / 32")
    p.add_argument("--ema-rampup", type=float, default=0.05)
    p.add_argument("--aug-p", type=float, default=None, help="Fixed augmentation probability (disables ADA)")
    p.add_argument("--ada-target", type=float, default=0.6)
    p.add_argument("--ada-kimg", type=float, default=100, help="Time for p to go from 0 to 1")
    p.add_argument("--ada-interval", type=int, default=4)
    p.add_argument("--amp", choices=["none", "bf16"], default=None,
                   help="bfloat16 autocast of the convolutions in the main G / D steps (regularisers stay fp32); default none. "
                        "May be changed when resuming")
    p.add_argument("--channels-last", action="store_true", help="NHWC memory format for G / D (same numerics)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile G.synthesis / D for the main G and D steps (fuses the element-wise ops; the "
                        "regularisers with double back-prop stay eager)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=12, help="DataLoader workers per GPU")
    p.add_argument("--draft-size", type=int, default=256, help="JPEG draft decode: shrink big images at decode time")
    # logging / checkpointing
    p.add_argument("--log-every", type=int, default=20, help="iterations")
    p.add_argument("--sample-kimg", type=float, default=100)
    p.add_argument("--fid-kimg", type=float, default=1000)
    p.add_argument("--fid-n", type=int, default=5000)
    p.add_argument("--ckpt-kimg", type=float, default=100)
    p.add_argument("--snapshot-kimg", type=float, default=2000, help="Numbered G_ema/D snapshots")
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def to_uint8(img) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


def make_transform(S: int):
    """Fixed preprocessing, identical in training, FID and the embedding extractor: resize shorter side, centre crop."""
    return transforms.Compose([transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                               transforms.CenterCrop(S), transforms.Lambda(to_uint8)])


def build_data(args, split: str) -> UnionImageDataset:
    return UnionImageDataset(args.artbench_dir, args.wikiart_dir, split, make_transform(args.img_size), args.draft_size)


def to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 127.5 - 1


def infinite_batches(loader: DataLoader, sampler: DistributedSampler, start_epoch: int):
    epoch = start_epoch
    while True:
        sampler.set_epoch(epoch)
        for x, _ in loader:
            yield x
        epoch += 1


# ── FID-like metric ───────────────────────────────────────────────────────────

class InceptionFeatures:
    """torchvision Inception-v3 (ImageNet weights) 2048-D pool features of uint8 images."""

    def __init__(self, device) -> None:
        import torchvision
        net = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, u8: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(u8.to(self.device).float() / 255, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std).float()


def fid_from_feats(f1: np.ndarray, f2: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


# ── training helpers ──────────────────────────────────────────────────────────

def requires_grad(m: torch.nn.Module, flag: bool) -> None:
    for p in m.parameters():
        p.requires_grad_(flag)


def sync_grads(params: list[torch.Tensor], world: int) -> None:
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    flat = torch.nan_to_num(flat, nan=0.0, posinf=1e5, neginf=-1e5)
    if world > 1:
        dist.all_reduce(flat)
        flat /= world
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def make_opt(net: torch.nn.Module, lr: float, reg_interval: int):
    c = reg_interval / (reg_interval + 1)
    return torch.optim.Adam(net.parameters(), lr=lr * c, betas=(0.0, 0.99 ** c), eps=1e-8)


def grid_image(imgs: torch.Tensor, nrow: int) -> torch.Tensor:
    return make_grid(imgs.float().clamp(-1, 1).add(1).div(2).cpu(), nrow=nrow, padding=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        saved = json.loads((resume_dir / "config.json").read_text())["args"]
        keep = {k: getattr(args, k) for k in ("resume", "wandb", "time_limit_min", "num_workers", "max_steps", "kimg", "amp", "compile", "channels_last")}
        for k, v in saved.items():
            setattr(args, k, v)
        for k, v in keep.items():
            if k in ("max_steps", "kimg", "amp") and v is None or k in ("compile", "channels_last") and not v:      # not given on the command line: keep the saved value
                continue
            setattr(args, k, v)
        run_dir = resume_dir
    else:
        args.kimg = 20000 if args.kimg is None else args.kimg
        args.amp = args.amp or "none"
        t = torch.tensor([int(time.time())], dtype=torch.int64, device=device)
        if world > 1:
            dist.broadcast(t, src=0)
        stamp = dt.datetime.fromtimestamp(int(t.item())).strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / args.runs_dir / f"{stamp}_{args.name}"
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    setup_logging(run_dir, rank)
    if rank == 0:
        for d in ("samples", "checkpoints"):
            (run_dir / d).mkdir(exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    gan_module.CHANNELS_LAST = args.channels_last

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds = build_data(args, "train")
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    global_batch = args.batch_size * world
    total_nimg = int(args.kimg * 1000)

    # ── model / optim ────────────────────────────────────────────────────────
    cfg = GanConfig(img_size=args.img_size, channel_base=args.channel_base, channel_max=args.channel_max,
                    map_layers=args.map_layers)
    torch.manual_seed(args.seed)
    G, D = Generator(cfg).to(device), Discriminator(cfg).to(device)
    if args.channels_last:
        G, D = G.to(memory_format=torch.channels_last), D.to(memory_format=torch.channels_last)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    for t in list(G.state_dict().values()) + list(D.state_dict().values()):     # identical init on every rank
        if world > 1:
            dist.broadcast(t, src=0)
    G_ema.load_state_dict(G.state_dict())
    optG, optD = make_opt(G, args.lr, args.g_reg_interval), make_opt(D, args.lr, args.d_reg_interval)
    aug = AugmentPipe(p=args.aug_p or 0.0)
    gp, dp = list(G.parameters()), list(D.parameters())
    L = cfg.num_ws

    it, cur_nimg, pl_mean, ada_p, best_fid = 0, 0, torch.zeros([], device=device), float(args.aug_p or 0.0), None
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        it, cur_nimg, ada_p, best_fid = ck["it"], ck["nimg"], ck["ada_p"], ck.get("best_fid")
        pl_mean.fill_(ck["pl_mean"])
        log.info("Resumed from %s at iteration %d (%.0f kimg)", ckpt_path, it, cur_nimg / 1000)
    aug.p = ada_p

    ema_kimg = args.ema_kimg if args.ema_kimg is not None else global_batch * 10 / 32
    n_params = {"G": sum(p.numel() for p in G.parameters()), "D": sum(p.numel() for p in D.parameters())}
    info = {"train_datasets": ["artbench", "wikiart"], "split": "train", "img_size": args.img_size, "unconditional": True,
            "preprocess": "resize shorter side (bicubic), centre crop, uint8 -> [-1, 1]",
            "embedding": "discriminator fc(512) activation, eval mode"}
    config = {"args": vars(args), "model": cfg.to_dict(), "info": info, "n_params": n_params,
              "train_images": len(train_ds), "world_size": world, "global_batch_size": global_batch,
              "ema_kimg": ema_kimg, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "torch": torch.__version__}
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"it": it, "kimg": cur_nimg / 1000, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": config["slurm_job_id"], "host": config["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("G %.1fM D %.1fM params, train=%d, world=%d, global_bs=%d, target %.0f kimg, ema %.1f kimg",
                 n_params["G"] / 1e6, n_params["D"] / 1e6, len(train_ds), world, global_batch, total_nimg / 1000, ema_kimg)
        run = init_wandb(args, config, run_dir)
        if run is not None:
            run.summary["run_dir"] = str(run_dir)
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(d: dict) -> None:
        if rank != 0:
            return
        metrics_f.write(json.dumps({"it": it, "kimg": cur_nimg / 1000, **d}) + "\n")
        metrics_f.flush()
        if run is not None:
            run.log({"kimg": cur_nimg / 1000, **d}, step=it)

    # ── FID machinery (real statistics cached in the run dir) ────────────────
    incep, real_feats = None, None

    def fid_setup() -> None:
        nonlocal incep, real_feats
        incep = InceptionFeatures(device)
        f_path = run_dir / f"fid_real_{args.fid_n}_{args.img_size}.npy"
        if rank == 0 and not f_path.exists():
            test = build_data(args, "test")
            idx = np.random.RandomState(1234).permutation(len(test))[:args.fid_n]
            sub = torch.utils.data.Subset(test, idx.tolist())
            fl = DataLoader(sub, batch_size=100, num_workers=min(args.num_workers * 2, 24))
            feats = torch.cat([incep(x).cpu() for x, _ in fl]).numpy()
            np.save(f_path, feats)
        if world > 1:
            dist.barrier()
        real_feats = np.load(f_path)

    @torch.no_grad()
    def compute_fid() -> float:
        if incep is None:
            fid_setup()
        n_local = args.fid_n // world
        g = torch.Generator(device=device).manual_seed(999 + rank)
        feats = []
        for i in range(0, n_local, 100):
            z = torch.randn(min(100, n_local - i), cfg.z_dim, device=device, generator=g)
            img = G_ema(z).clamp(-1, 1)
            feats.append(incep(((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)))
        feats = torch.cat(feats)
        if world > 1:
            gathered = [torch.zeros_like(feats) for _ in range(world)]
            dist.all_gather(gathered, feats)
            feats = torch.cat(gathered)
        val = fid_from_feats(feats.cpu().numpy(), real_feats) if rank == 0 else 0.0
        if world > 1:
            t = torch.tensor([val], device=device)
            dist.broadcast(t, src=0)
            val = t.item()
        return val

    # ── fixed latents for sample grids ───────────────────────────────────────
    grid_z = torch.randn(64, cfg.z_dim, generator=torch.Generator().manual_seed(2024)).to(device)

    @torch.no_grad()
    def log_samples(first_reals: torch.Tensor | None = None) -> None:
        if rank != 0:
            return
        out = {}
        for tr in (1.0, 0.7):
            imgs = G_ema(grid_z, truncation=tr, noise=False)
            out[tr] = grid_image(imgs, 8)
            save_image(out[tr], run_dir / "samples" / f"kimg_{int(cur_nimg / 1000):06d}_trunc{tr}.png")
        if first_reals is not None:
            save_image(grid_image(to_float(first_reals[:64]), 8), run_dir / "samples" / "real_reference.png")
        if run is not None:
            import wandb
            run.log({"kimg": cur_nimg / 1000,
                     "samples/G_ema": wandb.Image(out[1.0], caption=f"G_ema, {cur_nimg / 1000:.0f} kimg, fixed z"),
                     "samples/G_ema_trunc0.7": wandb.Image(out[0.7], caption="truncation 0.7")}, step=it)

    def weights() -> dict:
        return {"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(), "cfg": cfg.to_dict(),
                "info": info, "it": it, "nimg": cur_nimg}

    def checkpoint(numbered: bool = False) -> None:
        if rank != 0:
            return
        save_ckpt_atomic({**weights(), "optG": optG.state_dict(), "optD": optD.state_dict(), "ada_p": ada_p,
                          "pl_mean": float(pl_mean), "best_fid": best_fid, "args": vars(args)}, ckpt_path)
        if numbered:
            save_ckpt_atomic({"G_ema": G_ema.state_dict(), "D": D.state_dict(), "cfg": cfg.to_dict(), "info": info,
                              "it": it, "nimg": cur_nimg}, run_dir / "checkpoints" / f"snap_{int(cur_nimg / 1000):06d}.pt")

    # ── training loop ────────────────────────────────────────────────────────
    batches = infinite_batches(loader, sampler, start_epoch=cur_nimg // max(1, len(sampler) * world))
    stats = {k: torch.zeros([], device=device) for k in ("loss_G", "loss_D", "D_real", "D_fake", "r1", "pl", "ada_rt")}
    cnt = {"main": 0, "r1": 0, "pl": 0}
    ada_acc, ada_n = torch.zeros([], device=device), 0
    torch.manual_seed(args.seed * 1000 + rank + it)
    next_sample = (cur_nimg // int(args.sample_kimg * 1000) + 1) * int(args.sample_kimg * 1000)
    next_fid = (cur_nimg // int(args.fid_kimg * 1000) + 1) * int(args.fid_kimg * 1000)
    next_ckpt = (cur_nimg // int(args.ckpt_kimg * 1000) + 1) * int(args.ckpt_kimg * 1000)
    next_snap = (cur_nimg // int(args.snapshot_kimg * 1000) + 1) * int(args.snapshot_kimg * 1000)
    t_last, nimg_last = time.time(), cur_nimg
    stop_reason = "finished"
    G.train(); D.train()

    Gsyn, Dfwd = (torch.compile(G.synthesis), torch.compile(D)) if args.compile else (G.synthesis, D)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")

    def gen_ws(b: int, mixing: bool = True) -> torch.Tensor:
        z = torch.randn(b, cfg.z_dim, device=device)
        w = G.mapping(z)
        ws = w[:, None].repeat(1, L, 1)
        if mixing and args.style_mixing > 0:
            w2 = G.mapping(torch.randn(b, cfg.z_dim, device=device))[:, None].repeat(1, L, 1)
            cut = torch.randint(1, L, [], device=device)
            cut = torch.where(torch.rand([], device=device) < args.style_mixing, cut, torch.full_like(cut, L))
            ws = torch.where(torch.arange(L, device=device)[None, :, None] < cut, ws, w2)
        return ws

    if cur_nimg == 0 and rank == 0:
        first_reals = torch.stack([train_ds[i][0] for i in np.random.RandomState(0).permutation(len(train_ds))[:64]])
        log_samples(first_reals)
    done = cur_nimg >= total_nimg or (args.max_steps is not None and it >= args.max_steps)
    while not done:
        real = to_float(next(batches).to(device, non_blocking=True))
        real = torch.where(torch.rand(real.shape[0], 1, 1, 1, device=device) < 0.5, real, real.flip(3))   # x-flip
        b = real.shape[0]

        # -- G main
        requires_grad(G, True); requires_grad(D, False)
        ws = gen_ws(b)
        with torch.no_grad():
            w_batch = ws[:, 0].detach()
            G.mapping.w_avg.copy_(w_batch.mean(0).lerp(G.mapping.w_avg, 0.995))
        with autocast():
            loss_G = F.softplus(-Dfwd(aug(Gsyn(ws))).float()).mean()
        optG.zero_grad(set_to_none=True)
        loss_G.backward()
        sync_grads(gp, world); optG.step()
        stats["loss_G"] += loss_G.detach(); cnt["main"] += 1

        # -- G path-length regulariser (lazy)
        if it % args.g_reg_interval == 0 and args.pl_weight > 0:
            ws = gen_ws(max(2, b // 2), mixing=False)
            img = G.synthesis(ws)
            pl_noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])
            pl_grads = torch.autograd.grad((img * pl_noise).sum(), ws, create_graph=True)[0]
            pl_len = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean.copy_(pl_mean.lerp(pl_len.mean().detach(), 0.01))
            pen = (pl_len - pl_mean).square().mean()
            optG.zero_grad(set_to_none=True)
            (pen * args.pl_weight * args.g_reg_interval).backward()
            sync_grads(gp, world); optG.step()
            stats["pl"] += pl_len.mean().detach(); cnt["pl"] += 1

        # -- D main
        requires_grad(G, False); requires_grad(D, True)
        with torch.no_grad(), autocast():
            fake = Gsyn(gen_ws(b))
        with autocast():
            d_fake, d_real = Dfwd(aug(fake)).float(), Dfwd(aug(real)).float()
        loss_D = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
        optD.zero_grad(set_to_none=True)
        loss_D.backward()
        sync_grads(dp, world); optD.step()
        stats["loss_D"] += loss_D.detach(); stats["D_real"] += d_real.detach().mean(); stats["D_fake"] += d_fake.detach().mean()
        ada_acc += d_real.detach().sign().mean(); ada_n += 1

        # -- D R1 (lazy)
        if it % args.d_reg_interval == 0 and args.r1_gamma > 0:
            real_aug = aug(real).detach().requires_grad_(True)
            gr = torch.autograd.grad(D(real_aug).sum(), real_aug, create_graph=True)[0]
            r1 = gr.square().sum([1, 2, 3]).mean()
            optD.zero_grad(set_to_none=True)
            (r1 * (args.r1_gamma / 2) * args.d_reg_interval).backward()
            sync_grads(dp, world); optD.step()
            stats["r1"] += r1.detach(); cnt["r1"] += 1

        # -- G_ema
        ema_nimg = min(ema_kimg * 1000, cur_nimg * args.ema_rampup)
        beta = 0.5 ** (global_batch / max(ema_nimg, 1e-8))
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.copy_(p.lerp(pe, beta))
            for be, bu in zip(G_ema.buffers(), G.buffers()):
                be.copy_(bu)

        it += 1
        cur_nimg += global_batch

        # -- ADA
        if args.aug_p is None and it % args.ada_interval == 0:
            rt = reduce_mean(ada_acc / max(ada_n, 1), world).item()
            ada_p = float(np.clip(ada_p + np.sign(rt - args.ada_target) * global_batch * args.ada_interval
                                  / (args.ada_kimg * 1000), 0.0, 1.0))
            aug.p = ada_p
            stats["ada_rt"].fill_(rt)
            ada_acc.zero_(); ada_n = 0

        if it % args.log_every == 0:
            red = {k: reduce_mean(v.clone(), world) for k, v in stats.items()}
            m = {"train/loss_G": red["loss_G"].item() / cnt["main"], "train/loss_D": red["loss_D"].item() / cnt["main"],
                 "train/D_real": red["D_real"].item() / cnt["main"], "train/D_fake": red["D_fake"].item() / cnt["main"],
                 "train/ada_p": ada_p, "train/ada_rt": red["ada_rt"].item(),
                 "train/pl_mean": float(pl_mean), "train/mbstd_ema": float(D.mbstd_ema)}
            if cnt["r1"]:
                m["train/r1"] = red["r1"].item() / cnt["r1"]
            if cnt["pl"]:
                m["train/pl_len"] = red["pl"].item() / cnt["pl"]
            now = time.time()
            m["perf/img_per_s"] = (cur_nimg - nimg_last) / (now - t_last)
            if not all(np.isfinite(v) for v in m.values()):
                raise RuntimeError(f"non-finite metric at iteration {it}: {m}")
            emit(m)
            log.info("it %d %.0fkimg G %.3f D %.3f real %.2f fake %.2f p %.3f rt %.2f r1 %s pl %s %.0f img/s", it,
                     cur_nimg / 1000, m["train/loss_G"], m["train/loss_D"], m["train/D_real"], m["train/D_fake"],
                     ada_p, m["train/ada_rt"], f"{m.get('train/r1', float('nan')):.3f}",
                     f"{m.get('train/pl_len', float('nan')):.2f}", m["perf/img_per_s"])
            for v in stats.values():
                v.zero_()
            cnt = {"main": 0, "r1": 0, "pl": 0}
            t_last, nimg_last = now, cur_nimg

        if cur_nimg >= next_sample or (args.max_steps is not None and it == args.max_steps):
            log_samples(); next_sample += int(args.sample_kimg * 1000)
        if cur_nimg >= next_fid or (args.max_steps is not None and it == args.max_steps):
            fid = compute_fid()
            emit({"val/fid": fid})
            log.info("it %d %.0f kimg  FID-like (torchvision Inception, %d vs %d) = %.2f", it, cur_nimg / 1000,
                     args.fid_n, args.fid_n, fid)
            if best_fid is None or fid < best_fid:
                best_fid = fid
                if rank == 0:
                    save_ckpt_atomic({"G_ema": G_ema.state_dict(), "D": D.state_dict(), "cfg": cfg.to_dict(),
                                      "info": info, "it": it, "nimg": cur_nimg, "fid": fid},
                                     run_dir / "checkpoints" / "best_fid.pt")
            next_fid += int(args.fid_kimg * 1000)
        if cur_nimg >= next_ckpt:
            checkpoint(numbered=cur_nimg >= next_snap)
            next_ckpt += int(args.ckpt_kimg * 1000)
            if cur_nimg >= next_snap:
                next_snap += int(args.snapshot_kimg * 1000)

        if cur_nimg >= total_nimg or (args.max_steps is not None and it >= args.max_steps):
            done = True
        if args.time_limit_min is not None and it % 20 == 0 and not done:
            flag = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit_min), device=device)
            if world > 1:
                dist.broadcast(flag, src=0)
            if flag.item():
                stop_reason, done = "time_limit", True

    # ── wrap up ──────────────────────────────────────────────────────────────
    if stop_reason == "finished":
        if rank == 0:
            torch.save({"G_ema": G_ema.state_dict(), "D": D.state_dict(), "cfg": cfg.to_dict(), "info": info,
                        "it": it, "nimg": cur_nimg, "best_fid": best_fid, "ada_p": ada_p}, run_dir / "gan_final.pt")
            log.info("final weights: %s", run_dir / "gan_final.pt")
    else:
        log.info("time limit reached at iteration %d (%.0f kimg) -> checkpointing; resume with --resume %s",
                 it, cur_nimg / 1000, run_dir)
    checkpoint(numbered=False)
    if rank == 0:
        (run_dir / "status.json").write_text(json.dumps(
            {"status": stop_reason, "it": it, "kimg": cur_nimg / 1000, "target_kimg": total_nimg / 1000,
             "best_fid": best_fid, "finished_at": dt.datetime.now().isoformat()}, indent=2))
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
