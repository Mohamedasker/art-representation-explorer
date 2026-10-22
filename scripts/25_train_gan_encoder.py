#!/usr/bin/env python3
"""
Stretch goal of the GAN phase: train an INVERSION ENCODER for the StyleGAN2 of ``scripts/20_train_gan.py``.

The (frozen) generator G_ema is kept fixed; an encoder E (ResNet-34 trunk, ImageNet init, ``GanEncoder``) maps an image
to one latent w (512-D, the generator's intermediate latent space, broadcast to every layer).  Loss on real training
paintings (ArtBench + WikiArt train splits):  pixel L2 + VGG-16 perceptual distance between x and G(E(x)).  Half of
every batch is synthetic, G(w) for w ~ mapping(z): there E is additionally supervised with the true latent
(``--w-weight``), which keeps its output inside the region of W that the generator actually uses.  Embedding = E(x).

Reconstruction grids (input / G(E(input)) / mean-w reconstruction is NOT shown) go to ``<run>/recon`` and wandb; the held-out
reconstruction errors (test splits) are logged as val/*.  Output: ``runs/<stamp>_gan_enc/gan_enc_final.pt``.

    torchrun --standalone --nproc_per_node=4 scripts/25_train_gan_encoder.py --gan runs/<gan run>/gan_final.pt
"""

from __future__ import annotations

import argparse
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
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

from src.models.gan import GanConfig, GanEncoder, Generator
from src.training.common import init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("gan_enc")


def load_train_module():
    """Reuse the data helpers of the GAN training script (file name starts with a digit -> importlib)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gan", REPO_ROOT / "scripts" / "20_train_gan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VGGPerceptual(torch.nn.Module):
    """Unit-normalised VGG-16 feature differences at relu1_2 / 2_2 / 3_3 / 4_3 (LPIPS without the learned weights)."""

    def __init__(self) -> None:
        super().__init__()
        feats = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.slices = torch.nn.ModuleList([feats[:4], feats[4:9], feats[9:16], feats[16:23]])
        self.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b])
        x = ((x.clamp(-1, 1) + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for sl in self.slices:
            x = sl(x)
            fa, fb = x.chunk(2)
            fa, fb = fa / (fa.norm(dim=1, keepdim=True) + 1e-8), fb / (fb.norm(dim=1, keepdim=True) + 1e-8)
            loss = loss + (fa - fb).square().sum(1).mean()
        return loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gan", type=str, default=None, help="GAN checkpoint (gan_final.pt or checkpoints/last.pt)")
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue")
    p.add_argument("--name", type=str, default="gan_enc")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--artbench-dir", type=str, default="data/artbench")
    p.add_argument("--wikiart-dir", type=str, default="data/wikiart")
    p.add_argument("--draft-size", type=int, default=256)
    p.add_argument("--epochs", type=float, default=6.0, help="Passes over the real training images")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=24, help="Per-GPU: this many real + the same number synthetic")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--l2-weight", type=float, default=1.0)
    p.add_argument("--perc-weight", type=float, default=1.0)
    p.add_argument("--w-weight", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-n", type=int, default=2048)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    args = p.parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)
    tg = load_train_module()

    if args.resume:
        run_dir = Path(args.resume).resolve()
        saved = json.loads((run_dir / "config.json").read_text())["args"]
        keep = {k: getattr(args, k) for k in ("resume", "wandb", "time_limit_min", "num_workers")}
        for k, v in saved.items():
            setattr(args, k, v)
        for k, v in keep.items():
            setattr(args, k, v)
    else:
        if not args.gan:
            raise SystemExit("--gan <checkpoint> is required for a new run")
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
        for d in ("recon", "checkpoints"):
            (run_dir / d).mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── frozen generator ─────────────────────────────────────────────────────
    gan_ck = torch.load(args.gan, map_location="cpu", weights_only=False)
    cfg = GanConfig.from_dict(gan_ck["cfg"])
    G = Generator(cfg)
    G.load_state_dict(gan_ck["G_ema"])
    G = G.to(device).eval().requires_grad_(False)
    gan_run = Path(args.gan).resolve().parent
    gan_run = gan_run.parent if gan_run.name == "checkpoints" else gan_run
    L = cfg.num_ws

    def generate(w: torch.Tensor) -> torch.Tensor:
        return G.synthesis(w[:, None].repeat(1, L, 1), noise=False)

    # ── data ─────────────────────────────────────────────────────────────────
    class A:  # args shim for the GAN script's data helpers
        img_size, artbench_dir, wikiart_dir, draft_size = cfg.img_size, args.artbench_dir, args.wikiart_dir, args.draft_size
    train_ds = tg.build_data(A, "train")
    test_ds = tg.build_data(A, "test")
    val_idx = np.random.RandomState(1234).permutation(len(test_ds))[:args.val_n].tolist()
    val_ds = Subset(test_ds, val_idx)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=64, sampler=DistributedSampler(val_ds, world, rank, shuffle=False),
                            num_workers=min(args.num_workers, 8))
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = int(args.epochs * steps_per_epoch)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    # ── model ────────────────────────────────────────────────────────────────
    E = GanEncoder(cfg.w_dim, pretrained=True).to(device)
    E.w_avg.copy_(G.mapping.w_avg)
    perc = VGGPerceptual().to(device)
    ddp = DDP(E, device_ids=[local]) if world > 1 else E
    opt = torch.optim.AdamW(E.parameters(), lr=args.lr, weight_decay=1e-4)
    step = 0
    ck_path = run_dir / "checkpoints" / "last.pt"
    if args.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        E.load_state_dict(ck["E"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        log.info("Resumed at step %d", step)
    peak_lr = args.lr

    config = {"args": vars(args), "model": cfg.to_dict(), "gan_checkpoint": str(Path(args.gan).resolve()),
              "gan_run": gan_run.name, "gan_kimg": gan_ck.get("nimg", 0) / 1000, "train_images": len(train_ds),
              "world_size": world, "global_batch_size": 2 * args.batch_size * world, "steps_per_epoch": steps_per_epoch,
              "total_steps": total_steps, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        log.info("run dir %s; GAN %s (%.0f kimg); steps/epoch %d, total %d", run_dir, gan_run.name,
                 config["gan_kimg"], steps_per_epoch, total_steps)
        run = init_wandb(args, config, run_dir)
        if run is not None:
            run.summary["run_dir"] = str(run_dir)
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(d: dict) -> None:
        if rank != 0:
            return
        metrics_f.write(json.dumps({"step": step, **d}) + "\n")
        metrics_f.flush()
        if run is not None:
            run.log({"step": step, **d}, step=step)

    def weights() -> dict:
        return {"E": E.state_dict(), "w_dim": cfg.w_dim, "step": step, "gan_run": gan_run.name,
                "gan_checkpoint": str(Path(args.gan).resolve()), "img_size": cfg.img_size}

    @torch.no_grad()
    def validate(save_grid: bool = False) -> dict:
        E.eval()
        sums = torch.zeros(3, device=device)
        first = None
        for x, _ in val_loader:
            x = tg.to_float(x.to(device))
            rec = generate(E(x))
            sums += torch.stack([F.mse_loss(rec, x, reduction="sum") / (x[0].numel()),
                                 perc(rec, x) * x.shape[0], torch.tensor(float(x.shape[0]), device=device)])
            if first is None:
                first = (x[:12].cpu(), rec[:12].cpu())
        if world > 1:
            dist.all_reduce(sums)
        E.train()
        m = {"val/mse": (sums[0] / sums[2]).item(), "val/perceptual": (sums[1] / sums[2]).item()}
        if save_grid and rank == 0 and first is not None:
            grid = make_grid(torch.cat(first).add(1).div(2).clamp(0, 1), nrow=12, padding=2)
            save_image(grid, run_dir / "recon" / f"step_{step:06d}.png")
            if run is not None:
                import wandb
                run.log({"step": step, "samples/recon": wandb.Image(grid, caption=f"step {step}: top input, bottom G(E(input)) (held-out paintings)")}, step=step)
        return m

    # ── training ─────────────────────────────────────────────────────────────
    E.train()
    stop_reason, done = "finished", step >= total_steps
    acc = {k: torch.zeros([], device=device) for k in ("loss", "mse", "perc", "w")}
    n_acc, t_last, epoch = 0, time.time(), step // max(1, steps_per_epoch)
    if step == 0:
        emit(validate(save_grid=True))
    while not done:
        sampler.set_epoch(epoch)
        for x, _ in loader:
            x = tg.to_float(x.to(device, non_blocking=True))
            x = torch.where(torch.rand(x.shape[0], 1, 1, 1, device=device) < 0.5, x, x.flip(3))
            lr = lr_at(step, peak_lr, args, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.no_grad():                                     # synthetic half with known latents
                w_true = G.mapping(torch.randn(x.shape[0], cfg.z_dim, device=device))
                x_syn = generate(w_true)
            w_all = ddp(torch.cat([x, x_syn]))
            rec = generate(w_all[:x.shape[0]])
            mse = F.mse_loss(rec, x)
            pl = perc(rec, x)
            lw = F.mse_loss(w_all[x.shape[0]:], w_true)
            loss = args.l2_weight * mse + args.perc_weight * pl + args.w_weight * lw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            for k, v in zip(acc, (loss, mse, pl, lw)):
                acc[k] += v.detach()
            n_acc += 1
            if step % args.log_every == 0:
                red = {k: reduce_mean(v.clone(), world).item() / n_acc for k, v in acc.items()}
                now = time.time()
                emit({"train/loss": red["loss"], "train/mse": red["mse"], "train/perceptual": red["perc"],
                      "train/w_mse": red["w"], "lr": lr, "epoch": step / steps_per_epoch,
                      "perf/img_per_s": args.log_every * x.shape[0] * 2 * world / (now - t_last)})
                log.info("step %d/%d ep %.2f loss %.4f mse %.4f perc %.3f w %.3f lr %.1e", step, total_steps,
                         step / steps_per_epoch, red["loss"], red["mse"], red["perc"], red["w"], lr)
                for v in acc.values():
                    v.zero_()
                n_acc, t_last = 0, now
            if step % args.val_every == 0:
                m = validate(save_grid=True); emit(m)
                log.info("step %d  val mse %.4f perceptual %.3f", step, m["val/mse"], m["val/perceptual"])
            if step % args.ckpt_every == 0 and rank == 0:
                save_ckpt_atomic({**weights(), "opt": opt.state_dict()}, ck_path)
            if step >= total_steps:
                done = True
                break
            if args.time_limit_min is not None and step % 20 == 0:
                flag = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit_min), device=device)
                if world > 1:
                    dist.broadcast(flag, src=0)
                if flag.item():
                    stop_reason, done = "time_limit", True
                    break
        else:
            epoch += 1
    if stop_reason == "finished":
        m = validate(save_grid=True); emit(m)
        log.info("FINAL val mse %.4f perceptual %.3f", m["val/mse"], m["val/perceptual"])
        if rank == 0:
            torch.save({**weights(), "final_val": m}, run_dir / "gan_enc_final.pt")
            log.info("final weights: %s", run_dir / "gan_enc_final.pt")
    if rank == 0:
        save_ckpt_atomic({**weights(), "opt": opt.state_dict()}, ck_path)
        (run_dir / "status.json").write_text(json.dumps({"status": stop_reason, "step": step, "total_steps": total_steps,
                                                         "finished_at": dt.datetime.now().isoformat()}, indent=2))
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
