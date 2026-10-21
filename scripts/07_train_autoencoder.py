#!/usr/bin/env python3
"""
Train a convolutional autoencoder (reconstruction objective) on the union of
ArtBench-10 and WikiArt, with DistributedDataParallel and Weights & Biases.

Train set = ArtBench train + WikiArt train.  Validation set = ArtBench test +
WikiArt test (held out, optionally sub-sampled with --val-max-samples).

Every run creates a NEW directory ``runs/<YYYYmmdd-HHMMSS>_<name>/`` holding
config.json, checkpoints/, the final weights, train.log, metrics.jsonl,
recon/ (input-vs-output grids) and wandb/ (wandb run files).

Usage
-----
# single GPU / quick check
    python scripts/07_train_autoencoder.py --max-steps 20 --name smoke

# 4 GPUs on one node (DDP)
    torchrun --standalone --nproc_per_node=4 scripts/07_train_autoencoder.py \\
        --epochs 30 --name ae_union

# continue a run after the Slurm time limit (same run dir, same wandb run)
    torchrun --standalone --nproc_per_node=4 scripts/07_train_autoencoder.py \\
        --resume runs/20261015-120000_ae_union

On a JURECA compute node (usually no internet) wandb falls back to offline
mode automatically; sync later from the login node with
    wandb sync runs/<run>/wandb/offline-run-*
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.datasets import default_data_dir
from src.datasets.union import SOURCES, UnionImageDataset
from src.models.autoencoder import Autoencoder, AutoencoderConfig
from src.training.common import (OffsetSampler, init_wandb, lr_at, reduce_mean, save_ckpt_atomic,
                                 setup_dist, setup_logging)

log = logging.getLogger("ae")


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--name", type=str, default="ae_union", help="Short run name (suffix of the run dir)")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--artbench-dir", type=str, default=default_data_dir("artbench"))
    p.add_argument("--wikiart-dir", type=str, default=default_data_dir("wikiart"))
    # model
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=64)
    p.add_argument("--blocks-per-stage", type=int, default=1)
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak LR for global batch 256 (scaled linearly)")
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=16, help="DataLoader workers per GPU")
    # validation / logging / checkpointing
    p.add_argument("--val-max-samples", type=int, default=4096, help="Fixed random subset of the held-out split (0 = all)")
    p.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=500, help="Log reconstruction grids every N steps")
    p.add_argument("--n-samples", type=int, default=16, help="Images per reconstruction grid (half per dataset)")
    p.add_argument("--ckpt-every", type=int, default=500, help="Write checkpoints/last.pt every N steps")
    p.add_argument("--keep-ckpts", type=int, default=2, help="Numbered step checkpoints to keep (last.pt/best.pt always kept)")
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="Checkpoint and exit cleanly after this many minutes (set below the Slurm limit)")
    # wandb
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto",
                   help="auto: honour WANDB_MODE, else online if api.wandb.ai is reachable, else offline")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def build_data(args, world: int, rank: int):
    S = args.img_size
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    train_tf = transforms.Compose([
        transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomCrop(S),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(S, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(S),
        transforms.ToTensor(), norm,
    ])
    train_ds = UnionImageDataset(args.artbench_dir, args.wikiart_dir, "train", train_tf, draft_size=S)
    val_full = UnionImageDataset(args.artbench_dir, args.wikiart_dir, "test", val_tf, draft_size=S)

    g = torch.Generator().manual_seed(1234)  # fixed val subset, independent of run seed
    n_val = len(val_full) if args.val_max_samples <= 0 else min(args.val_max_samples, len(val_full))
    val_idx = sorted(torch.randperm(len(val_full), generator=g)[:n_val].tolist())
    val_ds = Subset(val_full, val_idx)

    # fixed images for reconstruction grids: n/2 per source dataset
    per = max(1, args.n_samples // len(SOURCES))
    sample_idx = []
    for sid in range(len(SOURCES)):
        sample_idx += [i for i in val_idx if val_full.sources[i] == sid][:per]
    sample_ds = Subset(val_full, sample_idx)
    return train_ds, val_ds, sample_ds


# ── misc ──────────────────────────────────────────────────────────────────────

def recon_loss(pred, target, args):
    pred = pred.float()
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return args.l1_weight * l1 + args.mse_weight * mse, l1, mse


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    # images in [-1, 1] -> data range 2
    return 10 * torch.log10(4.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def validate(model, loader, device, args, world: int):
    model.eval()
    sums = torch.zeros(4, device=device)  # loss*n, l1*n, mse*n, n
    for x, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            pred = model(x)
        loss, l1, mse = recon_loss(pred, x, args)
        n = x.shape[0]
        sums += torch.stack([loss * n, l1 * n, mse * n, torch.tensor(float(n), device=device)])
    if world > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    n = sums[3].clamp_min(1)
    loss, l1, mse = (sums[:3] / n).tolist()
    return {"val/loss": loss, "val/l1": l1, "val/mse": mse,
            "val/psnr": float(psnr_from_mse(torch.tensor(mse)))}


@torch.no_grad()
def make_recon_grid(model, sample_x: torch.Tensor, device, args):
    model.eval()
    x = sample_x.to(device).contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
        pred = model(x)
    model.train()
    to01 = lambda t: (t.float().clamp(-1, 1) + 1) / 2
    n = x.shape[0]
    per_row = n // 2 if n >= 8 else n
    # top block: inputs, bottom block: reconstructions, rows of ``per_row`` images
    grid = torch.cat([make_grid(to01(x).cpu(), nrow=per_row, padding=2),
                      make_grid(to01(pred).cpu(), nrow=per_row, padding=2)], dim=1)
    return grid


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)

    # ── run directory (rank 0 decides, everyone follows) ─────────────────────
    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        saved = json.loads((resume_dir / "config.json").read_text())["args"]
        keep = {k: getattr(args, k) for k in ("resume", "wandb", "time_limit_min", "num_workers", "max_steps")}
        for k, v in saved.items():
            setattr(args, k, v)
        for k, v in keep.items():  # CLI values that may legitimately change on resume
            if k != "max_steps" or v is not None:
                setattr(args, k, v)
        run_dir = resume_dir
    else:
        # all ranks derive the same timestamp from rank 0's clock
        t = torch.tensor([int(time.time())], dtype=torch.int64, device=device)
        if world > 1:
            dist.broadcast(t, src=0)
        stamp = dt.datetime.fromtimestamp(int(t.item())).strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / args.runs_dir / f"{stamp}_{args.name}"
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(run_dir, rank)
    if rank == 0:
        (run_dir / "recon").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, sample_ds = build_data(args, world, rank)
    sampler = OffsetSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    train_loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                           drop_last=True, persistent_workers=False,
                           prefetch_factor=4 if args.num_workers > 0 else None)
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=min(args.num_workers, 8), pin_memory=True)
    sample_x = torch.stack([sample_ds[i][0] for i in range(len(sample_ds))]) if rank == 0 else None

    # ── model / optim ────────────────────────────────────────────────────────
    mcfg = AutoencoderConfig(base_channels=args.base_channels, latent_channels=args.latent_channels,
                             blocks_per_stage=args.blocks_per_stage)
    model = Autoencoder(mcfg).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    peak_lr = args.lr * (args.batch_size * world) / 256
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    step, epoch, best_val = 0, 0, float("inf")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, epoch, best_val = ck["step"], ck["epoch"], ck["best_val"]
        log.info("Resumed from %s at step %d (epoch %d)", ckpt_path, step, epoch)
    # position inside the (deterministically shuffled) epoch
    batches_done_in_epoch = step - epoch * steps_per_epoch
    if batches_done_in_epoch >= steps_per_epoch:  # checkpoint was exactly at an epoch boundary
        epoch, batches_done_in_epoch = epoch + 1, 0

    # ── config + wandb ───────────────────────────────────────────────────────
    cfg = {
        "args": vars(args), "model": mcfg.to_dict(), "n_params": n_params,
        "train_images": len(train_ds), "val_images": len(val_ds), "world_size": world,
        "global_batch_size": args.batch_size * world, "peak_lr": peak_lr,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
    }
    run = None
    if rank == 0:
        if resume_dir is None or not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:  # keep original config.json, record the resume
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("params=%.2fM train=%d val=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d",
                 n_params / 1e6, len(train_ds), len(val_ds), world, cfg["global_batch_size"],
                 steps_per_epoch, total_steps)
        run = init_wandb(args, cfg, run_dir)
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

    def checkpoint(name: str = "last.pt", also_numbered: bool = False) -> None:
        if rank != 0:
            return
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "epoch": epoch,
                 "best_val": best_val, "model_cfg": mcfg.to_dict(), "args": vars(args)}
        save_ckpt_atomic(state, run_dir / "checkpoints" / name)
        if also_numbered:
            save_ckpt_atomic({k: state[k] for k in ("model", "step", "model_cfg")},
                             run_dir / "checkpoints" / f"step_{step:07d}.pt")
            old = sorted((run_dir / "checkpoints").glob("step_*.pt"))[:-args.keep_ckpts]
            for p in old:
                p.unlink()

    def log_samples() -> None:
        if rank != 0:
            return
        grid = make_recon_grid(model, sample_x, device, args)
        out = run_dir / "recon" / f"step_{step:07d}.png"
        save_image(grid, out)
        if run is not None:
            import wandb
            run.log({"step": step, "samples/recon_top_input_bottom_output":
                     wandb.Image(grid, caption=f"step {step}: top = input, bottom = reconstruction "
                                               f"(left half ArtBench, right half WikiArt)")}, step=step)

    def run_val() -> None:
        nonlocal best_val
        m = validate(ddp, val_loader, device, args, world)
        emit(m)
        log.info("step %d  val loss %.4f  l1 %.4f  psnr %.2f dB", step, m["val/loss"], m["val/l1"], m["val/psnr"])
        if m["val/loss"] < best_val:
            best_val = m["val/loss"]
            if rank == 0:
                save_ckpt_atomic({"model": model.state_dict(), "step": step, "model_cfg": mcfg.to_dict(),
                                  "val": m}, run_dir / "checkpoints" / "best.pt")

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    acc = torch.zeros(3, device=device)  # loss, l1, mse accumulated between logs
    n_acc = 0
    if rank == 0 and step == 0:
        log_samples()  # untrained baseline
    done = step >= total_steps
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **train_loader_kw)
        for x, _ in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            lr = lr_at(step, peak_lr, args, total_steps)
            for gr in opt.param_groups:
                gr["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                pred = ddp(x)
            loss, l1, mse = recon_loss(pred, x, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
            opt.step()
            step += 1
            imgs_since += x.shape[0] * world
            acc += torch.stack([loss.detach(), l1.detach(), mse.detach()])
            n_acc += 1

            if step % args.log_every == 0:
                a = reduce_mean(acc / n_acc, world)
                now = time.time()
                emit({"train/loss": a[0].item(), "train/l1": a[1].item(), "train/mse": a[2].item(),
                      "train/psnr": float(psnr_from_mse(a[2].cpu())), "lr": lr, "epoch": step / steps_per_epoch,
                      "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                      "perf/img_per_s": imgs_since / (now - t_last)})
                log.info("step %d/%d ep %.2f loss %.4f l1 %.4f lr %.2e %.0f img/s", step, total_steps,
                         step / steps_per_epoch, a[0].item(), a[1].item(), lr, imgs_since / (now - t_last))
                acc.zero_(); n_acc = 0; imgs_since = 0; t_last = now

            if step % args.val_every == 0:
                run_val()
            if step % args.sample_every == 0:
                log_samples()
            if step % args.ckpt_every == 0:
                checkpoint(also_numbered=True)

            out_of_time = False
            if args.time_limit_min is not None and step % 20 == 0:
                flag = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit_min), device=device)
                if world > 1:
                    dist.broadcast(flag, src=0)  # all ranks must agree to stop
                out_of_time = bool(flag.item())
            if out_of_time:
                stop_reason = "time_limit"
                done = True
                break
            if step >= total_steps:
                done = True
                break
        else:
            epoch += 1
            batches_done_in_epoch = 0
            if step >= total_steps:
                done = True
        if stop_reason == "time_limit":
            break

    # ── wrap up ──────────────────────────────────────────────────────────────
    if stop_reason == "finished":
        if step % args.val_every:
            run_val()
        if step % args.sample_every:
            log_samples()
        if rank == 0:
            torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "step": step,
                        "img_size": args.img_size}, run_dir / "autoencoder_final.pt")
            log.info("final weights: %s", run_dir / "autoencoder_final.pt")
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
    # epoch is already incremented inside the loop's else-branch when an epoch completes
    checkpoint(also_numbered=False)
    if rank == 0:
        (run_dir / "status.json").write_text(json.dumps(
            {"status": stop_reason, "step": step, "total_steps": total_steps,
             "finished_at": dt.datetime.now().isoformat()}, indent=2))
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
