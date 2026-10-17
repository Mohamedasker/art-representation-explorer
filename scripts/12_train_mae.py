#!/usr/bin/env python3
"""
Train a Masked Autoencoder (ViT-B/16, MAE objective: reconstruct 75 % masked patches) on ONE dataset
(ArtBench-10 or WikiArt), with DistributedDataParallel and Weights & Biases.

Init (``--init``):
  imagenet  (default) continue MAE pre-training from Meta's ImageNet-1k MAE (encoder + decoder), i.e. domain
            adaptation of the ImageNet representation; small LR, few epochs.
  scratch   random init, full MAE recipe (higher LR, longer warm-up, needs many more epochs to get anywhere).

Train set = the dataset's train split, validation = its test split (fixed subset, fixed masks: the masked-patch
reconstruction loss on held-out paintings).  Every run creates a NEW directory
``runs/<YYYYmmdd-HHMMSS>_<name>/`` with config.json, checkpoints/, mae_final.pt, train.log, metrics.jsonl,
recon/ (input / masked / reconstruction / reconstruction + visible patches) and wandb/.

Usage
-----
# single GPU / quick check
    python scripts/12_train_mae.py --dataset artbench --max-steps 20 --name smoke

# 4 GPUs on one node (DDP)
    torchrun --standalone --nproc_per_node=4 scripts/12_train_mae.py --dataset artbench --epochs 60

# continue a run after the Slurm time limit (same run dir, same wandb run)
    torchrun --standalone --nproc_per_node=4 scripts/12_train_mae.py --resume runs/<stamp>_mae_artbench_imagenet

Embeddings of the finished run:
    python scripts/02_extract_features.py --model mae --checkpoint runs/<run>/mae_final.pt --dataset artbench --split both
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

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.datasets import build_dataset, default_data_dir
from src.models.mae import (IMAGENET_MEAN, IMAGENET_STD, MAEConfig, MaskedAutoencoderViT,
                            load_imagenet_mae)
from src.training.common import (OffsetSampler, init_wandb, lr_at, reduce_mean, save_ckpt_atomic,
                                 setup_dist, setup_logging)

log = logging.getLogger("mae")


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--dataset", choices=["artbench", "wikiart"], default="artbench")
    p.add_argument("--init", choices=["imagenet", "scratch"], default="imagenet")
    p.add_argument("--name", type=str, default=None, help="Run name (default mae_<dataset>_<init>)")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset root (default data/<dataset>)")
    # model
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--no-norm-pix-loss", action="store_true", help="Raw-pixel targets instead of per-patch normalised")
    # optimisation
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=None,
                   help="Base LR per 256 images (peak = lr * global_batch / 256). Default: 4e-5 for --init imagenet, "
                        "1.5e-4 for --init scratch (the official MAE value)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=None, help="Default: 200 (imagenet) / 5 epochs (scratch)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=30, help="DataLoader workers per GPU")
    p.add_argument("--draft-size", type=int, default=320,
                   help="JPEG draft decode: shrink big images (WikiArt) at decode time to >= this many px")
    p.add_argument("--rrc-min-scale", type=float, default=0.2, help="RandomResizedCrop minimum area fraction")
    # validation / logging / checkpointing
    p.add_argument("--val-max-samples", type=int, default=4096, help="Fixed random subset of the test split (0 = all)")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=500, help="Log reconstruction grids every N steps")
    p.add_argument("--n-samples", type=int, default=8, help="Images per reconstruction grid")
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="Checkpoint and exit cleanly after this many minutes (set below the Slurm limit)")
    # wandb
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

class PathImages(Dataset):
    """Label-free images of one dataset split; big JPEGs are shrunk at decode time (PIL draft)."""

    def __init__(self, dataset: str, root: str, split: str, transform, draft_size: int) -> None:
        self.paths = build_dataset(dataset, root=root, split=split).get_paths()
        self.transform, self.draft_size = transform, draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = Image.open(self.paths[i])
        im.draft("RGB", (self.draft_size, self.draft_size))
        im = im.convert("RGB")
        return self.transform(im), 0


def build_data(args):
    S = args.img_size
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    bic = transforms.InterpolationMode.BICUBIC
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(args.rrc_min_scale, 1.0), interpolation=bic, antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    # same preprocessing as the embedding extractor (Resize 256 + CenterCrop 224)
    val_tf = transforms.Compose([transforms.Resize(int(S * 256 / 224), interpolation=bic, antialias=True),
                                 transforms.CenterCrop(S), transforms.ToTensor(), norm])
    root = args.data_dir or default_data_dir(args.dataset)
    train_ds = PathImages(args.dataset, root, "train", train_tf, args.draft_size)
    val_full = PathImages(args.dataset, root, "test", val_tf, args.draft_size)
    g = torch.Generator().manual_seed(1234)
    n_val = len(val_full) if args.val_max_samples <= 0 else min(args.val_max_samples, len(val_full))
    val_idx = sorted(torch.randperm(len(val_full), generator=g)[:n_val].tolist())
    val_ds = Subset(val_full, val_idx)
    sample_ds = Subset(val_full, val_idx[:: max(1, len(val_idx) // args.n_samples)][: args.n_samples])
    return train_ds, val_ds, sample_ds


# ── validation / visualisation ────────────────────────────────────────────────

def _denorm(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() * s + m).clamp(0, 1)


@torch.no_grad()
def validate(model, loader, device, args, world: int):
    """Masked-patch reconstruction loss on held-out images, with fixed masks (same on every call)."""
    model.eval()
    sums = torch.zeros(2, device=device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(4321 + dist.get_rank() if world > 1 else 4321)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                loss, _, _ = model(x, args.mask_ratio)
            sums += torch.stack([loss.float() * x.shape[0], torch.tensor(float(x.shape[0]), device=device)])
    if world > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    return {"val/loss": (sums[0] / sums[1].clamp_min(1)).item()}


@torch.no_grad()
def make_recon_grid(model: MaskedAutoencoderViT, x: torch.Tensor, device, args):
    """Rows: input / masked input / reconstruction / reconstruction with the visible patches pasted back."""
    model.eval()
    x = x.to(device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(7)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            _, pred, mask = model(x, args.mask_ratio)
    pred = pred.float()
    if model.cfg.norm_pix_loss:  # undo the per-patch normalisation with the true patch statistics
        t = model.patchify(x)
        pred = pred * (t.var(dim=-1, keepdim=True) + 1e-6) ** 0.5 + t.mean(dim=-1, keepdim=True)
    rec = model.unpatchify(pred)
    m = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, model.cfg.patch_size ** 2 * 3).float())
    rows = [x, x * (1 - m), rec, x * (1 - m) + rec * m]
    model.train()
    n = x.shape[0]
    return torch.cat([make_grid(_denorm(r).cpu(), nrow=n, padding=2) for r in rows], dim=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        saved = json.loads((resume_dir / "config.json").read_text())["args"]
        keep = {k: getattr(args, k) for k in ("resume", "wandb", "time_limit_min", "num_workers", "max_steps")}
        for k, v in saved.items():
            setattr(args, k, v)
        for k, v in keep.items():
            if k != "max_steps" or v is not None:
                setattr(args, k, v)
        run_dir = resume_dir
    else:
        if args.name is None:
            args.name = f"mae_{args.dataset}_{args.init}"
        if args.lr is None:
            args.lr = 4e-5 if args.init == "imagenet" else 1.5e-4
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, sample_ds = build_data(args)
    sampler = OffsetSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    if args.warmup_steps is None:
        args.warmup_steps = 200 if args.init == "imagenet" else 5 * steps_per_epoch
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True,
              prefetch_factor=4 if args.num_workers > 0 else None)
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=min(args.num_workers, 8), pin_memory=True)
    sample_x = torch.stack([sample_ds[i][0] for i in range(len(sample_ds))]) if rank == 0 else None

    # ── model / optim ────────────────────────────────────────────────────────
    have_ckpt = resume_dir is not None and (run_dir / "checkpoints" / "last.pt").exists()
    if args.init == "imagenet" and not have_ckpt:
        model = load_imagenet_mae()
        model.cfg.norm_pix_loss = not args.no_norm_pix_loss
        log.info("initialised from Meta's ImageNet-1k MAE ViT-B/16 (encoder + decoder)")
    else:  # scratch, or the weights are loaded from last.pt below
        model = MaskedAutoencoderViT(MAEConfig(img_size=args.img_size, norm_pix_loss=not args.no_norm_pix_loss))
    model = model.to(device)
    mcfg = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    peak_lr = args.lr * (args.batch_size * world) / 256
    opt = torch.optim.AdamW(model.param_groups(args.weight_decay), lr=peak_lr, betas=(0.9, 0.95))

    step, epoch, best_val = 0, 0, float("inf")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, epoch, best_val = ck["step"], ck["epoch"], ck["best_val"]
        log.info("Resumed from %s at step %d (epoch %d)", ckpt_path, step, epoch)
    batches_done_in_epoch = step - epoch * steps_per_epoch
    if batches_done_in_epoch >= steps_per_epoch:
        epoch, batches_done_in_epoch = epoch + 1, 0

    info = {"init": args.init, "train_dataset": args.dataset, "mask_ratio": args.mask_ratio,
            "img_size": args.img_size, "embedding": "mean-pooled encoder patch tokens, unmasked input"}

    # ── config + wandb ───────────────────────────────────────────────────────
    cfg = {
        "args": vars(args), "model": mcfg.to_dict(), "info": info, "n_params": n_params,
        "train_images": len(train_ds), "val_images": len(val_ds), "world_size": world,
        "global_batch_size": args.batch_size * world, "peak_lr": peak_lr,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
    }
    run = None
    if rank == 0:
        if resume_dir is None or not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("params=%.2fM train=%d val=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d peak_lr=%.2e",
                 n_params / 1e6, len(train_ds), len(val_ds), world, cfg["global_batch_size"],
                 steps_per_epoch, total_steps, peak_lr)
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

    def weights() -> dict:
        return {"model": model.state_dict(), "step": step, "model_cfg": mcfg.to_dict(), "info": info}

    def checkpoint(also_numbered: bool = False) -> None:
        if rank != 0:
            return
        save_ckpt_atomic({**weights(), "optimizer": opt.state_dict(), "epoch": epoch, "best_val": best_val,
                          "args": vars(args)}, run_dir / "checkpoints" / "last.pt")
        if also_numbered:
            save_ckpt_atomic(weights(), run_dir / "checkpoints" / f"step_{step:07d}.pt")
            for p in sorted((run_dir / "checkpoints").glob("step_*.pt"))[:-args.keep_ckpts]:
                p.unlink()

    def log_samples() -> None:
        if rank != 0:
            return
        grid = make_recon_grid(model, sample_x, device, args)
        save_image(grid, run_dir / "recon" / f"step_{step:07d}.png")
        if run is not None:
            import wandb
            run.log({"step": step, "samples/recon": wandb.Image(
                grid, caption=f"step {step}: rows = input / masked / reconstruction / reconstruction + visible")},
                step=step)

    def run_val() -> None:
        nonlocal best_val
        m = validate(ddp, val_loader, device, args, world)
        emit(m)
        log.info("step %d  val loss %.4f", step, m["val/loss"])
        if m["val/loss"] < best_val:
            best_val = m["val/loss"]
            if rank == 0:
                save_ckpt_atomic({**weights(), "val": m}, run_dir / "checkpoints" / "best.pt")

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    acc, n_acc = torch.zeros(1, device=device), 0
    if step == 0:
        run_val()       # baseline before any update (for --init imagenet: the ImageNet MAE on this domain)
        log_samples()
    done = step >= total_steps
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **kw)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            lr = lr_at(step, peak_lr, args, total_steps)
            for gr in opt.param_groups:
                gr["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                loss, _, _ = ddp(x, args.mask_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
            opt.step()
            step += 1
            imgs_since += x.shape[0] * world
            acc += loss.detach().float()
            n_acc += 1

            if step % args.log_every == 0:
                a = reduce_mean(acc / n_acc, world)
                now = time.time()
                emit({"train/loss": a[0].item(), "lr": lr, "epoch": step / steps_per_epoch,
                      "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                      "perf/img_per_s": imgs_since / (now - t_last)})
                log.info("step %d/%d ep %.2f loss %.4f lr %.2e %.0f img/s", step, total_steps,
                         step / steps_per_epoch, a[0].item(), lr, imgs_since / (now - t_last))
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
                    dist.broadcast(flag, src=0)
                out_of_time = bool(flag.item())
            if out_of_time:
                stop_reason, done = "time_limit", True
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
            torch.save({**weights(), "img_size": args.img_size}, run_dir / "mae_final.pt")
            log.info("final weights: %s", run_dir / "mae_final.pt")
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
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
