#!/usr/bin/env python3
"""
Train the discrete image tokenizer (VQ-VAE, ``src/models/vq_tokenizer.py``) of the autoregressive model on the TRAIN
splits of ArtBench-10 and WikiArt jointly (123k paintings, 128 x 128 px -> 16 x 16 = 256 tokens, codebook 4096), with
DistributedDataParallel and Weights & Biases.  Loss = L1 + VGG16 perceptual distance + codebook / commitment loss
(no adversarial loss).  Needs the pixel cache of ``scripts/20_prepare_ar_data.py``.

Validation = reconstruction on a fixed subset of both TEST splits: L1, PSNR, perceptual distance, codebook perplexity and
share of codes used.  Every run creates ``runs/<YYYYmmdd-HHMMSS>_<name>/`` with config.json, checkpoints/, vq_final.pt,
train.log, metrics.jsonl, recon/ (input / reconstruction grids) and wandb/.  Resumable (``--resume``).

    torchrun --standalone --nproc_per_node=4 scripts/21_train_vq.py --epochs 40            # 4 GPUs
    python scripts/21_train_vq.py --max-steps 20 --name smoke_vq                           # quick check
    torchrun ... scripts/21_train_vq.py --resume runs/<stamp>_vq_artbench_wikiart
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

from src.models.ar_data import DATASETS, pixel_file
from src.models.vq_tokenizer import VGGPerceptual, VQConfig, VQTokenizer
from src.training.common import init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("vq")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--name", type=str, default="vq_artbench_wikiart")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--cache-dir", type=str, default="data/ar_tokens")
    # model
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--codebook-size", type=int, default=4096)
    p.add_argument("--code-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--perceptual-weight", type=float, default=1.0)
    p.add_argument("--revive-every", type=int, default=200, help="re-initialise unused codes every N steps (0 = never)")
    p.add_argument("--revive-until", type=float, default=0.8, help="... until this fraction of training")
    # optimisation
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    # validation / logging / checkpointing
    p.add_argument("--val-n", type=int, default=2048, help="Fixed test images per dataset")
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


# ── data: both train splits as one array-like, random access from the page cache ─────────────────────────────

class PixelBank:
    def __init__(self, cache_dir: str, split: str) -> None:
        self.arrs = [np.load(pixel_file(d, split, cache_dir), mmap_mode="r") for d in DATASETS]
        self.offsets = np.cumsum([0] + [len(a) for a in self.arrs])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def get(self, idx: np.ndarray) -> np.ndarray:
        idx = np.sort(idx)
        out = np.empty((len(idx), 128, 128, 3), np.uint8)
        for k, a in enumerate(self.arrs):
            m = (idx >= self.offsets[k]) & (idx < self.offsets[k + 1])
            if m.any():
                out[m] = a[idx[m] - self.offsets[k]]
        return out


class Prefetcher:
    """Background thread producing uint8 batches for steps ``start, start+1, ...`` of this rank."""

    def __init__(self, bank: PixelBank, rank: int, world: int, bs: int, seed: int, start_step: int, steps_per_epoch: int):
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.args = (bank, rank, world, bs, seed, start_step, steps_per_epoch)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        bank, rank, world, bs, seed, step, spe = self.args
        n = len(bank)
        perm, ep = None, -1
        while True:
            if step // spe != ep:
                ep = step // spe
                perm = np.random.default_rng(seed * 1000003 + ep).permutation(n)
            j = (step % spe) * bs * world + rank * bs
            self.q.put(torch.from_numpy(bank.get(perm[j: j + bs])).pin_memory())
            step += 1

    def next(self) -> torch.Tensor:
        return self.q.get()


def to_input(u8: torch.Tensor, device, flip: bool = True) -> torch.Tensor:
    x = u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 127.5 - 1
    if flip:
        m = torch.rand(x.shape[0], device=device) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1), x.flip(-1), x)
    return x.contiguous(memory_format=torch.channels_last)


# ── validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, perc, val_sets, device, args, world, rank):
    """val_sets: {dataset: uint8 array (n, 128, 128, 3)}.  Returns metrics for every dataset and their mean."""
    model.eval()
    out = {}
    counts = torch.zeros(args.codebook_size, device=device)
    for name, arr in val_sets.items():
        s = torch.zeros(4, device=device)               # sum L1, sum PSNR, sum perceptual, n
        for i in range(rank * 64, len(arr), 64 * world):
            x = to_input(torch.from_numpy(arr[i: i + 64]), device, flip=False)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                rec, _, ids = model(x)
            rec = rec.float().clamp(-1, 1)
            mse01 = (((rec - x) / 2) ** 2).flatten(1).mean(1)
            s += torch.stack([((rec - x).abs() / 2).mean() * len(x), (10 * torch.log10(1 / mse01.clamp_min(1e-10))).sum(),
                              perc(rec, x) * len(x), torch.tensor(float(len(x)), device=device)])
            counts += torch.bincount(ids.reshape(-1), minlength=args.codebook_size).float()
        if world > 1:
            dist.all_reduce(s)
        out[f"val/{name}/l1"], out[f"val/{name}/psnr"], out[f"val/{name}/perceptual"] = (s[:3] / s[3]).tolist()
    if world > 1:
        dist.all_reduce(counts)
    p = counts / counts.sum()
    out["val/code_perplexity"] = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    out["val/codes_used"] = float((counts > 0).float().mean())
    for k in ("l1", "psnr", "perceptual"):
        out[f"val/{k}"] = float(np.mean([out[f"val/{n}/{k}"] for n in val_sets]))
    model.train()
    return out


@torch.no_grad()
def recon_grid(model, x, device, args):
    model.eval()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
        rec, _, _ = model(x)
    model.train()
    rows = [x, rec.float().clamp(-1, 1)]
    return torch.cat([make_grid((r.cpu() + 1) / 2, nrow=x.shape[0], padding=2) for r in rows], dim=1)


def main() -> None:
    args = parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        saved = json.loads((resume_dir / "config.json").read_text())["args"]
        keep = {k: getattr(args, k) for k in ("resume", "wandb", "time_limit_min", "max_steps")}
        for k, v in saved.items():
            setattr(args, k, v)
        for k, v in keep.items():
            if k != "max_steps" or v is not None:
                setattr(args, k, v)
        run_dir = resume_dir
    else:
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
        (run_dir / "recon").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── data ─────────────────────────────────────────────────────────────────
    bank = PixelBank(args.cache_dir, "train")
    val_sets = {}
    for d in DATASETS:
        a = np.load(pixel_file(d, "test", args.cache_dir), mmap_mode="r")
        sel = np.sort(np.random.default_rng(1234).permutation(len(a))[: args.val_n])
        val_sets[d] = np.ascontiguousarray(a[sel])
    steps_per_epoch = len(bank) // (args.batch_size * world)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    # ── model / optim ────────────────────────────────────────────────────────
    mcfg = VQConfig(base_ch=args.base_ch, codebook_size=args.codebook_size, code_dim=args.code_dim, beta=args.beta)
    model = VQTokenizer(mcfg).to(device).to(memory_format=torch.channels_last)
    perc = VGGPerceptual().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    step, best_val = 0, float("inf")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck["best_val"]
        log.info("Resumed from %s at step %d", ckpt_path, step)

    info = {"train_datasets": list(DATASETS), "img_size": 128, "n_tokens": mcfg.n_tokens,
            "codebook_size": mcfg.codebook_size, "code_dim": mcfg.code_dim, "name": args.name}
    cfg = {"args": vars(args), "model": mcfg.to_dict(), "info": info, "n_params": n_params,
           "train_images": len(bank), "world_size": world, "global_batch_size": args.batch_size * world,
           "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "host": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("params=%.2fM train=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d",
                 n_params / 1e6, len(bank), world, cfg["global_batch_size"], steps_per_epoch, total_steps)
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

    def checkpoint(numbered: bool = False) -> None:
        if rank != 0:
            return
        save_ckpt_atomic({**weights(), "optimizer": opt.state_dict(), "best_val": best_val, "args": vars(args)}, ckpt_path)
        if numbered:
            save_ckpt_atomic(weights(), run_dir / "checkpoints" / f"step_{step:07d}.pt")
            for p in sorted((run_dir / "checkpoints").glob("step_*.pt"))[:-args.keep_ckpts]:
                p.unlink()

    grid_x = None
    if rank == 0:
        grid_x = torch.cat([to_input(torch.from_numpy(val_sets[d][:: len(val_sets[d]) // 8][:8]), device, flip=False)
                            for d in DATASETS])

    def log_samples() -> None:
        if rank != 0:
            return
        grid = recon_grid(model, grid_x, device, args)
        save_image(grid, run_dir / "recon" / f"step_{step:07d}.png")
        if run is not None:
            import wandb
            run.log({"step": step, "samples/recon": wandb.Image(
                grid, caption=f"step {step}: top = input (8 ArtBench + 8 WikiArt test images), bottom = reconstruction")},
                step=step)

    def run_val() -> None:
        nonlocal best_val
        m = validate(model, perc, val_sets, device, args, world, rank)
        emit(m)
        log.info("step %d  val L1 %.4f PSNR %.2f perceptual %.4f  codes used %.3f perplexity %.0f", step, m["val/l1"],
                 m["val/psnr"], m["val/perceptual"], m["val/codes_used"], m["val/code_perplexity"])
        score = m["val/l1"] + args.perceptual_weight * m["val/perceptual"]
        if score < best_val:
            best_val = score
            if rank == 0:
                save_ckpt_atomic({**weights(), "val": m}, run_dir / "checkpoints" / "best.pt")

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    prefetch = Prefetcher(bank, rank, world, args.batch_size, args.seed, step, steps_per_epoch)
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    acc, n_acc = torch.zeros(5, device=device), 0
    if step == 0:
        run_val()
        log_samples()
    while step < total_steps:
        x = to_input(prefetch.next(), device)
        lr = lr_at(step, args.lr, args, total_steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            rec, qloss, ids = ddp(x)
        rec = rec.float()
        l1 = (rec - x).abs().mean()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            pl = perc(rec, x)
        pl = pl.float()
        loss = l1 + args.perceptual_weight * pl + qloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
        opt.step()
        step += 1
        imgs_since += x.shape[0] * world
        model.update_usage(ids)
        acc += torch.stack([loss.detach(), l1.detach(), pl.detach(), qloss.detach(),
                            (model.usage > 0.02).float().mean()])
        n_acc += 1

        if args.revive_every and step % args.revive_every == 0 and step < args.revive_until * total_steps:
            pool = model.last_z[torch.randperm(model.last_z.shape[0], device=device)[:1024]]
            pools = [torch.empty_like(pool) for _ in range(world)]
            if world > 1:
                dist.all_gather(pools, pool)
            else:
                pools = [pool]
            dead = None
            if rank == 0:
                dead = model.revive_dead_codes(torch.cat(pools))
            if world > 1:
                dist.broadcast(model.codebook.data, 0)
                dist.broadcast(model.usage, 0)
                nd = torch.tensor([len(dead) if dead is not None else 0], device=device)
                dist.broadcast(nd, 0)
                if rank != 0:
                    dead = torch.empty(int(nd.item()), dtype=torch.long, device=device)
                dist.broadcast(dead, 0)
            st = opt.state.get(model.codebook)
            if st and len(dead):
                st["exp_avg"][dead] = 0
                st["exp_avg_sq"][dead] = 0
            if rank == 0 and len(dead):
                log.info("step %d: re-initialised %d unused codes", step, len(dead))

        if step % args.log_every == 0:
            a = reduce_mean(acc / n_acc, world)
            now = time.time()
            emit({"train/loss": a[0].item(), "train/l1": a[1].item(), "train/perceptual": a[2].item(),
                  "train/quant_loss": a[3].item(), "train/codes_active": a[4].item(), "lr": lr,
                  "epoch": step / steps_per_epoch, "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                  "perf/img_per_s": imgs_since / (now - t_last)})
            log.info("step %d/%d ep %.2f loss %.4f l1 %.4f perc %.4f q %.4f active %.2f lr %.2e %.0f img/s", step,
                     total_steps, step / steps_per_epoch, a[0].item(), a[1].item(), a[2].item(), a[3].item(),
                     a[4].item(), lr, imgs_since / (now - t_last))
            acc.zero_(); n_acc = 0; imgs_since = 0; t_last = now
        if step % args.val_every == 0:
            run_val()
        if step % args.sample_every == 0:
            log_samples()
        if step % args.ckpt_every == 0:
            checkpoint(numbered=True)
        if args.time_limit_min is not None and step % 20 == 0:
            flag = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit_min), device=device)
            if world > 1:
                dist.broadcast(flag, src=0)
            if flag.item():
                stop_reason = "time_limit"
                break

    # ── wrap up ──────────────────────────────────────────────────────────────
    if stop_reason == "finished":
        if step % args.val_every:
            run_val()
        if step % args.sample_every:
            log_samples()
        if rank == 0:
            torch.save({**weights()}, run_dir / "vq_final.pt")
            log.info("final weights: %s", run_dir / "vq_final.pt")
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
    checkpoint(numbered=False)
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
