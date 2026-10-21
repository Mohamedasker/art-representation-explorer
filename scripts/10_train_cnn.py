#!/usr/bin/env python3
"""
Train a ResNet (supervised, multi-task classification) on ArtBench-10 + WikiArt.

Targets (see src/datasets/labeled_union.py): style (union vocabulary), genre (WikiArt only) and
artist (WikiArt + ArtBench images whose artist is in the vocabulary); missing labels are ignored.
Train = both train splits, validation = both test splits.  Same run-directory / DDP / wandb /
resume conventions as scripts/07_train_autoencoder.py:

    runs/<YYYYmmdd-HHMMSS>_<name>/  config.json, vocab.json, checkpoints/, cnn_final.pt,
                                    train.log, metrics.jsonl, wandb/

The globally pooled backbone feature (2048-D for resnet50) is the representation compared with
the unsupervised ones (scripts/02_extract_features.py --model cnn_supervised).

Usage
-----
    torchrun --standalone --nproc_per_node=4 scripts/10_train_cnn.py --epochs 30 --name cnn_resnet50
    torchrun --standalone --nproc_per_node=4 scripts/10_train_cnn.py --resume runs/<run>
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.datasets import default_data_dir
from src.datasets.labeled_union import HEADS, LabeledUnionDataset, build_vocab
from src.datasets.union import SOURCES
from src.models.supervised_cnn import MultiHeadCNN
from src.training.common import (OffsetSampler, init_wandb, lr_at, reduce_mean, save_ckpt_atomic,
                                 setup_dist, setup_logging)

log = logging.getLogger("cnn")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--name", type=str, default="cnn_resnet50")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--artbench-dir", type=str, default=default_data_dir("artbench"))
    p.add_argument("--wikiart-dir", type=str, default=default_data_dir("wikiart"))
    # model
    p.add_argument("--arch", type=str, default="resnet50", help="any timm model name")
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (default: from scratch)")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--drop-rate", type=float, default=0.1)
    # optimisation
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Peak LR for global batch 256 (scaled linearly)")
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--w-style", type=float, default=1.0)
    p.add_argument("--w-genre", type=float, default=0.5)
    p.add_argument("--w-artist", type=float, default=0.5)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=16, help="DataLoader workers per GPU")
    # validation / logging / checkpointing
    p.add_argument("--val-max-samples", type=int, default=4096, help="Fixed random subset of the held-out split (0 = all)")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="Checkpoint and exit cleanly after this many minutes (set below the Slurm limit)")
    # wandb
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


def build_data(args, vocab):
    S = args.img_size
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    bicubic = transforms.InterpolationMode.BICUBIC
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.35, 1.0), interpolation=bicubic, antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    val_tf = transforms.Compose([
        transforms.Resize(S, interpolation=bicubic, antialias=True), transforms.CenterCrop(S),
        transforms.ToTensor(), norm])
    kw = dict(draft_size=S, vocab=vocab)
    train_ds = LabeledUnionDataset(args.artbench_dir, args.wikiart_dir, "train", train_tf, **kw)
    val_full = LabeledUnionDataset(args.artbench_dir, args.wikiart_dir, "test", val_tf, **kw)
    g = torch.Generator().manual_seed(1234)
    n_val = len(val_full) if args.val_max_samples <= 0 else min(args.val_max_samples, len(val_full))
    val_idx = sorted(torch.randperm(len(val_full), generator=g)[:n_val].tolist())
    return train_ds, val_full, Subset(val_full, val_idx)


def masked_ce(logits: torch.Tensor, target: torch.Tensor, smoothing: float):
    """(mean CE over labelled samples, n labelled, n correct); target -1 = no label."""
    logits = logits.float()
    n = (target >= 0).sum()
    ce = F.cross_entropy(logits, target, ignore_index=-1, label_smoothing=smoothing, reduction="sum")
    correct = ((logits.argmax(1) == target) & (target >= 0)).sum()
    return ce / n.clamp_min(1), n, correct


def compute_loss(out: dict, tgt: torch.Tensor, src: torch.Tensor, args):
    """Returns total loss and a stats vector [loss_h*3, n_h*3, correct_h*3, style_correct/n per source*2*2]."""
    w = {"style": args.w_style, "genre": args.w_genre, "artist": args.w_artist}
    total, losses, ns, cs = 0.0, [], [], []
    for k, h in enumerate(HEADS):
        l, n, c = masked_ce(out[h], tgt[:, k], args.label_smoothing)
        total = total + w[h] * l
        losses.append(l.detach() * n); ns.append(n); cs.append(c)
    hit = (out["style"].argmax(1) == tgt[:, 0]).float()
    per_src = []
    for s in range(len(SOURCES)):
        m = src == s
        per_src += [(hit * m).sum(), m.sum().float()]
    stats = torch.stack([*losses, *[x.float() for x in ns], *[x.float() for x in cs], *per_src])
    return total, stats


def stats_to_metrics(stats: torch.Tensor, prefix: str) -> dict:
    s = stats.tolist()
    nh = len(HEADS)
    out = {}
    for k, h in enumerate(HEADS):
        n = max(s[nh + k], 1.0)
        out[f"{prefix}/loss_{h}"] = s[k] / n
        out[f"{prefix}/acc_{h}"] = s[2 * nh + k] / n
    base = 3 * nh
    for j, name in enumerate(SOURCES):
        out[f"{prefix}/acc_style_{name}"] = s[base + 2 * j] / max(s[base + 2 * j + 1], 1.0)
    return out


@torch.no_grad()
def validate(model, loader, device, args, world: int):
    model.eval()
    acc = torch.zeros(3 * len(HEADS) + 2 * len(SOURCES), device=device)
    for x, tgt, src in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        tgt, src = tgt.to(device), src.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            out = model(x)
        _, st = compute_loss(out, tgt, src, args)
        acc += st
    if world > 1:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    model.train()
    return stats_to_metrics(acc, "val")


def main() -> None:
    args = parse_args()
    t_start = time.time()
    rank, world, local, device = setup_dist()
    setup_logging(None, rank)

    # ── run directory ────────────────────────────────────────────────────────
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
        t = torch.tensor([int(time.time())], dtype=torch.int64, device=device)
        if world > 1:
            dist.broadcast(t, src=0)
        stamp = dt.datetime.fromtimestamp(int(t.item())).strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / args.runs_dir / f"{stamp}_{args.name}"
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(run_dir, rank)
    if rank == 0:
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── data ─────────────────────────────────────────────────────────────────
    vocab = build_vocab(args.wikiart_dir)
    train_ds, val_full, val_ds = build_data(args, vocab)
    sampler = OffsetSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    train_loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                           drop_last=True, prefetch_factor=4 if args.num_workers > 0 else None)

    def make_val_loader(ds):
        smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        return DataLoader(ds, batch_size=args.batch_size, sampler=smp,
                          num_workers=min(args.num_workers, 8), pin_memory=True)

    val_loader = make_val_loader(val_ds)

    # ── model / optim ────────────────────────────────────────────────────────
    head_sizes = {h: len(vocab[h]) for h in HEADS}
    model = MultiHeadCNN(args.arch, head_sizes, pretrained=args.pretrained, drop_rate=args.drop_rate)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    peak_lr = args.lr * (args.batch_size * world) / 256
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=peak_lr, betas=(0.9, 0.999))

    step, epoch, best_val = 0, 0, 0.0          # best_val = best val style accuracy
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

    # ── config + wandb ───────────────────────────────────────────────────────
    cfg = {"args": vars(args), "head_sizes": head_sizes, "n_params": n_params,
           "train_images": len(train_ds), "val_images": len(val_ds), "world_size": world,
           "global_batch_size": args.batch_size * world, "peak_lr": peak_lr,
           "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
           "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "torch": torch.__version__}
    run = None
    if rank == 0:
        if resume_dir is None or not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            (run_dir / "vocab.json").write_text(json.dumps(vocab, indent=1))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("%s params=%.2fM heads=%s train=%d val=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d",
                 args.arch, n_params / 1e6, head_sizes, len(train_ds), len(val_ds), world,
                 cfg["global_batch_size"], steps_per_epoch, total_steps)
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

    def model_state() -> dict:
        return {"backbone": model.backbone.state_dict(), "heads": model.heads.state_dict(),
                "arch": args.arch, "head_sizes": head_sizes, "vocab": vocab, "img_size": args.img_size,
                "step": step}

    def checkpoint(also_numbered: bool = False) -> None:
        if rank != 0:
            return
        save_ckpt_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                          "epoch": epoch, "best_val": best_val, "args": vars(args)},
                         run_dir / "checkpoints" / "last.pt")
        if also_numbered:
            save_ckpt_atomic(model_state(), run_dir / "checkpoints" / f"step_{step:07d}.pt")
            for p in sorted((run_dir / "checkpoints").glob("step_*.pt"))[:-args.keep_ckpts]:
                p.unlink()

    def run_val() -> None:
        nonlocal best_val
        m = validate(ddp, val_loader, device, args, world)
        emit(m)
        log.info("step %d  val acc style %.3f (artbench %.3f / wikiart %.3f)  genre %.3f  artist %.3f",
                 step, m["val/acc_style"], m["val/acc_style_artbench"], m["val/acc_style_wikiart"],
                 m["val/acc_genre"], m["val/acc_artist"])
        if m["val/acc_style"] > best_val:
            best_val = m["val/acc_style"]
            if rank == 0:
                save_ckpt_atomic({**model_state(), "val": m}, run_dir / "checkpoints" / "best.pt")

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    nstat = 3 * len(HEADS) + 2 * len(SOURCES)
    acc = torch.zeros(nstat, device=device)
    loss_acc = torch.zeros(1, device=device)
    n_acc = 0
    done = step >= total_steps
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **train_loader_kw)
        for x, tgt, src in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            tgt, src = tgt.to(device, non_blocking=True), src.to(device, non_blocking=True)
            lr = lr_at(step, peak_lr, args, total_steps)
            for gr in opt.param_groups:
                gr["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                out = ddp(x)
            loss, stats = compute_loss(out, tgt, src, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
            opt.step()
            step += 1
            imgs_since += x.shape[0] * world
            acc += stats.detach()
            loss_acc += loss.detach()
            n_acc += 1

            if step % args.log_every == 0:
                if world > 1:
                    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                lo = reduce_mean(loss_acc / n_acc, world)
                now = time.time()
                m = stats_to_metrics(acc, "train")
                emit({"train/loss": lo.item(), **m, "lr": lr, "epoch": step / steps_per_epoch,
                      "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                      "perf/img_per_s": imgs_since / (now - t_last)})
                log.info("step %d/%d ep %.2f loss %.4f style acc %.3f genre %.3f artist %.3f lr %.2e %.0f img/s",
                         step, total_steps, step / steps_per_epoch, lo.item(), m["train/acc_style"],
                         m["train/acc_genre"], m["train/acc_artist"], lr, imgs_since / (now - t_last))
                acc.zero_(); loss_acc.zero_(); n_acc = 0; imgs_since = 0; t_last = now

            if step % args.val_every == 0:
                run_val()
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
        # final evaluation on the complete held-out split
        full = validate(ddp, make_val_loader(val_full), device, args, world)
        full = {k.replace("val/", "val_full/"): v for k, v in full.items()}
        emit(full)
        log.info("FINAL (full held-out split): " + "  ".join(f"{k}={v:.4f}" for k, v in full.items() if "acc" in k))
        if rank == 0:
            torch.save(model_state(), run_dir / "cnn_final.pt")
            log.info("final weights: %s", run_dir / "cnn_final.pt")
            if run is not None:
                run.summary.update(full)
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
    checkpoint()
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
