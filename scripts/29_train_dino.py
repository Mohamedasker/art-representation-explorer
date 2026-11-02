#!/usr/bin/env python3
"""
Train DINO v1 (Caron et al. 2021; self-distillation with no labels) on ArtBench-10 or WikiArt, ViT-B/16.

The official recipe (facebookresearch/dino main_dino.py), sized to the data: student + EMA teacher (momentum cosine
0.996 -> 1), multi-crop (2 global 224 px crops scale 0.4-1, --n-local-crops local 96 px crops scale 0.05-0.4), colour
jitter / grayscale / blur / solarisation, centring + sharpening (teacher temperature warm-up 0.04 -> 0.07, student 0.1),
3-layer MLP head with a weight-normalised last layer, AdamW with linearly scaled LR (warm-up + cosine), weight decay
cosine 0.04 -> 0.4, stochastic depth 0.1, per-parameter gradient clipping, last layer frozen in the first epoch, bf16.

Initialisation (--init):
  scratch   random weights.
  imagenet  Meta's DINO ViT-B/16 ImageNet-1k checkpoint (backbone AND projection heads of student and teacher; the
            official "full checkpoint", downloaded once into pretrained/dino_vitb16_full/), trained further with a lower LR.

Data: the uint8 256 px caches of scripts/29_cache_images.py (train split of --dataset, no labels used for training).
Monitors, logged to wandb / metrics.jsonl: loss, teacher entropy, entropy of the batch-mean teacher distribution and its
KL to uniform (collapse indicators), and every --probe-every epochs kNN (k=20) / linear-probe accuracy of the TEACHER
backbone CLS token on the style labels (10k train -> 4k test images, like the SimCLR runs) and a feature-spread number.

    runs/<YYYYmmdd-HHMMSS>_<name>/  config.json, checkpoints/last.pt (student, teacher, optimiser, centre), dino_final.pt
                                    (TEACHER backbone = the weights that are embedded), train.log, metrics.jsonl, status.json

    torchrun --standalone --nproc_per_node=4 scripts/29_train_dino.py --dataset artbench --init scratch --epochs 200 \\
        --name dino_scratch_artbench
    torchrun --standalone --nproc_per_node=4 scripts/29_train_dino.py --resume runs/<run>
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from src.datasets.dino_data import CachedImages, DINOAugmentation, eval_transform_uint8
from src.models.dino import (FULL_CKPT, FULL_CKPT_URL, IMAGENET_MEAN, IMAGENET_STD, DINOLoss, build_model,
                             cosine_schedule, load_official_full_checkpoint)
from src.training.common import OffsetSampler, init_wandb, save_ckpt_atomic, setup_dist, setup_logging
from src.training.probes import feature_std, knn_acc, linear_acc

log = logging.getLogger("dino")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--dataset", choices=["artbench", "wikiart"], default="artbench")
    p.add_argument("--name", type=str, default=None, help="Run name (default dino_<init>_<dataset>)")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--init", choices=["scratch", "imagenet"], default="scratch")
    p.add_argument("--full-ckpt", type=str, default=FULL_CKPT)
    # model / DINO
    p.add_argument("--out-dim", type=int, default=None, help="Prototypes K (default 65536 for imagenet init, 16384 scratch)")
    p.add_argument("--norm-last-layer", type=int, choices=[0, 1], default=None,
                   help="Fix the weight-norm gain of the last layer to 1 (default 0 for imagenet init, as the official ViT-B, 1 scratch)")
    p.add_argument("--drop-path", type=float, default=0.1, help="Stochastic depth of the student")
    p.add_argument("--student-temp", type=float, default=0.1)
    p.add_argument("--teacher-temp", type=float, default=0.07)
    p.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    p.add_argument("--warmup-teacher-temp-epochs", type=float, default=None, help="default 15%% of the epochs")
    p.add_argument("--momentum-teacher", type=float, default=0.996)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--freeze-last-layer", type=float, default=1.0, help="Epochs with the last layer frozen")
    # augmentation
    p.add_argument("--n-local-crops", type=int, default=6)
    p.add_argument("--global-scale", type=float, nargs=2, default=(0.4, 1.0))
    p.add_argument("--local-scale", type=float, nargs=2, default=(0.05, 0.4))
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimiser steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=None, help="Peak LR for global batch 256 (scaled linearly); default 5e-4 scratch / 1e-4 imagenet")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=float, default=None, help="LR warm-up epochs (default 10 scratch / 2 imagenet)")
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--weight-decay-end", type=float, default=0.4)
    p.add_argument("--clip-grad", type=float, default=3.0, help="Per-parameter gradient-norm clip (official DINO), 0 = off")
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=28, help="DataLoader workers per GPU")
    # monitoring / checkpointing
    p.add_argument("--probe-every", type=int, default=5, help="kNN / linear probe every N epochs (0 = off)")
    p.add_argument("--probe-train-n", type=int, default=10240)
    p.add_argument("--probe-test-n", type=int, default=4096)
    p.add_argument("--probe-at-start", type=int, choices=[0, 1], default=None, help="Probe before training (default 1 for imagenet init)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500, help="Steps between resumable checkpoints")
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="Checkpoint and exit cleanly after this many minutes (set below the Slurm limit)")
    # wandb
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


def apply_defaults(args) -> None:
    ad = args.init == "imagenet"
    args.name = args.name or f"dino_{'adapt' if ad else 'scratch'}_{args.dataset}"
    if args.out_dim is None:
        args.out_dim = 65536 if ad else 16384
    if args.norm_last_layer is None:
        args.norm_last_layer = 0 if ad else 1
    if args.lr is None:
        args.lr = 1e-4 if ad else 5e-4
    if args.warmup_epochs is None:
        args.warmup_epochs = 2 if ad else 10
    if args.warmup_teacher_temp_epochs is None:
        args.warmup_teacher_temp_epochs = max(1.0, 0.15 * args.epochs)
    if args.probe_at_start is None:
        args.probe_at_start = int(ad)


def prep(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 (B, C, H, W) on the GPU -> normalised float."""
    return (x.float().div_(255) - mean) / std


@torch.no_grad()
def encode(backbone, ds, idx, device, world, rank, mean, std, bs=256) -> torch.Tensor:
    """CLS features of ds[idx] (centre 224 crop); each rank encodes idx[rank::world], result in the order of idx."""
    sub = Subset(ds, idx[rank::world].tolist())
    out = []
    backbone.eval()
    for x, _ in DataLoader(sub, batch_size=bs, num_workers=8, pin_memory=True):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out.append(backbone(prep(x.to(device, non_blocking=True), mean, std)).float())
    f = torch.cat(out)
    if world > 1:
        g = [torch.empty_like(f) for _ in range(world)]
        dist.all_gather(g, f)
        f = torch.stack(g, 1).reshape(-1, f.shape[1])           # undo the idx[r::W] interleaving
    return f


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
        apply_defaults(args)
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
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # ── data ─────────────────────────────────────────────────────────────────
    n_crops = 2 + args.n_local_crops
    train_ds = CachedImages(args.dataset, "train", DINOAugmentation(tuple(args.global_scale), tuple(args.local_scale),
                                                                    args.n_local_crops))
    ev_tf = eval_transform_uint8()
    probe_train = CachedImages(args.dataset, "train", ev_tf)
    probe_test = CachedImages(args.dataset, "test", ev_tf)
    g = np.random.default_rng(1234)                             # fixed probe subsets, independent of the run seed
    nt = min(args.probe_train_n, len(probe_train)) // world * world
    ne = min(args.probe_test_n, len(probe_test)) // world * world
    idx_tr = np.sort(g.permutation(len(probe_train))[:nt])
    idx_te = np.sort(g.permutation(len(probe_test))[:ne])
    n_cls = len(probe_train.class_names)

    sampler = OffsetSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                     prefetch_factor=4 if args.num_workers > 0 else None, persistent_workers=False)

    # ── model: student (stochastic depth) + EMA teacher ──────────────────────
    torch.manual_seed(args.seed)                                # identical init on all ranks
    student = build_model(args.out_dim, bool(args.norm_last_layer), args.drop_path)
    teacher = build_model(args.out_dim, bool(args.norm_last_layer), 0.0)
    if args.init == "imagenet" and resume_dir is None:
        if not Path(args.full_ckpt).exists():
            raise SystemExit(f"{args.full_ckpt} missing; download it on a login node:\n  curl -L -o {args.full_ckpt} {FULL_CKPT_URL}")
        load_official_full_checkpoint(student, teacher, args.full_ckpt)
        log.info("student and teacher initialised from %s", args.full_ckpt)
    else:
        teacher.load_state_dict(student.state_dict())
    student, teacher = student.to(device), teacher.to(device)
    for q in teacher.parameters():
        q.requires_grad = False
    dino_loss = DINOLoss(args.out_dim, n_crops, args.student_temp, args.center_momentum).to(device)
    n_params = sum(q.numel() for q in student.parameters())
    ddp = DDP(student, device_ids=[local] if device.type == "cuda" else None) if world > 1 else student

    reg, no_reg = [], []
    for name, q in student.named_parameters():
        if not q.requires_grad and name != "head.last_g":
            continue
        (no_reg if (name.endswith(".bias") or q.ndim == 1 or name == "head.last_g") else reg).append(q)
    opt = torch.optim.AdamW([{"params": reg}, {"params": no_reg, "weight_decay": 0.0}])
    last_layer = [student.head.last_v, student.head.last_g]
    s_params = [q for q in student.parameters()]
    t_params = [q for q in teacher.parameters()]

    peak_lr = args.lr * (args.batch_size * world) / 256
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    lr_sched = cosine_schedule(peak_lr, args.min_lr, total_steps, warmup_steps)
    wd_sched = cosine_schedule(args.weight_decay, args.weight_decay_end, total_steps)
    mom_sched = cosine_schedule(args.momentum_teacher, 1.0, total_steps)
    n_tw = int(args.warmup_teacher_temp_epochs * steps_per_epoch)
    tt_sched = np.concatenate([np.linspace(args.warmup_teacher_temp, args.teacher_temp, n_tw),
                               np.full(max(0, total_steps - n_tw), args.teacher_temp)])

    step, epoch = 0, 0
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["optimizer"])
        dino_loss.load_state_dict(ck["dino_loss"])
        step, epoch = ck["step"], ck["epoch"]
        log.info("Resumed from %s at step %d (epoch %d)", ckpt_path, step, epoch)
    batches_done_in_epoch = step - epoch * steps_per_epoch
    if batches_done_in_epoch >= steps_per_epoch:
        epoch, batches_done_in_epoch = epoch + 1, 0
    torch.manual_seed(args.seed + 1000 * rank + step)

    # ── config + wandb ───────────────────────────────────────────────────────
    cfg = {"args": vars(args), "n_params_student": n_params, "train_images": len(train_ds), "world_size": world,
           "global_batch_size": args.batch_size * world, "peak_lr": peak_lr, "steps_per_epoch": steps_per_epoch,
           "total_steps": total_steps, "warmup_steps": warmup_steps, "host": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
           "train_dataset": args.dataset, "init": args.init}
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("ViT-B/16 DINO init=%s dataset=%s params(student)=%.1fM out_dim=%d norm_last=%d train=%d world=%d "
                 "global_bs=%d steps/epoch=%d total_steps=%d peak_lr=%.2e warmup=%d steps", args.init, args.dataset,
                 n_params / 1e6, args.out_dim, args.norm_last_layer, len(train_ds), world, cfg["global_batch_size"],
                 steps_per_epoch, total_steps, peak_lr, warmup_steps)
        run = init_wandb(args, cfg, run_dir)
        if run is not None:
            run.summary["run_dir"] = str(run_dir)
            run.tags = tuple(run.tags or ()) + ("dino", args.dataset, f"init:{args.init}")
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(d: dict) -> None:
        if rank != 0:
            return
        metrics_f.write(json.dumps({"step": step, **d}) + "\n")
        metrics_f.flush()
        if run is not None:
            run.log({"step": step, **d}, step=step)

    def teacher_backbone_state() -> dict:
        return {"backbone": {k: v.cpu() for k, v in teacher.backbone.state_dict().items()},
                "arch": "vit_base_patch16", "embed_dim": 768, "train_dataset": args.dataset, "init": args.init,
                "step": step, "epoch": step / steps_per_epoch, "total_steps": total_steps, "out_dim": args.out_dim}

    def checkpoint() -> None:
        if rank != 0:
            return
        save_ckpt_atomic({"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": opt.state_dict(),
                          "dino_loss": dino_loss.state_dict(), "step": step, "epoch": epoch, "args": vars(args)}, ckpt_path)

    last_probe = {}

    def run_probe() -> None:
        t0 = time.time()
        ftr = encode(teacher.backbone, probe_train, idx_tr, device, world, rank, mean, std)
        fte = encode(teacher.backbone, probe_test, idx_te, device, world, rank, mean, std)
        ytr = torch.as_tensor(probe_train.labels[idx_tr]).to(device)
        yte = torch.as_tensor(probe_test.labels[idx_te]).to(device)
        m = {"val/knn_acc": knn_acc(ftr, ytr, fte, yte, n_cls=n_cls), "val/linear_acc": linear_acc(ftr, ytr, fte, yte, n_cls),
             "val/feat_std": feature_std(fte), "epoch": step / steps_per_epoch}
        last_probe.update(m)
        emit(m)
        log.info("step %d ep %.1f  probe (teacher CLS): kNN %.3f  linear %.3f  feat_std %.2f  (%.0fs)", step,
                 step / steps_per_epoch, m["val/knn_acc"], m["val/linear_acc"], m["val/feat_std"], time.time() - t0)
        if run is not None:
            run.summary.update({"final/knn_acc": m["val/knn_acc"], "final/linear_acc": m["val/linear_acc"],
                                "final/feat_std": m["val/feat_std"]})

    # ── train loop ───────────────────────────────────────────────────────────
    if args.probe_at_start and step == 0:
        run_probe()
    student.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    loss_acc = torch.zeros(1, device=device)
    n_acc = 0
    done = step >= total_steps
    last_probe_epoch = int(step / steps_per_epoch) if step else -1
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
        for crops, _ in loader:
            crops = [prep(c.to(device, non_blocking=True), mean, std) for c in crops]
            lr, wd = float(lr_sched[step]), float(wd_sched[step])
            opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = lr
            opt.param_groups[0]["weight_decay"] = wd
            log_step = (step + 1) % args.log_every == 0
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                with torch.no_grad():
                    t_out = teacher(crops[:2])
                s_out = ddp(crops)
            loss, info = dino_loss(s_out, t_out, float(tt_sched[step]), stats=log_step)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {loss.item()} at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = [q.grad for q in s_params if q.grad is not None]
            norms = torch._foreach_norm(grads)
            if args.clip_grad > 0:                              # per-parameter clipping, as official DINO
                coefs = [(args.clip_grad / (n + 1e-6)).clamp_(max=1.0) for n in norms]
                torch._foreach_mul_(grads, coefs)
            if epoch < args.freeze_last_layer:
                for q in last_layer:
                    q.grad = None
            opt.step()
            with torch.no_grad():                               # EMA teacher
                m = float(mom_sched[step])
                torch._foreach_mul_(t_params, m)
                torch._foreach_add_(t_params, [q.detach() for q in s_params], alpha=1 - m)
            step += 1
            imgs_since += args.batch_size * world
            loss_acc += loss.detach()
            n_acc += 1

            if log_step:
                if world > 1:
                    dist.all_reduce(loss_acc)
                    loss_acc /= world
                now = time.time()
                gn = torch.stack([n.float() for n in norms]).pow(2).sum().sqrt().item()
                d = {"train/loss": loss_acc.item() / n_acc, "train/lr": lr, "train/wd": wd, "train/momentum": m,
                     "train/teacher_temp": float(tt_sched[step - 1]), "train/grad_norm": gn,
                     "epoch": step / steps_per_epoch, "perf/img_per_s": imgs_since / (now - t_last),
                     "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
                d.update({f"train/{k}": v for k, v in info.items()})
                emit(d)
                log.info("step %d/%d ep %.2f loss %.4f ent %.2f marg_ent %.2f/%.2f lr %.2e wd %.3f mom %.4f %.0f img/s",
                         step, total_steps, step / steps_per_epoch, d["train/loss"], info["teacher_entropy"],
                         info["teacher_marginal_entropy"], info["max_entropy"], lr, wd, m, d["perf/img_per_s"])
                loss_acc.zero_(); n_acc = 0; imgs_since = 0; t_last = now

            if step % args.ckpt_every == 0:
                checkpoint()
            if args.probe_every and step % steps_per_epoch == 0 and (step // steps_per_epoch) % args.probe_every == 0 \
                    and step < total_steps:
                run_probe()
                last_probe_epoch = step // steps_per_epoch
                student.train()

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
        run_probe()
        if rank == 0:
            torch.save(teacher_backbone_state() | {"probe": last_probe}, run_dir / "dino_final.pt")
            log.info("final TEACHER backbone: %s", run_dir / "dino_final.pt")
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
    checkpoint()
    if rank == 0:
        (run_dir / "status.json").write_text(json.dumps(
            {"status": stop_reason, "step": step, "total_steps": total_steps, "probe": last_probe,
             "finished_at": dt.datetime.now().isoformat()}, indent=2))
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
