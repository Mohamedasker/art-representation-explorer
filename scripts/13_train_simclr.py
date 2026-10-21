#!/usr/bin/env python3
"""
Train SimCLR (ResNet-50 + projection head, NT-Xent with negatives gathered over all GPUs) on ONE dataset
(ArtBench-10 train or WikiArt train) with a named augmentation set, using DistributedDataParallel and wandb.

The augmentation set is picked with ``--aug-set`` from ``configs/aug_sets.yaml`` (baseline, color_heavy,
edge_texture, rotation, geometry_heavy, no_color, blur_heavy, crop_only; see ``--list-aug-sets``).
Default init is FROM SCRATCH (``--init imagenet_simclr`` continues from the published ImageNet SimCLR v1 weights).
The embedding used in the explorer is the global-pooled ResNet-50 output (2048-D, before the projection head).

Every run creates ``runs/<YYYYmmdd-HHMMSS>_<name>/`` with config.json, checkpoints/, simclr_final.pt, train.log,
metrics.jsonl, views/ (augmented view pairs), status.json and wandb/.  ``--resume <run dir>`` continues it.

    # single-GPU check
    python scripts/13_train_simclr.py --dataset artbench --aug-set baseline --max-steps 20 --name smoke
    # 4 GPUs
    torchrun --standalone --nproc_per_node=4 scripts/13_train_simclr.py --dataset artbench --aug-set color_heavy
    # continue after the Slurm time limit
    torchrun --standalone --nproc_per_node=4 scripts/13_train_simclr.py --resume runs/<run>

Monitors (every ``--probe-every`` epochs, on frozen encoder features, style labels of the dataset): kNN accuracy
(k=20, cosine) and linear-probe accuracy (train subset -> test subset), logged as ``val/knn_acc`` / ``val/linear_acc``.
On a JURECA compute node (no internet) wandb falls back to offline mode; sync from a login node with
    wandb sync runs/<run>/wandb/offline-run-*
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_two_view, list_aug_sets, load_aug_sets
from src.datasets.simclr_data import SplitImages
from src.models.simclr import LARS, SimCLR, SimCLRConfig, load_simclr_v1_imagenet, nt_xent
from src.training.common import (OffsetSampler, init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist,
                                 setup_logging)

log = logging.getLogger("simclr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--dataset", choices=["artbench", "wikiart"], default="artbench")
    p.add_argument("--aug-set", type=str, default="baseline", help="Named set from configs/aug_sets.yaml")
    p.add_argument("--list-aug-sets", action="store_true")
    p.add_argument("--name", type=str, default=None, help="Run name (default simclr_<dataset>_<aug-set>)")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--data-dir", type=str, default=None)
    # model
    p.add_argument("--init", choices=["scratch", "imagenet_simclr"], default="scratch")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--no-sync-bn", action="store_true", help="Per-GPU BatchNorm statistics instead of SyncBN")
    # optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size (global = x GPUs)")
    p.add_argument("--optimizer", choices=["lars", "adamw"], default="lars")
    p.add_argument("--lr", type=float, default=0.3, help="Base LR for global batch 256 (scaled linearly); LARS 0.3, AdamW ~1e-3")
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=float, default=10)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=30, help="DataLoader workers per GPU")
    # monitoring / checkpointing
    p.add_argument("--probe-every", type=int, default=10, help="kNN / linear probe every N epochs (0 = off)")
    p.add_argument("--probe-train-n", type=int, default=10240)
    p.add_argument("--probe-test-n", type=int, default=4096)
    p.add_argument("--views-every", type=int, default=50, help="Log augmented view pairs every N epochs")
    p.add_argument("--n-view-images", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10, help="Steps")
    p.add_argument("--ckpt-every", type=int, default=100, help="Steps")
    p.add_argument("--keep-ckpts", type=int, default=1)
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


class TwoViewDataset(Dataset):
    def __init__(self, base: SplitImages, two_view) -> None:
        self.base, self.tv = base, two_view

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        v1, v2 = self.tv(self.base.load(i))
        return v1, v2


def _worker_init(_):
    torch.set_num_threads(1)


# ── monitors ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode(model, ds, idx, device, args, world, rank):
    """Frozen-encoder features of ds[idx] (idx length divisible by world): each rank encodes a slice, all_gather."""
    sub = Subset(ds, idx[rank::world].tolist())
    loader = DataLoader(sub, batch_size=128, shuffle=False, num_workers=min(args.num_workers, 12), pin_memory=True,
                        worker_init_fn=_worker_init)
    model.eval()
    feats = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            feats.append(model.features(x).float())
    model.train()
    f = torch.cat(feats)
    if world > 1:
        out = [torch.zeros_like(f) for _ in range(world)]
        dist.all_gather(out, f)
        f = torch.stack(out, 1).reshape(-1, f.shape[1])      # restore the original order of idx (idx[r::W] interleaving)
    return f


def knn_acc(ftr, ytr, fte, yte, k=20, T=0.07, n_cls=None):
    ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
    n_cls = n_cls or int(ytr.max()) + 1
    correct = 0
    for i in range(0, fte.shape[0], 1024):
        sim = fte[i:i + 1024] @ ftr.t()
        w, nb = sim.topk(k, dim=1)
        votes = torch.zeros(sim.shape[0], n_cls, device=ftr.device).scatter_add_(1, ytr[nb], (w / T).exp())
        correct += (votes.argmax(1) == yte[i:i + 1024]).sum().item()
    return correct / fte.shape[0]


def linear_acc(ftr, ytr, fte, yte, n_cls, steps=100, wd=1e-4):
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp_min(1e-6)
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    clf = nn.Linear(a.shape[1], n_cls).to(a.device)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(a), ytr) + wd * clf.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (clf(b).argmax(1) == yte).float().mean().item()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.list_aug_sets:
        for n, s in load_aug_sets().items():
            print(f"{n:15s} {s['description']}")
        return
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
        if args.aug_set not in list_aug_sets():
            raise SystemExit(f"unknown --aug-set {args.aug_set!r}; choose from {list_aug_sets()}")
        args.name = args.name or f"simclr_{args.dataset}_{args.aug_set}"
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
        (run_dir / "views").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── data ─────────────────────────────────────────────────────────────────
    S = args.crop_size
    two_view = build_two_view(args.aug_set, crop_size=S)
    base_train = SplitImages(args.dataset, "train", root=args.data_dir)
    train_ds = TwoViewDataset(base_train, two_view)
    eval_tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(S), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    probe_train = SplitImages(args.dataset, "train", eval_tf, root=args.data_dir)
    probe_test = SplitImages(args.dataset, "test", eval_tf, root=args.data_dir)
    g = np.random.default_rng(1234)                           # fixed probe subsets, independent of the run seed
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
    args.warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                     prefetch_factor=4 if args.num_workers > 0 else None, worker_init_fn=_worker_init)

    # ── model / optimiser ────────────────────────────────────────────────────
    mcfg = SimCLRConfig(proj_dim=args.proj_dim, img_size=S)
    model = SimCLR(mcfg)
    if args.init == "imagenet_simclr" and not args.resume:
        model.encoder.load_state_dict(load_simclr_v1_imagenet().state_dict())
        log.info("encoder initialised from ImageNet SimCLR v1")
    model = model.to(device)
    if world > 1 and not args.no_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    peak_lr = args.lr * (args.batch_size * world) / 256
    if args.optimizer == "lars":
        opt = LARS(model.parameters(), lr=peak_lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=args.weight_decay)

    step, epoch = 0, 0
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, epoch = ck["step"], ck["epoch"]
        log.info("Resumed from %s at step %d (epoch %d)", ckpt_path, step, epoch)
    batches_done_in_epoch = step - epoch * steps_per_epoch
    if batches_done_in_epoch >= steps_per_epoch:
        epoch, batches_done_in_epoch = epoch + 1, 0

    # ── config + wandb ───────────────────────────────────────────────────────
    aug_cfg = load_aug_sets()[args.aug_set]
    cfg = {
        "args": vars(args), "dataset": args.dataset, "aug_set": args.aug_set, "aug_description": aug_cfg["description"],
        "aug_intent": aug_cfg["intent"], "aug_ops": aug_cfg["ops"], "init": args.init, "model": mcfg.to_dict(),
        "n_params": n_params, "train_images": len(train_ds), "world_size": world,
        "global_batch_size": args.batch_size * world, "peak_lr": peak_lr, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
    }
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("aug set %s: %s", args.aug_set, aug_cfg["description"])
        log.info("%r", two_view)
        log.info("params=%.1fM train=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d peak_lr=%.3f",
                 n_params / 1e6, len(train_ds), world, cfg["global_batch_size"], steps_per_epoch, total_steps, peak_lr)
        run = init_wandb(args, cfg, run_dir)
        if run is not None:
            run.summary["run_dir"] = str(run_dir)
            run.summary["aug_set"] = args.aug_set
            run.tags = tuple(run.tags or ()) + ("simclr", args.dataset, f"aug:{args.aug_set}", f"init:{args.init}")
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(d: dict) -> None:
        if rank != 0:
            return
        metrics_f.write(json.dumps({"step": step, **d}) + "\n")
        metrics_f.flush()
        if run is not None:
            run.log({"step": step, **d}, step=step)

    def checkpoint(name="last.pt", numbered=False) -> None:
        if rank != 0:
            return
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "epoch": epoch,
                 "model_cfg": mcfg.to_dict(), "args": vars(args)}
        save_ckpt_atomic(state, run_dir / "checkpoints" / name)
        if numbered:
            save_ckpt_atomic({k: state[k] for k in ("model", "step", "epoch", "model_cfg")},
                             run_dir / "checkpoints" / f"step_{step:07d}.pt")
            for p in sorted((run_dir / "checkpoints").glob("step_*.pt"))[:-args.keep_ckpts]:
                p.unlink()

    def log_views() -> None:
        if rank != 0:
            return
        n = args.n_view_images
        gen = np.random.default_rng(7)
        ids = gen.choice(len(base_train), n, replace=False)
        rows = []
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(IMAGENET_STD)[:, None, None]
        orig_tf = transforms.Compose([transforms.Resize((S, S), antialias=True), transforms.ToTensor()])
        for i in ids:
            img = base_train.load(int(i))
            v1, v2 = two_view(img)
            rows += [orig_tf(img), (v1 * std + mean).clamp(0, 1), (v2 * std + mean).clamp(0, 1)]
        grid = make_grid(torch.stack(rows), nrow=3 * 4, padding=2)      # 4 (original, view1, view2) triples per row
        save_image(grid, run_dir / "views" / f"step_{step:07d}.png")
        if run is not None:
            import wandb
            run.log({"step": step, "samples/aug_views": wandb.Image(
                grid, caption=f"[{args.aug_set}] triples: original | view 1 | view 2 (step {step})")}, step=step)

    def run_probe() -> None:
        t0 = time.time()
        ftr = encode(model, probe_train, idx_tr, device, args, world, rank)
        fte = encode(model, probe_test, idx_te, device, args, world, rank)
        if rank == 0:
            ytr = torch.as_tensor(probe_train.labels[idx_tr]).to(device)
            yte = torch.as_tensor(probe_test.labels[idx_te]).to(device)
            m = {"val/knn_acc": knn_acc(ftr, ytr, fte, yte, n_cls=n_cls),
                 "val/linear_acc": linear_acc(ftr, ytr, fte, yte, n_cls),
                 "val/feat_std": ftr.std(0).mean().item()}
            emit(m)
            log.info("step %d ep %.1f  probe: kNN %.3f  linear %.3f  (%.0fs)", step, step / steps_per_epoch,
                     m["val/knn_acc"], m["val/linear_acc"], time.time() - t0)
            if run is not None:
                run.summary.update({"final/knn_acc": m["val/knn_acc"], "final/linear_acc": m["val/linear_acc"]})
        if world > 1:
            dist.barrier()

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    acc = torch.zeros(3, device=device)
    n_acc = 0
    if step == 0:
        log_views()
    done = step >= total_steps
    last_probe_epoch = epoch if step else -1
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            v2 = v2.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            lr = lr_at(step, peak_lr, args, total_steps)
            for gr in opt.param_groups:
                gr["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                z = ddp(torch.cat([v1, v2]))            # one forward over both views (BN sees both)
            z1, z2 = z.chunk(2)
            loss, pos_acc = nt_xent(z1, z2, args.temperature, gather=world > 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
            opt.step()
            step += 1
            imgs_since += v1.shape[0] * world
            acc += torch.stack([loss.detach(), pos_acc, torch.tensor(0.0, device=device)])
            n_acc += 1

            if step % args.log_every == 0:
                a = reduce_mean(acc / n_acc, world)
                now = time.time()
                emit({"train/loss": a[0].item(), "train/pos_acc": a[1].item(), "lr": lr,
                      "epoch": step / steps_per_epoch, "perf/img_per_s": imgs_since / (now - t_last),
                      **({"train/grad_norm": float(gnorm)} if gnorm is not None else {})})
                log.info("step %d/%d ep %.2f loss %.4f pos_acc %.3f lr %.3f %.0f img/s", step, total_steps,
                         step / steps_per_epoch, a[0].item(), a[1].item(), lr, imgs_since / (now - t_last))
                acc.zero_(); n_acc = 0; imgs_since = 0; t_last = now
                if not torch.isfinite(a[0]):
                    raise RuntimeError("loss is not finite")

            if step % args.ckpt_every == 0:
                checkpoint(numbered=False)
            # epoch-based monitors (an epoch boundary = step multiple of steps_per_epoch)
            if step % steps_per_epoch == 0:
                ep = step // steps_per_epoch
                if args.probe_every and ep % args.probe_every == 0:
                    run_probe()
                    last_probe_epoch = ep
                if args.views_every and ep % args.views_every == 0:
                    log_views()
                if ep % 10 == 0:
                    checkpoint(numbered=True)

            out_of_time = False
            if args.time_limit_min is not None and step % 20 == 0:
                flag = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit_min), device=device)
                if world > 1:
                    dist.broadcast(flag, src=0)
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
        if args.probe_every and step // steps_per_epoch != last_probe_epoch or args.max_steps is not None:
            run_probe()
        if rank == 0:
            torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "step": step,
                        "img_size": S, "info": {"dataset": args.dataset, "aug_set": args.aug_set,
                                                "epochs": args.epochs, "init": args.init,
                                                "run": run_dir.name, "batch_size": args.batch_size * world}},
                       run_dir / "simclr_final.pt")
            log.info("final weights: %s", run_dir / "simclr_final.pt")
    else:
        log.info("time limit reached at step %d -> checkpointing; resume with: --resume %s", step, run_dir)
    checkpoint(numbered=False)
    if rank == 0:
        last = {}
        mf = run_dir / "metrics.jsonl"
        for line in mf.read_text().splitlines()[::-1]:
            d = json.loads(line)
            if "val/knn_acc" in d:
                last = {"knn_acc": d["val/knn_acc"], "linear_acc": d["val/linear_acc"], "probe_step": d["step"]}
                break
        (run_dir / "status.json").write_text(json.dumps(
            {"status": stop_reason, "step": step, "total_steps": total_steps, "epochs": step / steps_per_epoch,
             "finished_at": dt.datetime.now().isoformat(), **last}, indent=2))
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
