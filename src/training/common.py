"""Helpers shared by the training scripts (07 autoencoder, 10 supervised CNN):
DDP setup, wandb init with offline fallback, resumable sampler, LR schedule, atomic checkpoints, logging."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger("train")


# ── distributed helpers ───────────────────────────────────────────────────────

def setup_dist():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, local, device


def reduce_mean(t: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / world
    return t


# ── wandb ─────────────────────────────────────────────────────────────────────

def internet_ok(host: str = "api.wandb.ai", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def init_wandb(args, cfg: dict, run_dir: Path):
    """Returns a wandb run (rank 0 only) or None."""
    mode = args.wandb
    if mode == "auto":
        mode = os.environ.get("WANDB_MODE") or ("online" if internet_ok() else "offline")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; continuing without it (pip install wandb)")
        return None

    os.environ["WANDB_MODE"] = mode
    id_file = run_dir / "wandb_run_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    id_file.write_text(run_id)
    try:
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_dir.name,
            id=run_id, resume="allow", dir=str(run_dir), config=cfg, mode=mode,
            settings=wandb.Settings(init_timeout=60),
        )
    except Exception as e:  # online init failed (no network / bad key) -> offline fallback
        log.warning("wandb.init(mode=%s) failed (%s); falling back to offline", mode, e)
        os.environ["WANDB_MODE"] = mode = "offline"
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_dir.name,
            id=run_id, resume="allow", dir=str(run_dir), config=cfg, mode="offline",
        )
    wandb.define_metric("step")
    for m in ("train/*", "val/*", "lr", "perf/*", "samples/*"):
        wandb.define_metric(m, step_metric="step")
    (run_dir / "wandb_info.json").write_text(json.dumps({
        "mode": mode, "id": run_id, "name": run.name, "url": getattr(run, "url", None),
        "project": args.wandb_project,
        "sync_cmd": f"wandb sync {run_dir}/wandb/offline-run-*" if mode == "offline" else None,
    }, indent=2))
    log.info("wandb mode=%s url=%s", mode, getattr(run, "url", None))
    return run


class OffsetSampler(DistributedSampler):
    """DistributedSampler that can skip the first ``start`` indices (mid-epoch resume)."""

    start = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start, None)

    def __len__(self):
        return super().__len__() - self.start


def lr_at(step: int, peak: float, args, total_steps: int) -> float:
    if step < args.warmup_steps:
        return peak * (step + 1) / args.warmup_steps
    t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    t = min(1.0, t)
    return peak * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def save_ckpt_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def setup_logging(run_dir: Path | None, rank: int):
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format=f"%(asctime)s [r{rank}] %(message)s", datefmt="%H:%M:%S", force=True)
    if run_dir is not None and rank == 0:
        fh = logging.FileHandler(run_dir / "train.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(fh)
