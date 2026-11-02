#!/usr/bin/env python3
"""
Gently fine-tune the image tower of OpenAI CLIP on ONE art dataset (ArtBench-10 or WikiArt), without forgetting.

Contrastive loss between the images and captions built from the dataset's labels (see src/models/clip_ft.py:
"a landscape painting by Vincent Van Gogh, in the post impressionism style"; ArtBench has no genre).  Captions shared
by several images of a batch are not treated as negatives of each other.  The text tower and the logit scale are
frozen: the caption embeddings are a fixed table (CLIP's original text space) that the image tower is pulled towards.
Low peak LR with layer-wise decay and a few epochs; the weights of the original image tower are kept implicitly (they
are re-created by open_clip), so ``scripts/02_extract_features.py --model clip_ft --clip-alpha a`` can interpolate
(WiSE-FT).  Trained on the train split only; the test split is only used for monitoring.

Monitoring (step 0 = original CLIP, then every --val-every steps and at the end, on the test split of this dataset):
  val/caption_acc  image -> its caption among all unique training captions
  val/zs_style     zero-shot style accuracy ("a painting in the X style")
  generic/cifar100, generic/cifar10   zero-shot top-1 (rank 0, data/generic) -- the "did it forget?" probe

    runs/<YYYYmmdd-HHMMSS>_<name>/  config.json, captions.json, checkpoints/, clip_ft_final.pt,
                                    train.log, metrics.jsonl, status.json, wandb/

Usage
-----
    torchrun --standalone --nproc_per_node=4 scripts/28_train_clip_ft.py --dataset artbench --name clip_ft_artbench
    torchrun --standalone --nproc_per_node=4 scripts/28_train_clip_ft.py --dataset wikiart  --name clip_ft_wikiart
    torchrun --standalone --nproc_per_node=4 scripts/28_train_clip_ft.py --resume runs/<run>
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
import torch.distributed.nn.functional as dist_F
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.datasets import default_data_dir
from src.models import clip_ft as C
from src.training.common import OffsetSampler, init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("clip_ft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--dataset", choices=["artbench", "wikiart"], default="artbench")
    p.add_argument("--name", type=str, default=None, help="default clip_ft_<dataset>")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--data-dir", type=str, default=None, help="default data/<dataset>")
    p.add_argument("--wikiart-dir", type=str, default=default_data_dir("wikiart"))
    p.add_argument("--generic-dir", type=str, default="data/generic", help="CIFAR test sets for the forgetting probe")
    # model
    p.add_argument("--arch", type=str, default=C.DEFAULT_ARCH)
    p.add_argument("--pretrained", type=str, default=C.DEFAULT_PRETRAINED)
    p.add_argument("--no-style-in-caption", action="store_true",
                   help="leave the style out of the captions (style then is only used for evaluation)")
    # optimisation
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size (global batch = x world size)")
    p.add_argument("--lr", type=float, default=1e-5, help="Peak LR (not scaled with the batch size)")
    p.add_argument("--layer-decay", type=float, default=0.8, help="Layer-wise LR decay (1 = off)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=16, help="DataLoader workers per GPU")
    # validation / logging / checkpointing
    p.add_argument("--val-max-samples", type=int, default=4096, help="Fixed random subset of the test split (0 = all)")
    p.add_argument("--final-max-samples", type=int, default=0, help="Test images in the final evaluation (0 = all)")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--generic-max-images", type=int, default=0, help="Subset of the CIFAR test sets (0 = all 10k)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="Checkpoint and exit cleanly after this many minutes (set below the Slurm limit)")
    # wandb
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


def contrastive_loss(img_local, cid_local, img_all, cid_all, text_table, scale, rank, n_local):
    """Symmetric InfoNCE between the global batch of images and their (frozen) caption embeddings.

    Row i of the logits is one local image against all captions of the global batch (and the transposed for
    caption -> image).  Pairs with the same caption id other than the true pair are masked out (not negatives).
    Returns (loss, number of images whose own caption is the arg-max)."""
    txt_all = text_table[cid_all]                                     # (N, D)
    txt_local = text_table[cid_local]
    tgt = torch.arange(n_local, device=img_local.device) + rank * n_local
    same_i2t = cid_local[:, None] == cid_all[None, :]
    same_i2t[torch.arange(n_local), tgt] = False
    l_i2t = (scale * img_local @ txt_all.T).masked_fill(same_i2t, float("-inf"))
    same_t2i = same_i2t                                               # identical relation, rows are the local captions
    l_t2i = (scale * txt_local @ img_all.T).masked_fill(same_t2i, float("-inf"))
    loss = 0.5 * (F.cross_entropy(l_i2t, tgt) + F.cross_entropy(l_t2i, tgt))
    return loss, (l_i2t.argmax(1) == tgt).sum()


@torch.no_grad()
def evaluate(model, val_loader, device, args, text_table, style_text, rank) -> dict:
    """Held-out monitoring on the test subset (every rank handles a shard; features are gathered on rank 0 only)."""
    model.eval()
    n_cap = torch.zeros(2, device=device)                              # [caption correct, style correct]
    n = torch.zeros(1, device=device)
    for x, cid, sid in val_loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            f = F.normalize(model.encode_image(x.to(device, non_blocking=True)).float(), dim=-1)
        n_cap[0] += (( f @ text_table.T).argmax(1).cpu() == cid).sum().to(device)
        n_cap[1] += ((f @ style_text.T).argmax(1).cpu() == sid).sum().to(device)
        n += len(cid)
    if dist.is_initialized():
        dist.all_reduce(n_cap), dist.all_reduce(n)
    model.train()
    return {"val/caption_acc": (n_cap[0] / n).item(), "val/zs_style": (n_cap[1] / n).item()}


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
        args.name = args.name or f"clip_ft_{args.dataset}"
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── model, caption table ─────────────────────────────────────────────────
    model, tokenizer, _ = C.load_clip(args.arch, args.pretrained, device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.visual.parameters():
        p.requires_grad_(True)
    scale = float(model.logit_scale.exp().clamp(max=100))
    S = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
    mean, std = model.visual.preprocess_cfg["mean"], model.visual.preprocess_cfg["std"]

    data_dir = args.data_dir or default_data_dir(args.dataset)
    use_style = not args.no_style_in_caption
    tr_paths, tr_styles, tr_caps = C.dataset_items(args.dataset, data_dir, "train", args.wikiart_dir, use_style)
    te_paths, te_styles, te_caps = C.dataset_items(args.dataset, data_dir, "test", args.wikiart_dir, use_style)
    captions = sorted(set(tr_caps) | set(te_caps))
    cap_id = {c: i for i, c in enumerate(captions)}
    styles = sorted(set(tr_styles) | set(te_styles))
    style_id = {s: i for i, s in enumerate(styles)}
    text_table = C.encode_texts(model, tokenizer, captions, device)            # (n_captions, D), frozen
    style_text = C.encode_texts(model, tokenizer, [C.style_prompt(s) for s in styles], device)
    generic_text: dict = {}

    bicubic = transforms.InterpolationMode.BICUBIC
    norm = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.5, 1.0), interpolation=bicubic, antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize(S, interpolation=bicubic, antialias=True),
                                 transforms.CenterCrop(S), transforms.ToTensor(), norm])
    train_ds = C.CaptionedImages(tr_paths, [cap_id[c] for c in tr_caps], [style_id[s] for s in tr_styles], train_tf, S)
    val_full = C.CaptionedImages(te_paths, [cap_id[c] for c in te_caps], [style_id[s] for s in te_styles], val_tf, S)
    g = torch.Generator().manual_seed(1234)
    n_val = len(val_full) if args.val_max_samples <= 0 else min(args.val_max_samples, len(val_full))
    val_ds = Subset(val_full, sorted(torch.randperm(len(val_full), generator=g)[:n_val].tolist()))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=min(args.num_workers, 8), pin_memory=True,
                            sampler=DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False))

    sampler = OffsetSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    steps_per_epoch = len(sampler) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    train_loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                           prefetch_factor=4 if args.num_workers > 0 else None)

    # ── optimiser ────────────────────────────────────────────────────────────
    visual = model.visual
    ddp_visual = DDP(visual, device_ids=[local] if device.type == "cuda" else None) if world > 1 else visual
    scales = C.layer_scales(visual, args.layer_decay)
    groups: dict = {}
    for name, p in visual.named_parameters():
        wd = 0.0 if (p.ndim <= 1 or name in ("class_embedding", "positional_embedding")) else args.weight_decay
        groups.setdefault((scales[name], wd), []).append(p)
    opt = torch.optim.AdamW([{"params": ps, "weight_decay": wd, "lr_scale": s} for (s, wd), ps in groups.items()],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    step, epoch = 0, 0
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        visual.load_state_dict(ck["visual"])
        opt.load_state_dict(ck["optimizer"])
        step, epoch = ck["step"], ck["epoch"]
        log.info("Resumed from %s at step %d (epoch %d)", ckpt_path, step, epoch)
    batches_done_in_epoch = step - epoch * steps_per_epoch
    if batches_done_in_epoch >= steps_per_epoch:
        epoch, batches_done_in_epoch = epoch + 1, 0

    # ── config + wandb ───────────────────────────────────────────────────────
    cfg = {"args": vars(args), "n_train": len(train_ds), "n_test": len(val_full), "n_captions": len(captions),
           "world_size": world, "global_batch_size": args.batch_size * world, "logit_scale": scale,
           "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "host": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}
    run = None
    if rank == 0:
        if resume_dir is None or not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            (run_dir / "captions.json").write_text(json.dumps(
                {"captions": captions, "styles": styles, "example": dict(zip(tr_paths[:5], tr_caps[:5]))}, indent=1))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("%s/%s on %s: train=%d test=%d captions=%d (e.g. %r) world=%d global_bs=%d steps/epoch=%d total=%d",
                 args.arch, args.pretrained, args.dataset, len(train_ds), len(val_full), len(captions), tr_caps[0], world,
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

    def run_val(tag: str = "") -> dict:
        m = evaluate(model, val_loader, device, args, text_table, style_text, rank)
        if rank == 0:
            gen = C.generic_zero_shot(model, tokenizer, val_tf, args.generic_dir, device, num_workers=8,
                                      max_images=args.generic_max_images or None, text_cache=generic_text)
            m.update({f"generic/{k}": v for k, v in gen.items()})
            emit(m)
            log.info("step %d%s  " % (step, tag) + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        if dist.is_initialized():
            dist.barrier()
        return m

    def checkpoint() -> None:
        if rank == 0:
            save_ckpt_atomic({"visual": visual.state_dict(), "optimizer": opt.state_dict(), "step": step,
                              "epoch": epoch, "args": vars(args)}, run_dir / "checkpoints" / "last.pt")

    def final_state() -> dict:
        return {"visual": {k: v.detach().cpu() for k, v in visual.state_dict().items()}, "arch": args.arch,
                "pretrained": args.pretrained, "dataset": args.dataset, "step": step,
                "caption_style": use_style, "run": run_dir.name}

    # ── train loop ───────────────────────────────────────────────────────────
    if step == 0:
        run_val(" (original CLIP)")
    visual.train()
    stop_reason = "finished"
    t_last, imgs_since = time.time(), 0
    acc = torch.zeros(3, device=device)               # loss*n, correct, n
    done = step >= total_steps
    while not done:
        sampler.set_epoch(epoch)
        sampler.start = batches_done_in_epoch * args.batch_size
        loader = DataLoader(train_ds, sampler=sampler, **train_loader_kw)
        for x, cid, _sid in loader:
            x, cid = x.to(device, non_blocking=True), cid.to(device, non_blocking=True)
            lr = lr_at(step, args.lr, args, total_steps)
            for gr in opt.param_groups:
                gr["lr"] = lr * gr["lr_scale"]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                f = ddp_visual(x)
            img = F.normalize(f.float(), dim=-1)
            if world > 1:
                img_all = torch.cat(dist_F.all_gather(img), 0)                  # autograd-aware gather
                cid_all = torch.cat([t for t in _gather_ids(cid, world)], 0)
            else:
                img_all, cid_all = img, cid
            loss, correct = contrastive_loss(img, cid, img_all, cid_all, text_table, scale, rank, len(x))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(visual.parameters(), args.grad_clip) if args.grad_clip > 0 else None
            opt.step()
            step += 1
            imgs_since += x.shape[0] * world
            acc += torch.stack([loss.detach() * len(x), correct.float(), torch.tensor(float(len(x)), device=device)])

            if step % args.log_every == 0:
                if world > 1:
                    dist.all_reduce(acc)
                now = time.time()
                m = {"train/loss": (acc[0] / acc[2]).item(), "train/batch_acc": (acc[1] / acc[2]).item(), "lr": lr,
                     "epoch": step / steps_per_epoch, "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                     "perf/img_per_s": imgs_since / (now - t_last)}
                emit(m)
                log.info("step %d/%d ep %.2f loss %.4f batch_acc %.3f lr %.2e %.0f img/s", step, total_steps,
                         step / steps_per_epoch, m["train/loss"], m["train/batch_acc"], lr, m["perf/img_per_s"])
                acc.zero_(); imgs_since = 0; t_last = now

            if step % args.val_every == 0:
                run_val()
            if step % args.ckpt_every == 0:
                checkpoint()

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
        final_ds = val_full if args.final_max_samples <= 0 else Subset(val_full, list(range(min(args.final_max_samples, len(val_full)))))
        full_loader = DataLoader(final_ds, batch_size=args.batch_size, num_workers=min(args.num_workers, 8),
                                 sampler=DistributedSampler(final_ds, num_replicas=world, rank=rank, shuffle=False))
        m = evaluate(model, full_loader, device, args, text_table, style_text, rank)
        gen = C.generic_zero_shot(model, tokenizer, val_tf, args.generic_dir, device, num_workers=8,
                                  text_cache=generic_text) if rank == 0 else {}
        m = {k.replace("val/", "val_full/"): v for k, v in m.items()} | {f"generic_full/{k}": v for k, v in gen.items()}
        emit(m)
        log.info("FINAL: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        if rank == 0:
            torch.save(final_state(), run_dir / "clip_ft_final.pt")
            log.info("final weights: %s", run_dir / "clip_ft_final.pt")
            if run is not None:
                run.summary.update(m)
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


def _gather_ids(cid: torch.Tensor, world: int) -> list[torch.Tensor]:
    out = [torch.empty_like(cid) for _ in range(world)]
    dist.all_gather(out, cid)
    return out


if __name__ == "__main__":
    main()
