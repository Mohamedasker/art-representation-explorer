#!/usr/bin/env python3
"""
iGPT (Chen et al. 2020, "Generative Pretraining from Pixels"): a decoder-only transformer trained to predict the next PIXEL of a
low-resolution painting in raster order.  Pixels are tokens of a global 512-colour palette (9-bit k-means, scripts/26_prepare_igpt.py),
32 x 32 px = 1024 tokens per image.  Trained from scratch on the TRAIN splits of ArtBench-10 and WikiArt jointly (+ horizontally flipped
copies), unconditional (the two prefix tokens of ``ImageGPT`` are always "no label" and act as a start token), cross-entropy
next-token prediction, DistributedDataParallel + Weights & Biases.  Default size = iGPT-S: 24 layers, width 512, 8 heads, 75M parameters.

Validation = both TEST splits, in bits per token (one pixel = one palette index), bits per dimension (/3: three colour channels) and
perplexity -- of the colour-quantised 32 px images, not of 8-bit pixels.  Sample grids go to ``runs/<run>/samples/`` and wandb:
unconditional samples (full softmax and top-k), and completions of the LOWER half of real held-out paintings from the upper half
(iGPT's classic demo).  Writes ``ar_final.pt`` (last weights) and ``checkpoints/best.pt`` (lowest validation loss).

    torchrun --standalone --nproc_per_node=4 scripts/27_train_igpt.py --epochs 40
    python scripts/27_train_igpt.py --max-steps 10 --n-layer 2 --d-model 128 --n-head 2 --name smoke_igpt
    torchrun ... scripts/27_train_igpt.py --resume runs/<stamp>_ar_igpt_artbench_wikiart
Layer choice: scripts/24_probe_ar_layers.py; embeddings: scripts/02_extract_features.py --model ar_igpt.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

from src.models.ar_data import DATASETS, STYLE_NAMES, pixel_file, token_file
from src.models.igpt_data import name_of, palette_file, quantize, render, shrink
from src.models.image_gpt import GPTConfig, ImageGPT
from src.training.common import init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("igpt")
LN2 = math.log(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--res", type=int, default=32, help="image side in pixels (tokens = res^2)")
    p.add_argument("--name", type=str, default="ar_igpt_artbench_wikiart")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--cache-dir", type=str, default="data/ar_tokens")
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40, help="passes over train + flipped train (2 x 123k sequences)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=8000, help="sample grids every N steps (and at the end)")
    p.add_argument("--top-k", type=int, default=50, help="top-k of the second unconditional grid")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


def load_tokens(tok_name: str, split: str, cache_dir: str, device, flips: bool) -> torch.Tensor:
    toks = []
    for d in DATASETS:
        for flip in ((False, True) if flips else (False,)):
            toks.append(torch.from_numpy(np.load(token_file(tok_name, d, split, flip, cache_dir)).astype(np.int16)))
    ds_ids = [torch.full((len(t),), i // (2 if flips else 1), dtype=torch.long) for i, t in enumerate(toks)]
    return torch.cat(toks).to(device), torch.cat(ds_ids).to(device)


@torch.no_grad()
def validate(model, val, device, args, world, rank):
    toks, ds = val
    model.eval()
    sums = torch.zeros(2, 2, device=device)              # [dataset][sum loss, n]
    nd = torch.full((256,), model.null_ds, device=device)
    ns = torch.full((256,), model.null_style, device=device)
    for i in range(rank * 256, len(toks), 256 * world):
        t = toks[i: i + 256].long()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            _, logits = model(t, nd[: len(t)], ns[: len(t)], return_logits=True)
        l = F.cross_entropy(logits.float().transpose(1, 2), t, reduction="none").mean(1)
        for di in range(2):
            m = ds[i: i + 256] == di
            sums[di, 0] += l[m].sum()
            sums[di, 1] += m.sum()
    if world > 1:
        dist.all_reduce(sums)
    tot = sums.sum(0)
    out = {"val/loss": (tot[0] / tot[1]).item()}
    out["val/bits_per_token"] = out["val/loss"] / LN2
    out["val/bits_per_dim"] = out["val/bits_per_token"] / 3
    out["val/perplexity"] = math.exp(out["val/loss"])
    for di, d in enumerate(DATASETS):
        out[f"val/{d}_bits_per_dim"] = (sums[di, 0] / sums[di, 1]).item() / LN2 / 3
    model.train()
    return out


@torch.no_grad()
def make_sample_grids(model, palette, val_imgs, args, device) -> dict:
    """val_imgs: tokens (n, res^2) of real test images."""
    res, up = args.res, 3
    big = lambda t: F.interpolate(t, scale_factor=up, mode="nearest")
    g = torch.Generator(device=device).manual_seed(0)
    grids = {}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
        t = model.generate(32, generator=g)
        grids["unconditional"] = make_grid(big(render(t, palette, res)).cpu(), nrow=8, padding=2)
        t = model.generate(32, top_k=args.top_k, generator=g)
        grids["unconditional_topk"] = make_grid(big(render(t, palette, res)).cpu(), nrow=8, padding=2)
        n = len(val_imgs)
        half = res * res // 2
        comp = [model.generate(n, prompt=val_imgs[:, :half], generator=g) for _ in range(3)]
        shown = render(val_imgs, palette, res).clone()
        cond = shown.clone()
        cond.view(n, 3, res * res)[:, :, half:] = 0.5      # lower half greyed out
        rows = [shown, cond] + [render(c, palette, res) for c in comp]
        grids["completions"] = make_grid(big(torch.cat(rows)).cpu(), nrow=n, padding=2)
    return grids


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
        (run_dir / "samples").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    tok_name = name_of(args.res)
    palette = torch.from_numpy(np.load(palette_file(args.res, args.cache_dir))).to(device)
    train_toks, _ = load_tokens(tok_name, "train", args.cache_dir, device, flips=True)
    val = load_tokens(tok_name, "test", args.cache_dir, device, flips=False)
    n_train = len(train_toks)
    steps_per_epoch = n_train // (args.batch_size * world)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    mcfg = GPTConfig(vocab_size=palette.shape[0], n_tokens=args.res ** 2, n_layer=args.n_layer, n_head=args.n_head,
                     d_model=args.d_model, dropout=args.dropout)
    model = ImageGPT(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    opt = torch.optim.AdamW(model.param_groups(args.weight_decay), lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    step, best_val = 0, float("inf")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck["best_val"]
        log.info("Resumed from %s at step %d", ckpt_path, step)

    info = {"kind": "igpt", "tokenizer_name": tok_name, "tokenizer": None, "res": args.res,
            "palette": palette.cpu().tolist(), "train_datasets": list(DATASETS), "n_tokens": args.res ** 2,
            "codebook_size": int(palette.shape[0]),
            "embedding": "mean over the res^2 pixel positions of one intermediate block (layer picked by scripts/24_probe_ar_layers.py)"}
    cfg = {"args": vars(args), "model": mcfg.to_dict(), "info": {k: v for k, v in info.items() if k != "palette"},
           "n_params": n_params, "train_sequences": n_train, "val_sequences": len(val[0]), "world_size": world,
           "global_batch_size": args.batch_size * world, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
           "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}
    run = None
    if rank == 0:
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        else:
            with open(run_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": dt.datetime.now().isoformat(),
                                    "slurm_job_id": cfg["slurm_job_id"], "host": cfg["host"]}) + "\n")
        log.info("run dir: %s", run_dir)
        log.info("params=%.1fM train sequences=%d val=%d world=%d global_bs=%d steps/epoch=%d total_steps=%d",
                 n_params / 1e6, n_train, len(val[0]), world, cfg["global_batch_size"], steps_per_epoch, total_steps)
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

    # 8 fixed held-out paintings (4 per dataset) for the completion demo
    val_imgs = None
    if rank == 0:
        n_ab = (val[1] == 0).sum().item()
        idx = list(range(0, n_ab, n_ab // 4))[:4] + list(range(n_ab, len(val[0]), (len(val[0]) - n_ab) // 4))[:4]
        val_imgs = val[0][idx].long()

    def log_samples() -> None:
        if rank != 0:
            return
        t0 = time.time()
        grids = make_sample_grids(model, palette, val_imgs, args, device)
        model.train()
        caps = {"unconditional": "unconditional samples, temperature 1, full softmax",
                "unconditional_topk": f"unconditional samples, temperature 1, top-k {args.top_k}",
                "completions": "rows: real held-out painting / its upper half (grey = to be completed) / 3 sampled completions "
                               "of the lower half (4 ArtBench + 4 WikiArt test paintings)"}
        wb = {}
        for k, g in grids.items():
            save_image(g, run_dir / "samples" / f"step_{step:07d}_{k}.png")
            if run is not None:
                import wandb
                wb[f"samples/{k}"] = wandb.Image(g, caption=f"step {step}: {caps[k]}")
        if run is not None and wb:
            run.log({"step": step, **wb}, step=step)
        log.info("step %d: sample grids written (%.0fs)", step, time.time() - t0)

    def run_val() -> None:
        nonlocal best_val
        m = validate(model, val, device, args, world, rank)
        emit(m)
        log.info("step %d  val %.4f bits/dim (%.3f bits/token, ppl %.1f)  [artbench %.4f, wikiart %.4f bits/dim]", step,
                 m["val/bits_per_dim"], m["val/bits_per_token"], m["val/perplexity"], m["val/artbench_bits_per_dim"],
                 m["val/wikiart_bits_per_dim"])
        if m["val/loss"] < best_val:
            best_val = m["val/loss"]
            if rank == 0:
                save_ckpt_atomic({**weights(), "val": m}, run_dir / "checkpoints" / "best.pt")

    model.train()
    stop_reason = "finished"
    t_last, seqs_since = time.time(), 0
    acc, n_acc = torch.zeros(1, device=device), 0
    if step == 0:
        run_val()
    perm = None
    nd = torch.full((args.batch_size,), model.null_ds, device=device)
    ns = torch.full((args.batch_size,), model.null_style, device=device)
    while step < total_steps:
        if perm is None or step % steps_per_epoch == 0:
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed * 1000003 + step // steps_per_epoch)).to(device)
        j = (step % steps_per_epoch) * args.batch_size * world + rank * args.batch_size
        toks = train_toks[perm[j: j + args.batch_size]].long()
        lr = lr_at(step, args.lr, args, total_steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            loss = ddp(toks, nd, ns)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
        opt.step()
        step += 1
        seqs_since += len(toks) * world
        acc += loss.detach().float()
        n_acc += 1

        if step % args.log_every == 0:
            a = reduce_mean(acc / n_acc, world)
            now = time.time()
            emit({"train/loss": a[0].item(), "train/bits_per_dim": a[0].item() / LN2 / 3, "train/perplexity": math.exp(a[0].item()),
                  "lr": lr, "epoch": step / steps_per_epoch, "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                  "perf/seq_per_s": seqs_since / (now - t_last)})
            log.info("step %d/%d ep %.2f loss %.4f (%.4f bits/dim) lr %.2e %.0f seq/s", step, total_steps, step / steps_per_epoch,
                     a[0].item(), a[0].item() / LN2 / 3, lr, seqs_since / (now - t_last))
            acc.zero_(); n_acc = 0; seqs_since = 0; t_last = now
        if step % args.val_every == 0:
            run_val()
        if step % args.sample_every == 0 and step < total_steps:
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

    if stop_reason == "finished":
        if step % args.val_every:
            run_val()
        log_samples()
        if rank == 0:
            torch.save(weights(), run_dir / "ar_final.pt")
            log.info("final weights: %s", run_dir / "ar_final.pt")
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
