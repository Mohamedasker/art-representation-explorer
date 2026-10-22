#!/usr/bin/env python3
"""
Train the autoregressive image transformer (``src/models/image_gpt.py``) on the VQ tokens of the TRAIN splits of
ArtBench-10 and WikiArt jointly (+ their horizontally flipped copies), raster-order next-token prediction with
cross-entropy, DistributedDataParallel + Weights & Biases.  Needs the token arrays of ``scripts/22_tokenize_ar.py``.

Prefix tokens (dataset, style) are replaced by "no label" for a random ``--label-dropout`` share of the sequences, so the
same model generates unconditionally, style-conditionally (optionally with classifier-free guidance) and embeds unlabeled
images.  Validation = both TEST splits: bits per token / perplexity with labels and without (``val/uncond_*``), also split
by dataset.  Sample grids (unconditional, ArtBench styles, WikiArt styles) are decoded with the tokenizer and written to
``runs/<run>/samples/`` and to wandb.

    torchrun --standalone --nproc_per_node=4 scripts/23_train_ar.py --tokenizer runs/<vq run>/vq_final.pt --epochs 30
    python scripts/23_train_ar.py --tokenizer ... --max-steps 20 --name smoke_ar --n-layer 2 --d-model 256 --n-head 4
    torchrun ... scripts/23_train_ar.py --resume runs/<stamp>_ar_gpt_artbench_wikiart

Writes ``ar_final.pt`` (last weights) and ``checkpoints/best.pt`` (lowest unlabeled validation loss).
Embeddings: scripts/24_probe_ar_layers.py picks the layer, scripts/02_extract_features.py --model ar_gpt extracts them.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

from src.datasets.artbench import STYLES as ARTBENCH_STYLES
from src.datasets.wikiart import STYLES as WIKIART_STYLES
from src.models.ar_data import DATASETS, STYLE_NAMES, meta_file, token_file
from src.models.image_gpt import GPTConfig, ImageGPT
from src.training.common import init_wandb, lr_at, reduce_mean, save_ckpt_atomic, setup_dist, setup_logging

log = logging.getLogger("ar")
LN2 = math.log(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue (uses its config.json)")
    p.add_argument("--tokenizer", type=str, default=None, help="vq_final.pt of scripts/21_train_vq.py")
    p.add_argument("--name", type=str, default="ar_gpt_artbench_wikiart")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--cache-dir", type=str, default="data/ar_tokens")
    # model
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-dropout", type=float, default=0.2, help="share of sequences with both prefix tokens set to 'no label'")
    # optimisation
    p.add_argument("--epochs", type=int, default=30, help="passes over train + flipped train (2 x 123k sequences)")
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (needs a working Triton/gcc toolchain; fails on the compute nodes)")
    p.add_argument("--seed", type=int, default=0)
    # validation / logging / checkpointing
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=5000, help="decode sample grids every N steps (and at the end)")
    p.add_argument("--n-per-style", type=int, default=6)
    p.add_argument("--cfg-scale", type=float, default=1.5, help="classifier-free guidance scale of the conditional sample grids")
    p.add_argument("--top-k", type=int, default=600)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    p.add_argument("--wandb-project", type=str, default="art-representation-explorer")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def load_split(tok_name: str, split: str, cache_dir: str, device, flips: bool):
    """Token array (n, 256) int16, dataset id (n,), style id (n,) of both datasets (and flipped copies if ``flips``)."""
    toks, dss, sts = [], [], []
    for di, d in enumerate(DATASETS):
        st = np.load(meta_file(d, split, cache_dir))["style_id"].astype(np.int64)
        for flip in ((False, True) if flips else (False,)):
            t = np.load(token_file(tok_name, d, split, flip, cache_dir))
            toks.append(torch.from_numpy(t.astype(np.int16)))
            dss.append(torch.full((len(t),), di, dtype=torch.long))
            sts.append(torch.from_numpy(st))
    return torch.cat(toks).to(device), torch.cat(dss).to(device), torch.cat(sts).to(device)


@torch.no_grad()
def validate(model, val, device, args, world, rank):
    """Cross-entropy (nats / token) on the test splits, with labels and with 'no label' prefix, overall and per dataset."""
    toks, ds, st = val
    model.eval()
    out = {}
    sums = torch.zeros(2, 2, 2, device=device)                 # [labeled?][dataset][sum loss, n]
    nd, ns = torch.full_like(ds, model.null_ds), torch.full_like(st, model.null_style)
    for i in range(rank * 256, len(toks), 256 * world):
        sl = slice(i, i + 256)
        for k, (d_, s_) in enumerate(((ds[sl], st[sl]), (nd[sl], ns[sl]))):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
                _, logits = model(toks[sl].long(), d_, s_, return_logits=True)
            l = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), toks[sl].long(), reduction="none").mean(1)
            for di in range(2):
                m = ds[sl] == di
                sums[k, di, 0] += l[m].sum()
                sums[k, di, 1] += m.sum()
    if world > 1:
        dist.all_reduce(sums)
    for k, pre in enumerate(("val", "val/uncond")):
        tot = sums[k].sum(0)
        out[f"{pre}_loss"] = (tot[0] / tot[1]).item()
        out[f"{pre}_bits_per_token"] = out[f"{pre}_loss"] / LN2
        out[f"{pre}_perplexity"] = math.exp(out[f"{pre}_loss"])
        for di, d in enumerate(DATASETS):
            out[f"{pre}/{d}_bits_per_token"] = (sums[k, di, 0] / sums[k, di, 1]).item() / LN2
    model.train()
    return out


# ── samples ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(vq, toks: torch.Tensor) -> torch.Tensor:
    with torch.autocast(toks.device.type, dtype=torch.bfloat16):
        x = vq.decode_ids(toks)
    return ((x.float().clamp(-1, 1) + 1) / 2).cpu()


@torch.no_grad()
def make_sample_grids(model, vq, args, device) -> dict:
    g = torch.Generator(device=device).manual_seed(0)
    n = args.n_per_style
    kw = dict(temperature=1.0, top_k=args.top_k, generator=g)
    grids = {}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
        t = model.generate(24, **kw)
        grids["unconditional"] = make_grid(decode(vq, t), nrow=6, padding=2)
        for di, d in enumerate(DATASETS):
            styles = [s for s in STYLE_NAMES if s in (ARTBENCH_STYLES if d == "artbench" else WIKIART_STYLES)]
            styles = styles[:10] if d == "artbench" else styles[:: max(1, len(styles) // 10)][:10]
            rows = []
            for s in styles:
                t = model.generate(n, ds=di, style=STYLE_NAMES.index(s), cfg_scale=args.cfg_scale, **kw)
                rows.append(decode(vq, t))
            grids[f"{d}_styles"] = make_grid(torch.cat(rows), nrow=n, padding=2)
            grids[f"{d}_styles_labels"] = ", ".join(styles)
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
        if args.tokenizer is None:
            raise SystemExit("--tokenizer is required for a new run")
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

    # ── data ─────────────────────────────────────────────────────────────────
    if resume_dir is not None:
        tok_path = json.loads((run_dir / "config.json").read_text())["info"]["tokenizer"]
    else:
        tok_path = str(Path(args.tokenizer).resolve())
    from src.models.vq_tokenizer import VQConfig, VQTokenizer
    ck = torch.load(tok_path, map_location="cpu", weights_only=False)
    vq_cfg = VQConfig.from_dict(ck["model_cfg"])
    tok_run = Path(tok_path).parent.parent if Path(tok_path).parent.name == "checkpoints" else Path(tok_path).parent
    tok_name = tok_run.name
    train = load_split(tok_name, "train", args.cache_dir, device, flips=True)
    val = load_split(tok_name, "test", args.cache_dir, device, flips=False)
    n_train = len(train[0])
    steps_per_epoch = n_train // (args.batch_size * world)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    # ── model / optim ────────────────────────────────────────────────────────
    mcfg = GPTConfig(vocab_size=vq_cfg.codebook_size, n_tokens=vq_cfg.n_tokens, n_layer=args.n_layer,
                     n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
    model = ImageGPT(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else model
    fwd = torch.compile(ddp) if args.compile else ddp
    opt = torch.optim.AdamW(model.param_groups(args.weight_decay), lr=args.lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    step, best_val = 0, float("inf")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if resume_dir is not None and ckpt_path.exists():
        ck2 = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck2["model"])
        opt.load_state_dict(ck2["optimizer"])
        step, best_val = ck2["step"], ck2["best_val"]
        log.info("Resumed from %s at step %d", ckpt_path, step)

    info = {"tokenizer": tok_path, "tokenizer_name": tok_name, "train_datasets": list(DATASETS), "img_size": vq_cfg.img_size,
            "n_tokens": vq_cfg.n_tokens, "codebook_size": vq_cfg.codebook_size, "style_names": STYLE_NAMES,
            "embedding": "mean over the 256 image positions of one intermediate block (layer picked by scripts/24_probe_ar_layers.py)"}
    cfg = {"args": vars(args), "model": mcfg.to_dict(), "info": info, "n_params": n_params, "train_sequences": n_train,
           "val_sequences": len(val[0]), "world_size": world, "global_batch_size": args.batch_size * world,
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

    vq = None
    if rank == 0:
        vq = VQTokenizer(vq_cfg)
        vq.load_state_dict(ck["model"])
        vq = vq.to(device).eval()

    def log_samples() -> None:
        if rank != 0:
            return
        t0 = time.time()
        grids = make_sample_grids(model, vq, args, device)
        model.train()
        wb = {}
        for k, g in grids.items():
            if k.endswith("_labels"):
                continue
            save_image(g, run_dir / "samples" / f"step_{step:07d}_{k}.png")
            if run is not None:
                import wandb
                cap = f"step {step}: " + ("unconditional (no label), temperature 1, top-k %d" % args.top_k if k == "unconditional"
                      else f"rows = {grids[k + '_labels']} ({args.n_per_style} samples each, CFG scale {args.cfg_scale}, top-k {args.top_k})")
                wb[f"samples/{k}"] = wandb.Image(g, caption=cap)
        if run is not None and wb:
            run.log({"step": step, **wb}, step=step)
        log.info("step %d: sample grids written (%.0fs)", step, time.time() - t0)

    def run_val() -> None:
        nonlocal best_val
        m = validate(model, val, device, args, world, rank)
        emit(m)
        log.info("step %d  val %.3f bits/token (ppl %.1f), unlabeled %.3f bits/token  [artbench %.3f, wikiart %.3f]", step,
                 m["val_bits_per_token"], m["val_perplexity"], m["val/uncond_bits_per_token"],
                 m["val/artbench_bits_per_token"], m["val/wikiart_bits_per_token"])
        if m["val/uncond_loss"] < best_val:
            best_val = m["val/uncond_loss"]
            if rank == 0:
                save_ckpt_atomic({**weights(), "val": m}, run_dir / "checkpoints" / "best.pt")

    # ── train loop ───────────────────────────────────────────────────────────
    model.train()
    stop_reason = "finished"
    t_last, seqs_since = time.time(), 0
    acc, n_acc = torch.zeros(1, device=device), 0
    if step == 0:
        run_val()
    gen = torch.Generator(device=device)
    perm = None
    while step < total_steps:
        if perm is None or step % steps_per_epoch == 0:
            ep = step // steps_per_epoch
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed * 1000003 + ep)).to(device)
        j = (step % steps_per_epoch) * args.batch_size * world + rank * args.batch_size
        idx = perm[j: j + args.batch_size]
        toks, ds, st = train[0][idx].long(), train[1][idx], train[2][idx]
        gen.manual_seed(args.seed * 7919 + step * 131 + rank)
        drop = torch.rand(len(idx), device=device, generator=gen) < args.label_dropout
        ds = torch.where(drop, torch.full_like(ds, model.null_ds), ds)
        st = torch.where(drop, torch.full_like(st, model.null_style), st)
        lr = lr_at(step, args.lr, args, total_steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp):
            loss = fwd(toks, ds, st)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
        opt.step()
        step += 1
        seqs_since += len(idx) * world
        acc += loss.detach().float()
        n_acc += 1

        if step % args.log_every == 0:
            a = reduce_mean(acc / n_acc, world)
            now = time.time()
            emit({"train/loss": a[0].item(), "train/bits_per_token": a[0].item() / LN2, "train/perplexity": math.exp(a[0].item()),
                  "lr": lr, "epoch": step / steps_per_epoch, "train/grad_norm": float(gnorm) if gnorm is not None else 0.0,
                  "perf/seq_per_s": seqs_since / (now - t_last)})
            log.info("step %d/%d ep %.2f loss %.4f (%.3f bits/token) lr %.2e %.0f seq/s", step, total_steps,
                     step / steps_per_epoch, a[0].item(), a[0].item() / LN2, lr, seqs_since / (now - t_last))
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

    # ── wrap up ──────────────────────────────────────────────────────────────
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
