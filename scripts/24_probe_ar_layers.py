#!/usr/bin/env python3
"""
Which transformer block gives the best image representation?  For every block of the trained GPT
(``scripts/23_train_ar.py``), mean-pool its hidden states over the 256 image positions (same recipe as the embedding
extractor, from the cached tokens of the centre-cropped images) and measure how well the ART STYLE can be read out:

  * kNN accuracy   (k = 20, cosine, features z-scored with the probe-train statistics),
  * logistic regression accuracy (L2-regularised, full-batch L-BFGS on the GPU),

trained on a random 10k subset of the train split and evaluated on the whole test split, for ArtBench-10 and WikiArt.
Writes ``<run>/layer_probe.json`` (the curve and ``best_layer`` = highest mean of the four accuracies) and
``<run>/layer_probe.png``.

    python scripts/24_probe_ar_layers.py --checkpoint runs/<run>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.models.ar_data import DATASETS, meta_file, token_file
from src.models.ar_extractor import load_gpt, run_dir_of


@torch.no_grad()
def layer_features(gpt, toks: np.ndarray, layers, device, bs: int = 256) -> dict[int, torch.Tensor]:
    out = {l: [] for l in layers}
    for i in range(0, len(toks), bs):
        t = torch.from_numpy(toks[i: i + bs].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            h = gpt.pooled_hidden(t, layers)
        for l in layers:
            out[l].append(h[l].float())
    return {l: torch.cat(v) for l, v in out.items()}


def knn_acc(xtr, ytr, xte, yte, k: int = 20) -> float:
    xtr, xte = F.normalize(xtr, dim=1), F.normalize(xte, dim=1)
    n_cls = int(ytr.max()) + 1
    votes = torch.zeros(len(xte), n_cls, device=xte.device)
    for i in range(0, len(xte), 2048):
        sim, nn = (xte[i: i + 2048] @ xtr.t()).topk(k, dim=1)
        votes[i: i + 2048].scatter_add_(1, ytr[nn], sim.clamp_min(0))
    return (votes.argmax(1) == yte).float().mean().item()


def logreg_acc(xtr, ytr, xte, yte, l2: float = 1e-3, iters: int = 200) -> float:
    n_cls = int(ytr.max()) + 1
    W = torch.zeros(xtr.shape[1], n_cls, device=xtr.device, requires_grad=True)
    b = torch.zeros(n_cls, device=xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ W + b, ytr) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return ((xte @ W + b).argmax(1) == yte).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="GPT checkpoint (ar_final.pt / checkpoints/best.pt)")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--cache-dir", default="data/ar_tokens")
    ap.add_argument("--out-dir", default=None, help="default: the run directory")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt, ck = load_gpt(args.checkpoint, device)
    run = run_dir_of(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else run
    tok_name = ck["info"]["tokenizer_name"]
    layers = list(range(1, gpt.cfg.n_layer + 1))
    res = {"checkpoint": str(args.checkpoint), "step": ck.get("step"), "layers": layers, "n_probe_train": args.n_train,
           "metrics": {}}
    for ds in DATASETS:
        ytr_all = np.load(meta_file(ds, "train", args.cache_dir))["style_id"].astype(np.int64)
        yte = np.load(meta_file(ds, "test", args.cache_dir))["style_id"].astype(np.int64)
        sel = np.sort(np.random.default_rng(0).permutation(len(ytr_all))[: args.n_train])
        ttr = np.load(token_file(tok_name, ds, "train", False, args.cache_dir), mmap_mode="r")[sel]
        tte = np.load(token_file(tok_name, ds, "test", False, args.cache_dir))
        ftr, fte = layer_features(gpt, ttr, layers, device), layer_features(gpt, tte, layers, device)
        _, ytr = np.unique(ytr_all[sel], return_inverse=True)       # contiguous class ids over the classes present
        classes = np.unique(ytr_all[sel])
        keep = np.isin(yte, classes)
        yte_c = torch.from_numpy(np.searchsorted(classes, yte[keep])).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        res["metrics"][ds] = {"knn": [], "logreg": [], "chance": float(np.bincount(ytr).max() / len(ytr))}
        for l in layers:
            mu, sd = ftr[l].mean(0), ftr[l].std(0).clamp_min(1e-6)
            xtr, xte = (ftr[l] - mu) / sd, ((fte[l] - mu) / sd)[torch.from_numpy(keep).to(device)]
            res["metrics"][ds]["knn"].append(knn_acc(xtr, ytr_t, xte, yte_c))
            res["metrics"][ds]["logreg"].append(logreg_acc(xtr, ytr_t, xte, yte_c))
            print(f"{ds:9s} layer {l:2d}  kNN {res['metrics'][ds]['knn'][-1]:.4f}  logreg {res['metrics'][ds]['logreg'][-1]:.4f}",
                  flush=True)
    score = np.mean([res["metrics"][d][m] for d in DATASETS for m in ("knn", "logreg")], axis=0)
    res["mean_score"] = score.tolist()
    res["best_layer"] = int(layers[int(score.argmax())])
    print("mean of the four accuracies per layer:", np.round(score, 4).tolist(), "-> best layer", res["best_layer"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_probe.json").write_text(json.dumps(res, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
        for a, ds in zip(ax, DATASETS):
            a.plot(layers, res["metrics"][ds]["knn"], "o-", label="kNN (k=20)")
            a.plot(layers, res["metrics"][ds]["logreg"], "s-", label="logistic regression")
            a.axhline(res["metrics"][ds]["chance"], color="grey", ls=":", label="majority class")
            a.axvline(res["best_layer"], color="C3", ls="--", lw=0.8)
            a.set_title(f"{ds}: style probe (10k train, full test)"); a.set_xlabel("transformer block"); a.set_ylabel("accuracy")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "layer_probe.png", dpi=130)
    except Exception as e:                                          # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
