"""Frozen-feature monitors used during self-supervised training: kNN (k=20, cosine, temperature-weighted vote) and a
linear probe (L2-regularised logistic regression, L-BFGS on standardised features), plus a feature-collapse indicator.
Same definitions as in scripts/13_train_simclr.py, so the numbers are comparable with the SimCLR runs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def feature_std(f: torch.Tensor) -> float:
    """Mean per-dimension std of the L2-normalised features, times sqrt(D): ~1 for spread-out (isotropic) features,
    -> 0 when the representation collapses to a point or a low-dimensional subspace."""
    f = F.normalize(f.float(), dim=1)
    return (f.std(0).mean() * f.shape[1] ** 0.5).item()
