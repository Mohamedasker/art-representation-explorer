"""
SimCLR (Chen et al., 2020, "A Simple Framework for Contrastive Learning of Visual Representations").

* encoder  : torchvision ResNet-50 (global-average-pooled 2048-D feature = THE representation used in the explorer)
* head     : projection MLP 2048 -> 2048 (BN, ReLU) -> 128, used only for the loss and discarded afterwards
* loss     : NT-Xent over the 2N views of a global batch; with DDP the embeddings of all GPUs are gathered
             (with gradients), so every anchor sees ``2 * global_batch - 2`` negatives
* optimiser: LARS (You et al., 2017) as in the paper, or AdamW (see scripts/13_train_simclr.py)

Published ImageNet weights (SimCLR v1, ResNet-50 1x, official TF checkpoint converted to PyTorch by
tonylins/simclr-converter, hosted by lightly-ai, Apache-2.0):
    https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x
"""

from __future__ import annotations

import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SIMCLR_V1_URL = "https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x/resolve/main/resnet50-1x.pth"
SIMCLR_V1_PATH = Path(__file__).resolve().parents[2] / "pretrained" / "simclrv1_r50_1x" / "resnet50-1x.pth"


@dataclass
class SimCLRConfig:
    arch: str = "resnet50"
    feat_dim: int = 2048
    proj_hidden: int = 2048
    proj_dim: int = 128
    img_size: int = 224

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimCLRConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SimCLR(nn.Module):
    def __init__(self, cfg: SimCLRConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SimCLRConfig()
        enc = getattr(torchvision.models, self.cfg.arch)(weights=None)
        enc.fc = nn.Identity()
        self.encoder = enc
        c = self.cfg
        self.projector = nn.Sequential(
            nn.Linear(c.feat_dim, c.proj_hidden, bias=False), nn.BatchNorm1d(c.proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(c.proj_hidden, c.proj_dim),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:        # (B, 2048) pooled encoder output
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # (B, 128) projection (loss space)
        return self.projector(self.encoder(x))


def load_simclr_v1_imagenet(path: str | Path = SIMCLR_V1_PATH) -> nn.Module:
    """torchvision ResNet-50 with the published ImageNet SimCLR v1 (1x) weights, ``fc`` removed (outputs 2048-D).

    The checkpoint (``{"state_dict": ...}`` with torchvision key names, incl. the linear-eval ``fc`` which is
    dropped here) is downloaded to ``pretrained/simclrv1_r50_1x/`` on first use - do that on a login node.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SIMCLR_V1_URL, path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Identity()
    model.load_state_dict(sd)
    return model


# ── NT-Xent with gathered negatives ───────────────────────────────────────────

class _GatherGrad(torch.autograd.Function):
    """all_gather whose backward sums the gradients of all ranks and returns the local slice."""

    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, 0)

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        n = g.shape[0] // dist.get_world_size()
        r = dist.get_rank()
        return g[r * n:(r + 1) * n]


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2, gather: bool = False):
    """NT-Xent / InfoNCE over the 2B views of a (global) batch.  Returns (mean loss over this rank's anchors, top-1
    accuracy of picking the positive).  With ``gather`` the keys come from all ranks; each rank computes the loss of its
    own anchors only, so DDP's gradient averaging yields the gradient of the global-batch mean loss."""
    z1, z2 = F.normalize(z1.float(), dim=1), F.normalize(z2.float(), dim=1)
    B = z1.shape[0]
    if gather and dist.is_initialized() and dist.get_world_size() > 1:
        W, r = dist.get_world_size(), dist.get_rank()
        g1, g2 = _GatherGrad.apply(z1), _GatherGrad.apply(z2)          # (W*B, d) each
        keys = torch.cat([g1, g2], 0)                                   # (2WB, d)
        q_idx = torch.cat([torch.arange(B) + r * B, torch.arange(B) + W * B + r * B]).to(z1.device)
        pos_idx = torch.cat([q_idx[B:], q_idx[:B]])                     # view1 <-> view2 of the same image
        queries = torch.cat([z1, z2], 0)
    else:
        keys = queries = torch.cat([z1, z2], 0)
        q_idx = torch.arange(2 * B, device=z1.device)
        pos_idx = torch.cat([q_idx[B:], q_idx[:B]])
    logits = queries @ keys.t() / temperature                           # (2B, 2WB)
    logits = logits.masked_fill(F.one_hot(q_idx, keys.shape[0]).bool(), float("-inf"))   # drop self-similarity
    loss = F.cross_entropy(logits, pos_idx)
    acc = (logits.argmax(1) == pos_idx).float().mean()
    return loss, acc


# ── LARS ──────────────────────────────────────────────────────────────────────

class LARS(torch.optim.Optimizer):
    """LARS with momentum (SimCLR/BYOL style): per-layer trust ratio for weights; biases and BN parameters
    (1-D tensors) get neither weight decay nor adaptation."""

    def __init__(self, params, lr: float, momentum: float = 0.9, weight_decay: float = 1e-6, eta: float = 0.001) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if p.ndim > 1:
                    d = d.add(p, alpha=g["weight_decay"])
                    pn, dn = torch.norm(p), torch.norm(d)
                    q = torch.where((pn > 0) & (dn > 0), g["eta"] * pn / dn, torch.ones_like(pn))
                    d = d.mul(q)
                st = self.state[p]
                if "mu" not in st:
                    st["mu"] = torch.zeros_like(p)
                mu = st["mu"]
                mu.mul_(g["momentum"]).add_(d)
                p.add_(mu, alpha=-g["lr"])
