"""
Gentle CLIP fine-tuning on one art dataset (ArtBench-10 or WikiArt): shared pieces of the training script
(scripts/28_train_clip_ft.py), the evaluation script (scripts/29_eval_clip_ft.py) and the extractor
(src/models/clip_ft_extractor.py).

Recipe (aim: fit the art data better without forgetting what CLIP learned):
  * contrastive loss between the images and templated captions built from the dataset's labels,
  * the text tower and the logit scale are FROZEN, so the caption embeddings are a fixed table that anchors the image
    tower to CLIP's original text space (and need to be computed only once),
  * low peak LR with layer-wise decay, few epochs,
  * the final image tower can be interpolated with the original weights (WiSE-FT, ``blend_visual``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.artbench import ArtBenchDataset
from src.datasets.wikiart import WikiArtDataset
from src.metadata import attributes as A

try:
    import open_clip
except ImportError:                                       # pragma: no cover
    open_clip = None

# OpenAI weights need the QuickGELU activation; plain "ViT-B-32" + "openai" (what the older frozen `clip_vitb32`
# representation uses) builds GELU instead and shifts the features (mean cosine 0.96 to the exact model).
DEFAULT_ARCH, DEFAULT_PRETRAINED = "ViT-B-32-quickgelu", "openai"
DATASET_CLASSES = {"artbench": ArtBenchDataset, "wikiart": WikiArtDataset}
UNKNOWN = re.compile(r"^unknown", re.I)
GENRE_PHRASE = {"sketch_and_study": "sketch", "illustration": "illustration", "nude_painting": "nude painting"}


# ── captions ──────────────────────────────────────────────────────────────────

def pretty(slug: str) -> str:
    """'claude-monet' / 'art_nouveau' -> 'Claude Monet' / 'art nouveau'-style words (capitalised by the caller)."""
    return re.sub(r"[-_]+", " ", slug).strip()


def make_caption(style: str, genre: str | None, artist: str | None, use_style: bool = True) -> str:
    """'a landscape painting by Vincent Van Gogh, in the post impressionism style' (parts without label are left out)."""
    kind = "painting"
    if genre and not UNKNOWN.match(genre):
        g = GENRE_PHRASE.get(genre, pretty(genre))
        kind = g if g.endswith(("painting", "illustration", "sketch")) else f"{g} painting"
    cap = f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
    if artist and not UNKNOWN.match(artist):
        cap += f" by {pretty(artist).title()}"
    if use_style:
        cap += f", in the {pretty(style)} style"
    return cap


def style_prompt(style: str) -> str:
    return f"a painting in the {pretty(style)} style"


def dataset_items(dataset: str, root: str | Path, split: str, wikiart_root: str | Path | None = None,
                  use_style: bool = True):
    """(paths, style names, captions) of one split of one dataset."""
    ds = DATASET_CLASSES[dataset](root, split=split)
    paths = [str(p) for p in ds.get_paths()]
    styles = [Path(p).parent.name for p in paths]
    genres: list[str | None] = [None] * len(paths)
    artists: list[str | None] = [None] * len(paths)
    if dataset == "wikiart":
        from src.datasets.labeled_union import _hf_labels
        lab = _hf_labels(Path(root))
        for i, p in enumerate(paths):
            l = lab.get(Path(p).stem)
            if l:
                genres[i], artists[i] = l["genre"], l["artist"]
    else:
        artists = [A.artbench_artist(Path(p).stem) for p in paths]
    caps = [make_caption(s, g, a, use_style) for s, g, a in zip(styles, genres, artists)]
    return paths, styles, caps


class CaptionedImages(Dataset):
    """``(image, caption_id, style_id)``; caption ids index ``captions`` (the unique captions of the training split)."""

    def __init__(self, paths, caption_ids, style_ids, transform, draft_size: int = 224):
        self.paths, self.caption_ids, self.style_ids = paths, np.asarray(caption_ids), np.asarray(style_ids)
        self.transform, self.draft_size = transform, draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i])
        img.draft("RGB", (self.draft_size * 2, self.draft_size * 2))
        return self.transform(img.convert("RGB")), int(self.caption_ids[i]), int(self.style_ids[i])


# ── model ─────────────────────────────────────────────────────────────────────

def load_clip(arch: str = DEFAULT_ARCH, pretrained: str = DEFAULT_PRETRAINED, device="cpu"):
    """(open_clip model, tokenizer, eval preprocess transform)."""
    if open_clip is None:
        raise ImportError("open_clip_torch is required (pip install open-clip-torch)")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    return model, open_clip.get_tokenizer(arch), preprocess


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device, batch_size: int = 512) -> torch.Tensor:
    """L2-normalised text features, (len(texts), D), float32, on ``device``."""
    out = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i:i + batch_size]).to(device)
        out.append(F.normalize(model.encode_text(tok).float(), dim=-1))
    return torch.cat(out)


def layer_scales(visual: torch.nn.Module, decay: float) -> dict[str, float]:
    """Layer-wise LR factors per open_clip VisionTransformer parameter (embeddings smallest, last block / head = 1)."""
    n = len(visual.transformer.resblocks)
    out = {}
    for name, _ in visual.named_parameters():
        m = re.match(r"transformer\.resblocks\.(\d+)\.", name)
        layer = int(m.group(1)) + 1 if m else (n + 1 if name.startswith(("ln_post", "proj", "head")) else 0)
        out[name] = decay ** (n + 1 - layer)
    return out


def blend_visual(ft_state: dict, base_state: dict, alpha: float) -> dict:
    """WiSE-FT: ``alpha`` * fine-tuned + (1 - alpha) * original weights (alpha = 1: fine-tuned, 0: original)."""
    if alpha >= 1.0:
        return ft_state
    return {k: (alpha * v.float() + (1 - alpha) * base_state[k].float()).to(v.dtype) if v.is_floating_point() else v
            for k, v in ft_state.items()}


def load_ft_visual(model, ckpt: dict, alpha: float = 1.0) -> None:
    """Load the fine-tuned image tower of a clip_ft checkpoint into ``model`` (a freshly built pretrained CLIP)."""
    base = {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}
    model.visual.load_state_dict(blend_visual(ckpt["visual"], base, alpha))


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_images(model, loader, device, amp: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Normalised image features for a loader yielding ``(image, *rest)``; returns (features, [stacked rest...])."""
    feats, rest = [], None
    for x, *r in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            f = model.encode_image(x.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=-1))
        rest = [[] for _ in r] if rest is None else rest
        for lst, v in zip(rest, r):
            lst.append(torch.as_tensor(v))
    return torch.cat(feats), [torch.cat(r) for r in (rest or [])]


@torch.no_grad()
def zero_shot_accuracy(img_feats: torch.Tensor, labels: torch.Tensor, class_text: torch.Tensor) -> float:
    return float(((img_feats @ class_text.T).argmax(1).cpu() == labels.cpu()).float().mean())


def generic_testsets(root: str | Path) -> dict[str, tuple[list[str], object]]:
    """{'cifar100': (class names, torchvision test set), 'cifar10': ...} for the sets found under ``root``."""
    import torchvision
    sets = {}
    for name, cls, sub in (("cifar100", torchvision.datasets.CIFAR100, "cifar-100-python"),
                           ("cifar10", torchvision.datasets.CIFAR10, "cifar-10-batches-py")):
        if (Path(root) / sub).is_dir():
            ds = cls(str(root), train=False, download=False)
            sets[name] = ([c.replace("_", " ") for c in ds.classes], ds)
    return sets


class _Transformed(Dataset):
    def __init__(self, ds, tf):
        self.ds, self.tf = ds, tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]
        return self.tf(img.convert("RGB")), y


def generic_zero_shot(model, tokenizer, preprocess, root, device, batch_size: int = 256, num_workers: int = 4,
                      max_images: int | None = None, text_cache: dict | None = None) -> dict[str, float]:
    """Zero-shot top-1 on CIFAR-100 / CIFAR-10 test (prompt 'a photo of a {class}.') -- the 'did it forget?' probe."""
    from torch.utils.data import DataLoader, Subset
    out = {}
    for name, (classes, ds) in generic_testsets(root).items():
        if text_cache is not None and name in text_cache:
            text = text_cache[name]
        else:
            text = encode_texts(model, tokenizer, [f"a photo of a {c}." for c in classes], device)
            if text_cache is not None:
                text_cache[name] = text
        if max_images:
            ds = Subset(ds, list(range(min(max_images, len(ds)))))
        loader = DataLoader(_Transformed(ds, preprocess), batch_size=batch_size, num_workers=num_workers)
        feats, (labels,) = encode_images(model, loader, device)
        out[name] = zero_shot_accuracy(feats, labels, text)
    return out
