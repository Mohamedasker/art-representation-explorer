"""
Nudity flags for the explorer.

Every image gets two independent detector scores plus the dataset's own metadata, and a boolean flag
that is used to hide the image by default:

  * ``nn``    — NudeNet (ONNX object detector for exposed body parts): highest confidence among
                exposed female breast / genitalia, male genitalia, buttocks and anus.
  * ``clip``  — CLIP ViT-L/14 zero-shot: probability mass on "nude / naked / erotic" prompts versus
                a broad set of non-nude painting prompts.
  * ``meta``  — the image is explicitly labelled as a nude: WikiArt genre ``nude_painting`` (ArtBench: the title
                says nude / naked / ...).

An image is flagged when ANY of them fires (high recall: the default is to hide, and a harmless
image that is hidden by mistake costs less than an explicit one that is shown).  The raw scores are
saved, so thresholds can be changed without re-running the detectors (``build_flags``).

  * ``clip32``— the "nudity" concept score of the explorer's CLIP ViT-B/32 attributes (data/attributes/*_all.npz,
                ``clip_nude``): what people see when they colour the maps by "nudity".  Flagged from 2.5 upwards (about the
                top 7 %), so that the top of that colouring is hidden too.  Loose on purpose: it also catches sculptures
                and abstract body forms.
  * ``probe`` — logistic regression on CLIP ViT-L/14 image features, trained on WikiArt's own ``nude_painting``
                genre labels (scripts/16_train_nudity_probe.py): the most precise signal, but it misses nudes
                that WikiArt files under other genres (statues, male nudes, mythological and religious scenes).

Images are keyed by the last four components of their path, ``<dataset>/<split>/<style>/<file>``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

FLAGS_FILE = Path(__file__).resolve().parents[2] / "data" / "nudity" / "flags.npz"

NN_EXPLICIT = ("FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
               "BUTTOCKS_EXPOSED", "ANUS_EXPOSED")

CLIP_POSITIVE = [
    "a nude painting of a naked person",
    "a painting of a naked woman with bare breasts",
    "a painting of a naked man",
    "an erotic painting",
    "a nude figure with exposed genitals",
    "a drawing of a nude body",
]
CLIP_NEGATIVE = [
    "a painting of a clothed person",
    "a portrait of a person wearing clothes",
    "a landscape painting",
    "a still life painting",
    "a painting of a building",
    "an abstract painting",
    "a painting of animals",
    "a religious painting of saints in robes",
    "a painting of a crowd of clothed people",
    "a sketch of a face",
    "a painting of the sea",
    "a decorative pattern or ornament",
    "a photograph of an object",
]

# Thresholds, set from sample sheets of ArtBench and the WikiArt genre labels (scripts/16_train_nudity_probe.py):
#   probe >= 0.54  -> ~90 % recall of WikiArt's nude_painting labels, almost all genuine nudes on ArtBench
#   clip  >= 0.9   -> CLIP zero-shot is only reliable at the very top (0.5 flags 19 % of ArtBench, mostly clothed figures)
#   nn    >= 0.6   -> NudeNet is precise above 0.6 but noisy at 0.35-0.6 (portraits, landscapes), and misses many paintings
DEFAULT_THRESHOLDS = {"nn": 0.6, "clip": 0.9, "probe": 0.40, "clip32": 2.5}


def image_key(path: str | Path) -> str:
    """``data/artbench/train/baroque/x.jpg`` -> ``artbench/train/baroque/x.jpg``."""
    return "/".join(Path(str(path)).parts[-4:])


# ── detectors ─────────────────────────────────────────────────────────────────

def clip_nudity_scores(files: Sequence[str | Path], device: str = "cuda", batch_size: int = 256,
                       workers: int = 16, model_name: str = "ViT-L-14", pretrained: str = "openai"):
    """(P(nude prompts), L2-normalised image features) per image, from CLIP zero-shot over
    CLIP_POSITIVE + CLIP_NEGATIVE.  The features are kept so a probe can be trained on them."""
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    prompts = CLIP_POSITIVE + CLIP_NEGATIVE
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
    n_pos = len(CLIP_POSITIVE)

    class DS(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            with Image.open(files[i]) as im:
                return preprocess(im.convert("RGB"))

    out, feats = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for x in tqdm(DataLoader(DS(), batch_size=batch_size, num_workers=workers), desc="CLIP ViT-L/14"):
            f = model.encode_image(x.to(device))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f.float() @ t.float().T).softmax(dim=-1)
            out.append(p[:, :n_pos].sum(dim=-1).cpu())
            feats.append(f.half().cpu())
    return torch.cat(out).numpy().astype(np.float32), torch.cat(feats).numpy()


_DETECTOR = None


def _nn_init() -> None:
    """One single-threaded detector per worker process.  NudeDetector opens its ONNX session with onnxruntime's
    default of one thread per core, which with dozens of workers exceeds the per-user thread limit."""
    global _DETECTOR
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    orig = ort.InferenceSession
    ort.InferenceSession = lambda path, *a, **k: orig(path, sess_options=so, providers=["CPUExecutionProvider"])
    try:
        from nudenet import NudeDetector
        _DETECTOR = NudeDetector()
    finally:
        ort.InferenceSession = orig


def _nn_one(path: str) -> tuple[float, str]:
    best, cls = 0.0, ""
    try:
        for d in _DETECTOR.detect(path):
            if d["class"] in NN_EXPLICIT and d["score"] > best:
                best, cls = float(d["score"]), d["class"]
    except Exception:
        pass
    return best, cls


def nudenet_scores(files: Sequence[str | Path], workers: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(max exposed-part confidence, its class) per image."""
    from tqdm import tqdm
    with ProcessPoolExecutor(workers, initializer=_nn_init) as ex:
        res = list(tqdm(ex.map(_nn_one, [str(f) for f in files], chunksize=64), total=len(files), desc="NudeNet"))
    return np.array([r[0] for r in res], dtype=np.float32), np.array([r[1] for r in res])


# ── flags ─────────────────────────────────────────────────────────────────────

def build_flags(clip: np.ndarray, nn: np.ndarray, meta: np.ndarray, thresholds: dict | None = None,
                probe: np.ndarray | None = None, clip32: np.ndarray | None = None) -> np.ndarray:
    """Reason bitmask per image: 1 = NudeNet, 2 = CLIP ViT-L/14 zero-shot, 4 = metadata, 8 = probe,
    16 = CLIP ViT-B/32 nudity concept (flagged when non-zero)."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = ((nn >= th["nn"]).astype(np.uint8) | ((clip >= th["clip"]).astype(np.uint8) << 1)
           | (meta.astype(np.uint8) << 2))
    if probe is not None:
        out = out | ((probe >= th["probe"]).astype(np.uint8) << 3)
    if clip32 is not None:
        out = out | ((np.nan_to_num(clip32, nan=-99) >= th["clip32"]).astype(np.uint8) << 4)
    return out


_CACHE: dict | None = None


def load_flags(paths: Sequence[str | Path], flags_file: Path | None = None) -> np.ndarray | None:
    """Boolean array aligned with ``paths`` (True = flagged), or None if no flags file / images missing from it."""
    global _CACHE
    f = Path(flags_file or FLAGS_FILE)
    if not f.exists():
        return None
    if _CACHE is None or _CACHE["file"] != str(f) or _CACHE["mtime"] != f.stat().st_mtime:
        z = np.load(f, allow_pickle=False)
        _CACHE = {"file": str(f), "mtime": f.stat().st_mtime,
                  "row": {str(k): i for i, k in enumerate(z["keys"])}, "reason": z["reason"]}
    row = _CACHE["row"]
    sel = np.array([row.get(image_key(p), -1) for p in paths])
    if (sel < 0).any():
        return None
    return _CACHE["reason"][sel] > 0
