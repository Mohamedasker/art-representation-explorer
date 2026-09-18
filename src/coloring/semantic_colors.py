"""
Semantic color values from CLIP zero-shot predictions.

For each image we compute cosine similarity to a set of text prompts and
assign the best-matching concept as the semantic label.  This is useful for
exploring high-level themes (e.g. "portrait", "landscape", "abstract") that
cut across official art-style labels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.models.base_extractor import RepresentationBundle


# ── Default prompt sets ───────────────────────────────────────────────────────

SUBJECT_PROMPTS: list[str] = [
    "a painting of a portrait or face",
    "a painting of a landscape or nature scene",
    "a painting of an architectural interior or building",
    "a painting of a seascape or water scene",
    "a painting with abstract or geometric forms",
    "a painting of a still life with objects",
    "a painting of a battle or historical scene",
    "a painting of a religious or mythological scene",
    "a painting of animals or wildlife",
    "a painting of urban life or people in a city",
]

MOOD_PROMPTS: list[str] = [
    "a dark, melancholic painting",
    "a bright, joyful painting",
    "a mysterious, eerie painting",
    "a serene, peaceful painting",
    "a dramatic, intense painting",
    "a romantic, dreamy painting",
]

PERIOD_PROMPTS: list[str] = [
    "an artwork from the medieval period",
    "an artwork from the Renaissance",
    "an artwork from the 17th or 18th century",
    "an artwork from the 19th century",
    "a modern artwork from the 20th century",
    "a contemporary artwork",
]


# ── Main function ─────────────────────────────────────────────────────────────

def compute_semantic_labels(
    bundle: RepresentationBundle,
    prompts: Optional[list[str]] = None,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    temperature: float = 100.0,
) -> dict:
    """
    Assign zero-shot semantic labels using CLIP.

    Parameters
    ----------
    bundle:
        Representation bundle whose ``vectors`` are CLIP features (normalised).
        If the bundle was produced by a non-CLIP extractor the function still
        works but re-encodes the images, which requires paths to be available.
    prompts:
        Custom text prompts — one per class.  Defaults to subject prompts.
    clip_model_name:
        CLIP architecture (only used when re-encoding is needed).
    clip_pretrained:
        CLIP pretrained weights tag.
    temperature:
        Logit temperature.

    Returns
    -------
    dict
        Colorvar dict with best-matching concept per image.
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip_torch is required for semantic coloring.\n"
            "Install with:  pip install open-clip-torch"
        )

    if prompts is None:
        prompts = SUBJECT_PROMPTS

    # -- encode text prompts --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)  # (C, D)

    # -- check if bundle vectors are CLIP features (same dim) --
    img_feats_np = bundle.vectors.astype(np.float32)
    clip_dim = int(text_feats.shape[-1])

    if img_feats_np.shape[1] == clip_dim:
        # assume these are already CLIP image features
        img_feats = torch.tensor(img_feats_np, device=device)
        img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    else:
        # re-encode images from disk — slow but correct
        print(
            f"  Bundle dim {img_feats_np.shape[1]} ≠ CLIP dim {clip_dim}. "
            "Re-encoding images with CLIP …"
        )
        from PIL import Image as PILImage
        from tqdm import tqdm

        img_list: list[torch.Tensor] = []
        for path in tqdm(bundle.paths, desc="CLIP re-encoding"):
            img = PILImage.open(str(path)).convert("RGB")
            img_list.append(preprocess(img))

        batch_size = 64
        all_feats: list[torch.Tensor] = []
        for i in range(0, len(img_list), batch_size):
            batch = torch.stack(img_list[i:i + batch_size]).to(device)
            with torch.no_grad():
                f = model.encode_image(batch)
                f = torch.nn.functional.normalize(f, dim=-1)
            all_feats.append(f)
        img_feats = torch.cat(all_feats, dim=0)

    # -- compute similarity and pick best concept --
    with torch.no_grad():
        logits = img_feats @ text_feats.T * temperature   # (N, C)
        probs = logits.softmax(dim=-1)
        best_idx = probs.argmax(dim=-1).cpu().numpy()     # (N,)

    # Strip "a painting of " / "an artwork from " prefix for readability
    short_prompts = [p.split(" of ")[-1].split(" from ")[-1].strip() for p in prompts]
    label_names = np.array([short_prompts[i] for i in best_idx], dtype=str)

    return {
        "values": best_idx,
        "names": label_names,
        "colormap": "tab20",
        "label": "Semantic subject (CLIP zero-shot)",
        "is_categorical": True,
    }


# ── Convenience wrappers ──────────────────────────────────────────────────────

def from_subject(bundle: RepresentationBundle, **kwargs) -> dict:
    return compute_semantic_labels(bundle, prompts=SUBJECT_PROMPTS, **kwargs)


def from_mood(bundle: RepresentationBundle, **kwargs) -> dict:
    return compute_semantic_labels(bundle, prompts=MOOD_PROMPTS, **kwargs)


def from_period(bundle: RepresentationBundle, **kwargs) -> dict:
    return compute_semantic_labels(bundle, prompts=PERIOD_PROMPTS, **kwargs)
