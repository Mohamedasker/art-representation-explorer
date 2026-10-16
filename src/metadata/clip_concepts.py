"""
CLIP zero-shot content concepts (people, nudity, religion, scene type, ...).

Scores are computed from the *existing* CLIP image embeddings (``representations/*_clip_*.npz``),
so no images are re-read: only the text side of CLIP is run here.

  * numeric score per concept  = 100 * (cos(image, concept prompts) - cos(image, "a painting"))
  * categorical groups          = argmax of the softmax over the group's prompts (logit scale 100)
"""

from __future__ import annotations

import numpy as np
import torch

TEMPLATES = ["a painting of {}", "an artwork showing {}", "a photo of a painting of {}"]
NULL_PROMPT = "a painting"

PEOPLE = "CLIP: people & figures"
SCENE = "CLIP: scene & content"
TECH = "CLIP: technique, medium & frame"

AESTH = "CLIP: aesthetics (subjective)"
MOOD = "CLIP: mood & sensations"

# key -> (label, ui group, phrases)
CONCEPTS: dict[str, tuple[str, str, list[str]]] = {
    "woman":       ("woman", PEOPLE, ["a woman", "a young woman", "a lady"]),
    "man":         ("man", PEOPLE, ["a man", "an old man", "a gentleman"]),
    "child":       ("child", PEOPLE, ["a child", "a baby", "children playing"]),
    "crowd":       ("group of people", PEOPLE, ["a group of people", "a crowd", "many people"]),
    "face":        ("portrait / face", PEOPLE, ["a portrait of a person", "a close-up of a face"]),
    "nude":        ("nudity", PEOPLE, ["a nude person", "a naked body", "a nude woman"]),
    "elderly":     ("elderly person", PEOPLE, ["an old person with wrinkles", "an elderly man or woman"]),
    "smiling":     ("smiling / laughing", PEOPLE, ["a smiling person", "people laughing"]),
    "gaze":        ("looking at the viewer", PEOPLE, ["a person looking straight at the viewer"]),
    "armor":       ("armour / uniform", PEOPLE, ["a knight in armour", "a soldier in uniform"]),
    "royal":       ("royalty / nobility", PEOPLE, ["a king or queen with a crown", "a nobleman in rich clothes"]),
    "christian":   ("Christian / religious", SCENE, ["Jesus Christ", "the Virgin Mary and the Christ child",
                                                     "saints and angels", "a biblical scene", "a church altarpiece"]),
    "mythology":   ("mythology / allegory", SCENE, ["a scene from Greek mythology", "gods and goddesses", "an allegory"]),
    "city":        ("city / street", SCENE, ["a city street", "a town with houses", "a cityscape"]),
    "architecture": ("architecture", SCENE, ["a building", "a cathedral or church exterior", "a palace"]),
    "landscape":   ("landscape", SCENE, ["a landscape with fields and hills", "mountains and trees", "a river valley"]),
    "sea":         ("sea / boats", SCENE, ["the sea with ships", "a harbour with boats", "waves on a beach"]),
    "interior":    ("interior", SCENE, ["an interior of a room", "a person in a room"]),
    "still_life":  ("still life", SCENE, ["a still life of objects on a table", "fruit and a bottle"]),
    "flowers":     ("flowers", SCENE, ["flowers in a vase", "a flower garden"]),
    "war":         ("battle / war", SCENE, ["a battle with soldiers", "a war scene"]),
    "abstract":    ("abstract", SCENE, ["an abstract composition of shapes and colours", "abstract geometric forms"]),
    "night":       ("night", SCENE, ["a night scene", "moonlight in the dark"]),
    "winter":      ("snow / winter", SCENE, ["a snowy winter landscape", "snow"]),
    "text":        ("text / ornament", SCENE, ["a decorative pattern", "an ornament or design", "calligraphy or text"]),
    "sketch":      ("sketch / drawing", TECH, ["a pencil sketch", "a line drawing on paper", "an ink drawing"]),
    "bw":          ("black & white", TECH, ["a black and white image", "a monochrome grayscale picture"]),
    "sepia":       ("sepia / monotone", TECH, ["a sepia toned image", "a monochrome blue picture"]),
    "print":       ("print / engraving", TECH, ["a woodblock print", "an engraving or etching"]),
    "photo":       ("photograph", TECH, ["a photograph", "a black and white photograph"]),
    "frame":       ("picture frame visible", TECH, ["an ornate gold picture frame", "a framed painting hanging on a wall"]),
    "animals":     ("animals (any)", SCENE, ["a horse", "a dog", "animals in a field", "birds"]),
}

# key -> (label, ui group, {class: phrase}, templates or None for TEMPLATES)
# exclusive groups shown as categorical colourings (argmax of the softmax over the classes)
GROUPS: dict[str, tuple[str, str, dict[str, str | list[str]], list[str] | None]] = {
    "clip_people": ("CLIP: people (gender / age)", PEOPLE, {
        "woman": "a woman", "man": "a man", "child": "a child", "group of people": "a group of people",
        "no people": "a scene without any people"}, None),
    "clip_count": ("CLIP: number of people", PEOPLE, {
        "none": "a scene without any people", "one person": "a single person", "two people": "two people",
        "small group (3-6)": "a small group of four people", "crowd": "a large crowd of people"}, None),
    "clip_scene": ("CLIP: scene type", SCENE, {
        "portrait": "a portrait", "landscape": "a landscape", "cityscape": "a city or street scene",
        "seascape": "a seascape with boats", "still life": "a still life", "interior": "an interior scene",
        "religious": "a religious scene", "mythological": "a mythological scene", "battle": "a battle scene",
        "abstract": "an abstract painting", "sketch / drawing": "a sketch or drawing",
        "everyday life": "a scene of everyday life", "animals": "animals", "nude": "a nude figure"}, None),
    "clip_medium": ("CLIP: medium", TECH, {
        "oil painting": "an oil painting on canvas", "watercolour / gouache": "a watercolor painting",
        "pastel / chalk": "a pastel drawing", "pencil / charcoal sketch": "a pencil or charcoal sketch",
        "pen & ink drawing": "a pen and ink drawing", "woodblock print": "a japanese woodblock print",
        "engraving / etching": "an engraving or etching", "fresco / mural": "a fresco or wall painting",
        "sculpture": "a sculpture", "photograph": "a photograph",
        "graphic design / poster": "a graphic design poster or illustration"}, ["{}", "a photo of {}", "an artwork, {}"]),
    "clip_colormode": ("CLIP: colour mode", TECH, {
        "colourful": "a colourful painting", "black & white": "a black and white image",
        "sepia / one colour": "a sepia or monochrome image"}, None),
    "clip_frame": ("CLIP: frame", TECH, {
        "framed": "a painting in an ornate golden frame", "no frame": "a painting filling the whole image with no frame"}, None),
}


CONCEPTS.update({
    "mood_fear":     ("fear / horror", MOOD, ["a frightening horror scene", "a dark nightmare"]),
    "mood_love":     ("love / romance", MOOD, ["a romantic scene of love", "lovers embracing"]),
    "mood_awe":      ("awe / sublime", MOOD, ["a sublime awe-inspiring vista", "a dramatic stormy sky"]),
    "mood_lonely":   ("loneliness", MOOD, ["a lonely solitary figure", "isolation and emptiness"]),
    "mood_mystery":  ("mystery / dreamlike", MOOD, ["a mysterious dreamlike scene", "a surreal fantasy"]),
    "mood_humour":   ("humour / satire", MOOD, ["a funny satirical caricature", "a humorous scene"]),
})

GROUPS.update({
    "clip_mood": ("CLIP: dominant mood", MOOD, {
        "joy / happiness": "a joyful happy scene", "sadness / melancholy": "a sad melancholic scene",
        "peace / calm": "a peaceful calm serene scene", "aggression / violence": "an aggressive violent scene",
        "fear / dread": "a frightening scene full of dread", "love / tenderness": "a tender loving scene",
        "awe / drama": "a dramatic awe-inspiring scene", "loneliness": "a lonely isolated scene",
        "neutral / none": ["a plain portrait", "a plain landscape", "a plain still life", "a decorative pattern"]}, None),
})

# opposing prompt pairs: score = 100 * (cos(image, positive) - cos(image, negative)); high = first label
# key -> (label, ui group, positive phrases, negative phrases)
PAIRS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "beauty":     ("beautiful \u2194 ugly", AESTH,
                   ["a beautiful painting", "a gorgeous artwork", "a stunning masterpiece"],
                   ["an ugly painting", "an unattractive artwork", "a hideous picture"]),
    "creative":   ("creative / original \u2194 mediocre / unoriginal", AESTH,
                   ["a highly creative and original artwork", "an innovative, imaginative painting"],
                   ["a mediocre, unoriginal artwork", "a boring, derivative painting"]),
    "artistic":   ("artistic / masterful \u2194 generic / amateur", AESTH,
                   ["a masterful artistic painting", "a painting by a great master"],
                   ["a generic amateur painting", "a clumsy amateur drawing"]),
    "quality":    ("high quality \u2194 low quality", AESTH,
                   ["a high quality professional artwork", "a refined, well executed painting"],
                   ["a low quality poor artwork", "a crude, badly executed painting"]),
    "harmony":    ("harmonious \u2194 chaotic", AESTH,
                   ["a harmonious, balanced composition", "an elegant, orderly painting"],
                   ["a chaotic, messy composition", "a cluttered, disordered painting"]),
    "happy_sad":  ("happy \u2194 sad", MOOD, ["a happy cheerful painting", "a joyful scene"],
                   ["a sad gloomy painting", "a melancholic, sorrowful scene"]),
    "peace_aggr": ("peaceful \u2194 aggressive", MOOD, ["a peaceful, calm painting", "a serene quiet scene"],
                   ["an aggressive, violent painting", "an angry, brutal scene"]),
    "warm_cold":  ("warm / cozy \u2194 cold / bleak", MOOD, ["a warm cozy inviting painting", "a bright warm scene"],
                   ["a cold bleak desolate painting", "a harsh dark chilly scene"]),
    "calm_energy": ("calm \u2194 energetic / dynamic", MOOD, ["a still, calm painting", "a motionless quiet scene"],
                    ["an energetic dynamic painting", "a scene full of movement and action"]),
}


def _encode_text(texts: list[str], model_name: str, pretrained: str, device: str = "cpu") -> np.ndarray:
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            f = model.encode_text(tok(texts[i:i + 128]).to(device))
            out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out)


def text_embeddings(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """Returns (concept[C,D], null[D], groups{key:(rows, class_of_row)}, pairs[P,2,D]) — all L2-normalised."""
    texts, spans = [], {}

    def add(key, phrases, templates=TEMPLATES):
        start = len(texts)
        for ph in phrases:
            if templates is None:
                texts.append(ph)
            else:
                texts.extend(t.format(ph) for t in templates)
        spans[key] = (start, len(texts))

    add("null", [NULL_PROMPT], templates=None)
    for k, (_, _, phrases) in CONCEPTS.items():
        add(k, phrases)
    for k, (_, _, classes, templates) in GROUPS.items():
        for cname, ph in classes.items():
            add(f"{k}/{cname}", [ph] if isinstance(ph, str) else ph,
                templates=templates if templates is not None else TEMPLATES)
    for k, (_, _, pos, neg) in PAIRS.items():
        add(f"pair/{k}/pos", pos, templates=None)
        add(f"pair/{k}/neg", neg, templates=None)
    emb = _encode_text(texts, model_name, pretrained)

    def mean(key):
        a, b = spans[key]
        v = emb[a:b].mean(0)
        return v / np.linalg.norm(v)

    concept = np.stack([mean(k) for k in CONCEPTS])
    # a class with several phrases keeps one row per phrase; score() takes the max over a class's rows
    groups = {}
    for k, g in GROUPS.items():
        rows, owner = [], []
        for ci, (c, ph) in enumerate(g[2].items()):
            n = 1 if isinstance(ph, str) else len(ph)
            a, b = spans[f"{k}/{c}"]
            per = (b - a) // n
            for j in range(n):
                v = emb[a + j * per:a + (j + 1) * per].mean(0)
                rows.append(v / np.linalg.norm(v)); owner.append(ci)
        groups[k] = (np.stack(rows), np.array(owner))
    pairs = np.stack([np.stack([mean(f"pair/{k}/pos"), mean(f"pair/{k}/neg")]) for k in PAIRS])   # [P, 2, D]
    return concept, mean("null"), groups, pairs


def score(image_emb: np.ndarray, concept: np.ndarray, null: np.ndarray, groups: dict, pairs: np.ndarray):
    """image_emb: [N, D] (any norm). Returns (concept_scores[N,C], {group: argmax[N] int16}, pair_scores[N,P])."""
    x = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    num = 100.0 * (x @ concept.T - (x @ null)[:, None])
    pair = 100.0 * (np.einsum("nd,pd->np", x, pairs[:, 0]) - np.einsum("nd,pd->np", x, pairs[:, 1]))
    cats = {}
    for k, (emb, owner) in groups.items():
        logits = 100.0 * (x @ emb.T)
        n = owner.max() + 1
        per_class = np.stack([logits[:, owner == c].max(1) for c in range(n)], axis=1)
        cats[k] = per_class.argmax(1).astype(np.int16)
    return num.astype(np.float32), cats, pair.astype(np.float32)
