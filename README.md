# Art Representation Explorer

A seminar project for the **RL Seminar** at the Institute for Computer Vision, University of Koblenz.

We generate and compare image representations of the [ArtBench-10](https://github.com/liaopeiyuan/artbench) and [WikiArt](https://huggingface.co/datasets/huggan/wikiart) datasets using three families of models, then visualise them with PCA, UMAP, and t-SNE — coloured by a choice of semantic, perceptual, or cluster-based variables.

---

## What this repo does

| Step | Script | What it produces |
|------|--------|-----------------|
| 1 | `scripts/01_download_artbench.py` | `data/artbench/{train,test}/<style>/*.jpg` |
| 1 | `scripts/01_download_wikiart.py` | `data/wikiart/{train,test}/<style>/*.jpg` |
| 2 | `scripts/02_extract_features.py --dataset {artbench,wikiart}` | `representations/<dataset>_<model>_<split>.npz` |
| 3 | `scripts/04_compute_color_profiles.py --dataset {artbench,wikiart}` | `representations/<dataset>_color_profiles_<split>.npz` |
| 4 | `scripts/03_visualize.py` | `visualizations/<stem>.png` + `.html` |

### Supported datasets

| `--dataset` value | Description | Styles | Split |
|--------------------|-------------|--------|-------|
| `artbench` (default) | [ArtBench-10](https://github.com/liaopeiyuan/artbench), 60k images, 256×256 px | 10 | official train/test |
| `wikiart` | [WikiArt](https://huggingface.co/datasets/huggan/wikiart), ~81k images scraped from wikiart.org | 27 | synthesised (deterministic per-style split, `--test-fraction`, default 10%) |

### Supported models

| Alias | Architecture | Weights |
|-------|-------------|---------|
| `resnet50` | ResNet-50 | ImageNet (torchvision via timm) |
| `resnet50` + `--checkpoint` | ResNet-50 | Fine-tuned on ArtBench-10 |
| `efficientnet_b0` | EfficientNet-B0 | ImageNet |
| `vit_base_patch16_224` | ViT-B/16 | ImageNet |
| `clip_vitb32` | CLIP ViT-B/32 | OpenAI |
| `clip_vitl14` | CLIP ViT-L/14 | OpenAI |
| `clip_vitb32_laion` | CLIP ViT-B/32 | LAION-2B |
| `dinov2_vitb14` | DINOv2 ViT-B/14 | Meta AI (LVD-142M) |
| `dinov2_vitl14` | DINOv2 ViT-L/14 | Meta AI |
| `dino_vitb16` | DINO ViT-B/16 | Meta AI (ImageNet) |

### Supported color variables

| `--color` value | Description |
|-----------------|-------------|
| `label` | Art style class (10 categories) |
| `mean_hue` | Circular mean HSV hue |
| `mean_saturation` | Mean HSV saturation |
| `mean_brightness` | Mean HSV brightness |
| `colorfulness` | Hasler & Süsstrunk (2003) metric |
| `cluster` | K-Means / GMM / DBSCAN cluster id |
| `lbp_entropy` | LBP texture entropy |
| `contrast` | GLCM contrast |
| `homogeneity` | GLCM homogeneity |
| `energy` | GLCM energy |
| `semantic_subject` | CLIP zero-shot subject class |
| `semantic_mood` | CLIP zero-shot mood |
| `semantic_period` | CLIP zero-shot art period |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/art-representation-explorer.git
cd art-representation-explorer
pip install -r requirements.txt
```

### 2. Download a dataset

```bash
# ArtBench-10 (~3 GB, official Berkeley tar by default)
python scripts/01_download_artbench.py

# WikiArt (~30 GB, via the huggan/wikiart HuggingFace mirror)
python scripts/01_download_wikiart.py
```

`01_download_artbench.py` downloads directly from the official Berkeley
server by default (`--method direct`); pass `--method huggingface` to try
HuggingFace mirrors instead. WikiArt has no official archive, so
`01_download_wikiart.py` always goes through HuggingFace `datasets` and
carves out its own train/test split (`--test-fraction`, default 10%) since
the upstream dataset ships only one split.

### 3. Extract features

```bash
# ResNet-50 with ImageNet weights, on ArtBench-10
python scripts/02_extract_features.py \
    --model resnet50 \
    --dataset artbench \
    --split train \
    --out representations/artbench_resnet50_imagenet_train.npz

# CLIP ViT-B/32, on WikiArt
python scripts/02_extract_features.py \
    --model clip_vitb32 \
    --dataset wikiart \
    --split train \
    --out representations/wikiart_clip_vitb32_train.npz

# DINOv2 ViT-B/14
python scripts/02_extract_features.py \
    --model dinov2_vitb14 \
    --dataset artbench \
    --split train \
    --out representations/artbench_dinov2_vitb14_train.npz
```

`--dataset` defaults to `artbench` and picks the matching default
`--data-dir` (`data/artbench` or `data/wikiart`); pass `--data-dir` to
override it.

### 4. Pre-compute color profiles (optional, speeds up visualisation)

```bash
python scripts/04_compute_color_profiles.py --dataset artbench --split train
python scripts/04_compute_color_profiles.py --dataset wikiart --split train
```

### 5. Visualise

```bash
# UMAP coloured by art style label
python scripts/03_visualize.py \
    --repr representations/clip_vitb32_train.npz \
    --color label \
    --method umap \
    --out-dir visualizations/clip_umap_label

# t-SNE coloured by colorfulness
python scripts/03_visualize.py \
    --repr representations/resnet50_imagenet_train.npz \
    --color colorfulness \
    --color-file representations/color_profiles_train.npz \
    --method tsne

# PCA 3D, cluster colours
python scripts/03_visualize.py \
    --repr representations/dinov2_vitb14_train.npz \
    --color cluster \
    --n-clusters 15 \
    --method pca \
    --n-components 3

# Compare two models in one command
python scripts/03_visualize.py \
    --repr representations/resnet50_imagenet_train.npz \
            representations/clip_vitb32_train.npz \
    --color label \
    --method umap
```

Each run produces:
- `<out-dir>/<stem>.png`  — static matplotlib figure
- `<out-dir>/<stem>.html` — interactive Plotly scatter (open in browser)

---

## Training an autoencoder (ArtBench-10 + WikiArt)

`scripts/07_train_autoencoder.py` trains a convolutional autoencoder
(`src/models/autoencoder.py`, reconstruction loss = L1 + MSE) on the union of the
ArtBench-10 and WikiArt **train** splits and validates on the held-out **test**
splits.  It uses DDP, logs to [Weights & Biases](https://wandb.ai)
(train/val loss, PSNR, learning rate, input-vs-reconstruction grids, full config) and
writes everything into a fresh `runs/<YYYYmmdd-HHMMSS>_<name>/` directory
(`config.json`, `checkpoints/`, `autoencoder_final.pt`, `train.log`, `metrics.jsonl`,
`recon/`, `wandb/`).

```bash
# 4 GPUs on one JURECA devel node (account jinm16): smoke test, then training
sbatch scripts/slurm/train_autoencoder.sbatch

# continue after the 2 h devel limit (same run dir, same wandb run)
RESUME=runs/<stamp>_ae_union sbatch scripts/slurm/train_autoencoder.sbatch

# by hand
torchrun --standalone --nproc_per_node=4 scripts/07_train_autoencoder.py --epochs 30
```

One-time: `wandb login` on the login node (stores the key in `~/.netrc`).  Compute nodes
without internet fall back to offline mode automatically; sync afterwards from the
login node with `wandb sync runs/<run>/wandb/offline-run-*`.  Force a mode with
`--wandb {online,offline,disabled}` or `WANDB_MODE`.

## SimCLR representations and augmentation sets

Contrastive self-supervised representations (SimCLR, Chen et al. 2020) of ArtBench-10 and WikiArt, and a family of SimCLR
models trained with different **augmentation sets** to study how the augmentations shape the representation.

| Model | Weights | File slug (`representations/<dataset>_<slug>_{train,test,all}.npz`) |
|-------|---------|------|
| `simclr_imagenet` | published ImageNet-1k SimCLR v1 ResNet-50 (1x), frozen ([lightly-ai/simclrv1-imagenet1k-resnet50-1x](https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x), converted official checkpoint; ImageNet itself is not available locally, so no ImageNet training here) | `simclr_imagenet` |
| `simclr` + `--checkpoint` | ResNet-50 SimCLR trained **from scratch** here on the train split of ONE dataset with augmentation set `<aug>` | `simclr_<artbench\|wikiart>_<aug>` |

*Embedding* = global-average-pooled ResNet-50 output (2048-D, before the projection head). Input: Resize 256, CenterCrop 224
(the published ImageNet weights expect raw [0, 1] pixels, ours ImageNet mean/std).
*Recipe*: ResNet-50 + 2-layer projection head (2048-2048-128), NT-Xent (temperature 0.2) over the whole global batch (4 GPUs x 256 = 1024
images, negatives gathered across GPUs with gradients), SyncBN, LARS (base lr 0.3 x batch/256, weight decay 1e-6), 10 warm-up epochs + cosine,
224 px views, bf16, **200 epochs** for every set and dataset (so variants are comparable).
Monitors logged to wandb every 10 epochs: kNN (k=20) and linear-probe accuracy on the style labels (10k train -> 4k test images), loss,
positive-pair accuracy, and example view pairs (`samples/aug_views`, also in `runs/<run>/views/`).

```bash
python scripts/13_train_simclr.py --list-aug-sets
# 4 GPUs (smoke test first, then the run; self-resubmitting with CHAIN=1; devel 2 h limit, or any partition with --partition/--time)
DATASET=artbench AUG_SET=baseline TRAIN_ARGS="--epochs 200" CHAIN=1 NEXT_TIME=02:00:00 sbatch scripts/slurm/train_simclr.sbatch
RESUME=runs/<stamp>_simclr_artbench_baseline sbatch scripts/slurm/train_simclr.sbatch
# embeddings (train + test + merged), then the explorer pages
ENTRIES="imagenet@artbench,wikiart runs/<run>/simclr_final.pt@artbench" sbatch scripts/slurm/embed_simclr.sbatch
python scripts/14_preview_augmentations.py --dataset wikiart      # grids of all sets -> visualizations/aug_sets/
```

Augmentation sets are defined in [`configs/aug_sets.yaml`](configs/aug_sets.yaml) (ops: `src/augmentations/ops.py`) and selected by `--aug-set`;
each run's `config.json` stores the set's description and op list. Every view = crop + the listed ops, applied independently per view
(`p` = probability per view).

| `--aug-set` | Crop | Photometric / other | Purpose |
|---|---|---|---|
| `baseline` | RRC scale 0.08-1 | flip 0.5; colour jitter (0.8, 0.8, 0.8, 0.2) p 0.8; grayscale 0.2; blur sigma 0.1-2 p 0.5 | standard SimCLR v1 reference |
| `color_heavy` | RRC 0.08-1 | flip; colour jitter (0.9, 0.9, 0.9, **0.5**) p 1.0; channel shuffle 0.5; channel drop 0.3; grayscale 0.5; solarize 0.5; posterize 0.3; blur 0.5 | invariance to colour / palette |
| `edge_texture` | RRC **0.05-0.4** (local texture crops) | flip; weak jitter (0.4, 0.4, 0.4, 0.1); grayscale 0.2; sharpen **or** strong blur (sigma 2-6) p 0.5; one of Sobel / Laplacian edge map, high-pass, edge-enhance p 0.5; patch shuffle (2x2-4x4 tiles) p 0.5 | reliance on edge / texture statistics instead of layout and palette |
| `rotation` | RRC 0.08-1 + rotation **0-360 deg** (corner-free) | hflip + vflip; baseline colour / blur | rotation invariance |
| `geometry_heavy` | RRC **0.02-1**, ratio 0.5-2 | flip; rotation +-30 deg p 0.5; perspective 0.5 p 0.7; shear 20 deg p 0.5; baseline colour / blur | viewpoint / layout invariance |
| `no_color` | RRC 0.08-1 | flip; brightness/contrast jitter; **grayscale p 1.0**; blur 0.5 | no colour information at all |
| `blur_heavy` | RRC 0.08-1 | flip; baseline colour; **blur sigma 2-8 p 1.0**; noise 0.3 | destroys fine texture/edges; colour + low-frequency layout remain |
| `crop_only` | RRC 0.08-1 | flip only | control: colour histogram becomes a shortcut |

Quantitative comparison of the variants (`scripts/17_compare_simclr.py`, `18_investigate_nocolor.py`, `19_grayscale_eval.py`; results in `docs/`):
colour / texture predictability from the embedding, invariance to grayscale / hue / rotation / blur, CKA to the baseline
(`docs/simclr_comparison_<dataset>.md`, `docs/simclr_invariance_<dataset>.md`), why the always-grayscale model still encodes colour
(`docs/simclr_nocolor_investigation_artbench.md`) and style accuracy under grayscale input (`docs/simclr_grayscale_<dataset>.md`).

Training details of every run: `runs/<stamp>_simclr_<dataset>_<aug set>/` (config.json, metrics.jsonl, views/, checkpoints/, status.json,
simclr_final.pt). Results table (run dir, wandb run, probe accuracy): `docs/simclr_runs.md`.

---

## Explicit images are hidden in the explorer by default

Paintings with nudity are flagged automatically and **hidden by default** in every explorer page, the overview and the feature-map
viewer: they are not drawn, cannot be hovered, selected or listed, and their image files are never requested. The header checkbox
"show explicit images" (also in the overview's hero box) turns the filter off; the choice is stored in the browser (one key,
`artExplorer.showExplicit`) and shared by all pages.

A flag is raised when ANY of these signals fires (so it errs on the side of hiding): NudeNet >= 0.6, CLIP ViT-L/14 zero-shot >= 0.9,
a logistic-regression probe on CLIP ViT-L/14 features trained on WikiArt's `nude_painting` genre label (>= 0.40), the dataset's own
label (WikiArt genre `nude_painting`; ArtBench titles with nude / naked / ...), or the CLIP ViT-B/32 "nudity" concept score that
the maps can be coloured by (>= 2.5, about the top 7 %). Currently about 9.7 % of ArtBench-10 and 9.6 % of WikiArt are flagged.
The flags are imperfect: some nudes are missed, and some statues, abstract body forms and clothed figures are hidden.

```bash
pip install nudenet                                    # also installs onnxruntime
python scripts/15_flag_nudity.py                       # detectors on all images (uses the explorer thumbnails) -> data/nudity/
python scripts/16_train_nudity_probe.py                # score every signal against the WikiArt genre labels, train the probe
python scripts/15_flag_nudity.py --flags-only          # re-apply thresholds (--nn-thr / --clip-thr / --probe-thr / --clip32-thr)
python scripts/rerender_explorer.py                    # put the flags into the existing pages (no recomputation)
```

`scripts/08_build_explorer.py` reads `data/nudity/flags.npz` for every new page (`src/metadata/nudity.py`), so pages of new models
need no extra step as long as the flags exist. `data/` is not versioned: regenerate the flags with the commands above.
**The hiding is done in the page only**: the flagged thumbnail files are still in `visualizations/explorer/thumbs/` (and in
`featuremaps/data/`), so do not publish those folders as they are if the audience must not be able to reach them.

---

## Using representations programmatically

```python
from src.models.base_extractor import RepresentationBundle
from src.reduction.dimensionality import reduce
from src.visualization.scatter import visualize
from src.coloring.label_colors import from_labels

# Load a saved representation
bundle = RepresentationBundle.load("representations/clip_vitb32_train.npz")
print(bundle)  # RepresentationBundle(n=50000, dim=512, model='clip_vit_b_32_openai', …)

# Project to 2D with UMAP
proj = reduce(bundle.vectors, method="umap", n_components=2)

# Colour by label, save PNG + HTML
colorvar = from_labels(bundle)
visualize(proj.coords, colorvar, paths=bundle.paths,
          title="CLIP ViT-B/32  ·  UMAP  ·  Art style",
          out_dir="visualizations", stem="clip_umap_label")
```

---

## Repository layout

```
art-representation-explorer/
├── scripts/                 # Entry-point CLI scripts
│   ├── 01_download_artbench.py
│   ├── 01_download_wikiart.py
│   ├── 02_extract_features.py
│   ├── 03_visualize.py
│   └── 04_compute_color_profiles.py
├── src/
│   ├── datasets/
│   │   ├── artbench.py      # ArtBench PyTorch Dataset
│   │   ├── wikiart.py       # WikiArt PyTorch Dataset
│   │   └── __init__.py      # `--dataset` name → Dataset class registry
│   ├── models/
│   │   ├── base_extractor.py   # Abstract base + RepresentationBundle
│   │   ├── cnn_extractor.py    # ResNet / EfficientNet / ViT via timm
│   │   ├── clip_extractor.py   # CLIP via open_clip
│   │   └── ssl_extractor.py    # DINOv2 / DINO via torch.hub
│   ├── reduction/
│   │   └── dimensionality.py   # PCA / UMAP / t-SNE wrappers
│   ├── visualization/
│   │   └── scatter.py          # Matplotlib + Plotly scatter
│   └── coloring/
│       ├── label_colors.py     # Class label → colour
│       ├── color_profile.py    # Hue, saturation, brightness, colorfulness
│       ├── cluster_colors.py   # K-Means / GMM / DBSCAN
│       ├── texture_colors.py   # LBP entropy, GLCM features
│       └── semantic_colors.py  # CLIP zero-shot semantic labels
├── data/                    # Downloaded dataset (git-ignored)
├── representations/         # Saved .npz bundles (git-ignored)
├── visualizations/          # Output PNG + HTML files (git-ignored)
├── notebooks/               # Jupyter exploration notebooks
├── requirements.txt
└── pyproject.toml
```

---

## Adding your own model

1. Subclass `BaseExtractor` in `src/models/`:

```python
from src.models.base_extractor import BaseExtractor

class MyExtractor(BaseExtractor):
    @property
    def model_name(self) -> str:
        return "my_model"

    @property
    def layer_name(self) -> str:
        return "penultimate"

    def _build_model(self):
        ...  # return nn.Module

    def _get_transform(self):
        ...  # return torchvision transform

    def _forward(self, batch):
        ...  # return (B, D) tensor
```

2. Extract and save:

```python
extractor = MyExtractor()
bundle = extractor.extract(dataset, save_path="representations/my_model_train.npz")
```

3. Visualise exactly like any other bundle.

## Adding your own color variable

Any dict with the following keys works as a `colorvar`:

```python
colorvar = {
    "values":         np.array(...),   # shape (N,) — numeric or int
    "names":          np.array(...),   # shape (N,) — human-readable str
    "colormap":       "viridis",       # matplotlib colormap name
    "label":          "My variable",   # axis / legend label
    "is_categorical": False,           # True → discrete legend
}
```

Pass it directly to `visualize()` or `03_visualize.py` via `--color`.

---

## Citation

If you use ArtBench-10 in your work, please cite the original paper:

```bibtex
@misc{liao2022artbench,
    title     = {The ArtBench Dataset: Benchmarking Generalisation in Image Generation},
    author    = {Peiyuan Liao and Xiuyu Li and Xihui Liu and Kurt Keutzer},
    year      = {2022},
    eprint    = {2206.11404},
    archivePrefix = {arXiv},
}
```

---

## License

MIT — see `LICENSE`.
