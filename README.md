# Art Representation Explorer

A seminar project for the **RL Seminar** at the Institute for Computer Vision, University of Koblenz.

We generate and compare image representations of the [ArtBench-10](https://github.com/liaopeiyuan/artbench) dataset using three families of models, then visualise them with PCA, UMAP, and t-SNE — coloured by a choice of semantic, perceptual, or cluster-based variables.

---

## What this repo does

| Step | Script | What it produces |
|------|--------|-----------------|
| 1 | `scripts/01_download_artbench.py` | `data/artbench/{train,test}/<style>/*.jpg` |
| 2 | `scripts/02_extract_features.py` | `representations/<model>_<split>.npz` |
| 3 | `scripts/04_compute_color_profiles.py` | `representations/color_profiles_<split>.npz` |
| 4 | `scripts/03_visualize.py` | `visualizations/<stem>.png` + `.html` |

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

### 2. Download ArtBench-10

```bash
python scripts/01_download_artbench.py
```

This uses the HuggingFace `datasets` library by default (~3 GB download).

### 3. Extract features

```bash
# ResNet-50 with ImageNet weights
python scripts/02_extract_features.py \
    --model resnet50 \
    --split train \
    --out representations/resnet50_imagenet_train.npz

# CLIP ViT-B/32
python scripts/02_extract_features.py \
    --model clip_vitb32 \
    --split train \
    --out representations/clip_vitb32_train.npz

# DINOv2 ViT-B/14
python scripts/02_extract_features.py \
    --model dinov2_vitb14 \
    --split train \
    --out representations/dinov2_vitb14_train.npz
```

### 4. Pre-compute color profiles (optional, speeds up visualisation)

```bash
python scripts/04_compute_color_profiles.py --split train
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
│   ├── 02_extract_features.py
│   ├── 03_visualize.py
│   └── 04_compute_color_profiles.py
├── src/
│   ├── datasets/
│   │   └── artbench.py      # ArtBench PyTorch Dataset
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
