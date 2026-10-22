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

## Supervised ViT-B/16 (ImageNet vs. trained on the art labels)

A plain Vision Transformer (timm `vit_base_patch16_224`, 224 px, 85.9M parameters) trained on the style / genre / artist labels of
ArtBench-10 + WikiArt exactly like the supervised ResNet-50 (`scripts/10_train_cnn.py`): train splits of both datasets jointly,
three linear heads on the encoder output, the same loss weights (style 1, genre 0.5, artist 0.5, label smoothing 0.1), RandomResizedCrop
(0.35-1) + flip, AdamW + cosine schedule, bf16, 4 GPUs DDP, held-out evaluation on both test splits, wandb, `runs/<stamp>_<name>/`
run directories (resumable). Differences forced by the architecture: 224 px instead of 256, mean = std = 0.5 normalisation (what the
timm ViT weights expect), stochastic depth 0.1, a lower peak LR (AdamW, global batch 256: 2.5e-4 from scratch).

| Model (`--model`) | Weights | File slug (`representations/<dataset>_<slug>_{train,test,all}.npz`) |
|-------|---------|------|
| `vit_imagenet` | timm `vit_base_patch16_224.augreg_in1k` (ImageNet-1k supervised only, no ImageNet-21k), frozen | `vit_imagenet` |
| `vit_supervised` + `--checkpoint` | trained here **from scratch** (`--init scratch`; parallel to `cnn_supervised`) | `vit_supervised` |
| `vit_supervised` + `--checkpoint` | **ImageNet init, fine-tuned** on the art labels (`--init imagenet`, peak LR 5e-5, layer-wise LR decay 0.75) | `vit_supervised_ft` |

*Embedding* = the final-layer **CLS token** after the encoder's last LayerNorm (768-D, the input of the heads), for all three. `--vit-pool mean`
gives the mean of the 196 patch tokens, `both` their concatenation (files get a `_mean` / `_both` suffix). Pre-processing: resize 224,
centre crop, normalise with 0.5/0.5; vectors are not standardised.

```bash
# 4 GPUs (smoke test first; INIT=scratch|imagenet; self-resubmitting on a normal partition with CHAIN=1)
INIT=scratch  SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_vit.sbatch
INIT=imagenet SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_vit.sbatch
RESUME=runs/<stamp>_vit_scratch sbatch scripts/slurm/train_vit.sbatch
# embeddings (train + test + merged)
python scripts/02_extract_features.py --model vit_imagenet --dataset artbench --split all
python scripts/02_extract_features.py --model vit_supervised --checkpoint runs/<run>/vit_final.pt --dataset wikiart --split all
```

The ImageNet weights are read from the Hugging Face cache (fetch them once on a login node, compute nodes have no internet):
`python -c "import timm; timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)"`.
Test accuracies and run directories: `docs/vit_runs.md`.

---

## Autoregressive models (iGPT, VQ tokens + GPT)

Two generative autoregressive families trained from scratch on ArtBench-10 + WikiArt (train splits jointly), shown as the model type "Autoregressive transformer" with a
Weights menu (`docs/ar_runs.md`: sizes, losses, layer curves, samples, wandb links):

| Model (`--model`) | What | Embedding | File slug |
|---|---|---|---|
| `ar_igpt` + `--checkpoint` | **iGPT**: 76.7M GPT on 32 x 32 px images, one pixel = one of 512 palette colours (`scripts/26_prepare_igpt.py`, `27_train_igpt.py`) | mean over the 1024 positions of the hidden states of ONE intermediate block (layer picked with a style probe, `24_probe_ar_layers.py`), 512-D | `ar_igpt` |
| `ar_gpt` + `--checkpoint` | VQ tokenizer (128 px -> 16 x 16 tokens, 4096 codes; `21_train_vq.py`, `22_tokenize_ar.py`) + 160M GPT (`23_train_ar.py`, optional dataset / style prefix tokens) | same recipe on the tokens, 1024-D | `ar_gpt` |
| `ar_vq` + `--checkpoint` | tokenizer only (baseline) | quantised latents pooled to 4 x 4, 512-D | `ar_vq` |

The real image is tokenised with fixed preprocessing (no sampling) and fed through the transformer in eval mode with "no label" prefix tokens; `--ar-layer N` overrides the probed layer.

```bash
python scripts/20_prepare_ar_data.py && python scripts/26_prepare_igpt.py          # pixel cache (128 px) and iGPT palette / tokens, once, login node
IGPT_EPOCHS=14 CHAIN=1 sbatch scripts/slurm/train_igpt.sbatch                      # iGPT, 4 GPUs (devel: 2 h segments, self-resubmitting)
SKIP_SMOKE=1 CHAIN=1 NEXT_AR=1 sbatch --partition=dc-gpu --time=04:00:00 scripts/slurm/train_vq.sbatch   # VQ tokenizer, then tokenising + GPT training
python scripts/24_probe_ar_layers.py --checkpoint runs/<run>/checkpoints/best.pt   # layer-by-layer style probe -> layer_probe.json
IGPT_RUN=runs/<igpt run> AR_RUN=runs/<vq-gpt run> sbatch scripts/slurm/embed_ar.sbatch   # embeddings (train/test/all), explorer pages
```

---

## Explicit images are hidden in the explorer by default

## CLIP gently fine-tuned on ArtBench / WikiArt (fit the data without forgetting)

Two separate runs of the OpenAI CLIP ViT-B/32 (`ViT-B-32-quickgelu`, the exact model): one on ArtBench-10 train, one on WikiArt
train (`scripts/28_train_clip_ft.py`; the test splits are only used for monitoring). Recipe, chosen to adapt the image tower to the art
data while keeping CLIP's knowledge:

* contrastive loss between the images and captions built from the labels ("a landscape painting by Vincent Van Gogh, in the post
  impressionism style"; ArtBench has style + artist only, WikiArt genre + artist + style); captions shared by several images of a batch
  are not negatives of each other (`--no-style-in-caption` leaves the style out, then it is only used for evaluation),
* **text tower and logit scale frozen**: the caption embeddings are a fixed table in CLIP's original text space, the image tower is pulled
  towards it (also means the text side costs nothing during training),
* peak LR 1e-5 (not scaled), layer-wise LR decay 0.8, warm-up 200 steps, cosine, 5 epochs, global batch 512 (4 x 128), bf16,
* **WiSE-FT**: `--clip-alpha a` interpolates the fine-tuned and the original image tower (`a * fine-tuned + (1 - a) * original`), e.g. 0.5.

| Model (`--model`) | Weights | File slug (`representations/<dataset>_<slug>_{train,test,all}.npz`) |
|-------|---------|------|
| `clip_vitb32_qgelu` | OpenAI CLIP, frozen (exact model) | `clip_vitb32_qgelu` |
| `clip_ft` + `--checkpoint runs/<run>/clip_ft_final.pt` | fine-tuned on ArtBench / WikiArt (`--clip-alpha 1`) | `clip_ft_artbench` / `clip_ft_wikiart` |
| `clip_ft` + `--clip-alpha 0.5` | 50 % interpolation with the original | `clip_ft_<dataset>_wise50` |

Embedding = L2-normalised CLIP image feature (512-D), same pre-processing as the frozen extractor. Every model is embedded on **both**
datasets (cross-dataset: how much does the ArtBench-tuned model change the WikiArt space?).

```bash
# 4 GPUs (smoke test first; self-resubmitting on a normal partition with CHAIN=1)
DATASET=artbench sbatch scripts/slurm/train_clip_ft.sbatch
DATASET=wikiart  sbatch scripts/slurm/train_clip_ft.sbatch
# embeddings (frozen + alpha 1.0 / 0.5 of both runs), docs/clip_ft_eval.md, explorer pages
CLIP_FT_ARTBENCH=runs/<stamp>_clip_ft_artbench CLIP_FT_WIKIART=runs/<stamp>_clip_ft_wikiart sbatch scripts/slurm/embed_clip_ft.sbatch
```

Monitoring (`metrics.jsonl` / wandb; step 0 is the original CLIP): `val/caption_acc` (image -> its caption among all captions),
`val/zs_style` (zero-shot style accuracy), `generic/cifar100`, `generic/cifar10` (zero-shot on CIFAR test sets in `data/generic`,
the "did it forget?" probe; download once on a login node with torchvision). `scripts/29_eval_clip_ft.py` writes `docs/clip_ft_eval.md`:
zero-shot style, CIFAR and kNN accuracy of the original and all fine-tuned models, with the test images that have a near-duplicate
(CLIP cosine >= 0.95) in the other dataset's train split left out of the cross-dataset numbers.

---

## GAN (StyleGAN2) representations

A StyleGAN2 trained from scratch on the **train splits of ArtBench-10 and WikiArt jointly** (123,143 paintings, unconditional, 128 px;
`scripts/20_train_gan.py`, model in `src/models/gan.py`). Pure PyTorch, no custom CUDA ops (modulated convolutions in the non-fused form,
the FIR resampling filters written with avg-pooling / bilinear interpolation), skip-architecture generator (23M parameters) and residual
discriminator with minibatch-stddev (24M). Regularisation as in StyleGAN2-ADA: adaptive discriminator augmentation, lazy R1, path-length
regularisation, G EMA. 4 GPUs (gradients all-reduced by hand because of the double back-prop), bf16 autocast, wandb,
`runs/<stamp>_gan_union/` (config, `checkpoints/`, `samples/` grids of the EMA generator with fixed latents, `metrics.jsonl`, `gan_final.pt`),
resumable and self-chaining like the other trainers. A FID-like number (torchvision Inception-v3 features, 5000 generated vs 5000 held-out
real images, not comparable to published FIDs) is logged every 1000 kimg. Run, settings, stability and samples: `docs/gan_runs.md`.

| Model (`--model`) | What is embedded | File slug (`representations/<dataset>_<slug>_{train,test,all}.npz`) |
## Autoregressive models (iGPT, VQ tokens + GPT)

Two generative autoregressive families trained from scratch on ArtBench-10 + WikiArt (train splits jointly), shown as the model type "Autoregressive transformer" with a
Weights menu (`docs/ar_runs.md`: sizes, losses, layer curves, samples, wandb links):

| Model (`--model`) | What | Embedding | File slug |
|---|---|---|---|
| `ar_igpt` + `--checkpoint` | **iGPT**: 76.7M GPT on 32 x 32 px images, one pixel = one of 512 palette colours (`scripts/26_prepare_igpt.py`, `27_train_igpt.py`) | mean over the 1024 positions of the hidden states of ONE intermediate block (layer picked with a style probe, `24_probe_ar_layers.py`), 512-D | `ar_igpt` |
| `ar_gpt` + `--checkpoint` | VQ tokenizer (128 px -> 16 x 16 tokens, 4096 codes; `21_train_vq.py`, `22_tokenize_ar.py`) + 160M GPT (`23_train_ar.py`, optional dataset / style prefix tokens) | same recipe on the tokens, 1024-D | `ar_gpt` |
| `ar_vq` + `--checkpoint` | tokenizer only (baseline) | quantised latents pooled to 4 x 4, 512-D | `ar_vq` |

The real image is tokenised with fixed preprocessing (no sampling) and fed through the transformer in eval mode with "no label" prefix tokens; `--ar-layer N` overrides the probed layer.

```bash
python scripts/20_prepare_ar_data.py && python scripts/26_prepare_igpt.py          # pixel cache (128 px) and iGPT palette / tokens, once, login node
IGPT_EPOCHS=14 CHAIN=1 sbatch scripts/slurm/train_igpt.sbatch                      # iGPT, 4 GPUs (devel: 2 h segments, self-resubmitting)
SKIP_SMOKE=1 CHAIN=1 NEXT_AR=1 sbatch --partition=dc-gpu --time=04:00:00 scripts/slurm/train_vq.sbatch   # VQ tokenizer, then tokenising + GPT training
python scripts/24_probe_ar_layers.py --checkpoint runs/<run>/checkpoints/best.pt   # layer-by-layer style probe -> layer_probe.json
IGPT_RUN=runs/<igpt run> AR_RUN=runs/<vq-gpt run> sbatch scripts/slurm/embed_ar.sbatch   # embeddings (train/test/all), explorer pages
```

---

## DINO v1 trained on ArtBench / WikiArt

DINO v1 (Caron et al. 2021, self-distillation without labels) ViT-B/16, trained here on the train split of ONE dataset at a time
(`scripts/29_train_dino.py`, model in `src/models/dino.py`): student + EMA teacher (momentum cosine 0.996 -> 1), multi-crop (2 global
224 px crops scale 0.4-1, 6 local 96 px crops scale 0.05-0.4), colour jitter / grayscale / blur / solarisation, centring + sharpening
(teacher temperature 0.04 -> 0.07, student 0.1), 3-layer MLP head with weight-normalised last layer, AdamW with linearly scaled LR
(warm-up + cosine), weight decay 0.04 -> 0.4, stochastic depth 0.1, per-parameter gradient clipping, bf16, 4 A100 (DDP, global batch 256,
~950 img/s). Data loading reads uint8 256 px caches (`scripts/29_cache_images.py`, `data/cache256/`, git-ignored).

| Weights (`representations/<dataset>_<slug>_{train,test,all}.npz`) | Init | Epochs | peak LR (batch 256) | prototypes | kNN / linear (teacher, style, 10k -> 4k) |
|------|------|-----|------|-----|------|
| `dino_adapt_artbench` | ImageNet-1k DINO (backbone + heads) | 50 | 1e-4 | 65536 | 0.585 / 0.552 (before: 0.542 / 0.556) |
| `dino_adapt_wikiart` | ImageNet-1k DINO | 50 | 1e-4 | 65536 | 0.561 / 0.460 (before: 0.518 / 0.468) |
| `dino_scratch_artbench` | random | 200 | 5e-4 | 16384 | 0.528 / 0.540 |
| `dino_scratch_wikiart` | random | 200 | 5e-4 | 16384 | 0.517 / 0.436 |

The adapted runs start from Meta's full checkpoint (`pretrained/dino_vitb16_full/`, download URL in `src/models/dino.py`). Embedding = CLS
token of the **teacher** backbone (768-D, L2-normalised), Resize 256 + CenterCrop 224, ImageNet mean/std, as for the pretrained `dino_vitb16`.
All four models are embedded on both datasets (cross-dataset). Collapse monitors (teacher entropy, KL of the mean teacher output to
uniform, feature spread) and kNN / linear probes every 5 epochs are logged to wandb / `runs/<run>/metrics.jsonl`.

```bash
python scripts/29_cache_images.py                                   # once, login node
DATASET=artbench INIT=scratch  TRAIN_ARGS="--epochs 200" TIME_LIMIT_MIN=465 CHAIN=1 NEXT_TIME=08:00:00 \
    sbatch --partition=dc-gpu --time=08:00:00 scripts/slurm/train_dino.sbatch      # INIT=imagenet --epochs 50 for the adapted runs
DINO_RUNS="runs/<adapt_ab> runs/<adapt_wa> runs/<scratch_ab> runs/<scratch_wa>" sbatch scripts/slurm/dino_embeddings.sbatch
python scripts/02_extract_features.py --model dino_train --checkpoint runs/<run>/dino_final.pt --dataset artbench --split all
```

---

## Explicit images are hidden in the explorer by default

|-------|---------|------|
| `gan_disc` + `--checkpoint runs/<run>/gan_final.pt` | **discriminator features**: the 512-D output of its last hidden layer (input of the real/fake logit), eval mode, no gradients | `gan_disc` |
| `gan_enc` + `--checkpoint runs/<enc run>/gan_enc_final.pt` | **inversion encoder**: latent w (512-D) of a ResNet-34 trained to invert the frozen generator (`scripts/25_train_gan_encoder.py`; pixel L2 + VGG-16 perceptual loss on real paintings, latent supervision on generated ones) | `gan_enc` |

Both use the same fixed pre-processing for every image (resize the shorter side to 128, centre crop, scale to [-1, 1]); in eval mode the
discriminator's minibatch-stddev channel uses its training EMA, so a vector does not depend on the other images of its batch. Vectors are
not normalised. The explorer shows one **GAN (StyleGAN2)** model type with a Weights menu (discriminator features / inversion encoder).

```bash
# 4 GPUs: smoke test + run in 2 h devel jobs that resubmit themselves (same run dir, same wandb run) until KIMG is reached
KIMG=8000 CHAIN=1 NEXT_PARTITION=dc-gpu-devel NEXT_TIME=02:00:00 NEXT_TIME_LIMIT_MIN=108 TRAIN_ARGS="--amp bf16 --channels-last" \
    sbatch scripts/slurm/train_gan.sbatch
# the inversion encoder (needs the trained GAN)
GAN_RUN=runs/<stamp>_gan_union sbatch scripts/slurm/train_gan_encoder.sbatch
# embeddings (train + test + merged, both datasets) and the explorer pages
GAN_RUN=runs/<stamp>_gan_union ENC_RUN=runs/<stamp>_gan_enc sbatch scripts/slurm/embed_gan.sbatch
```

The FID metric needs the torchvision Inception-v3 weights (the encoder the VGG-16 weights) in the torch hub cache: fetch them once on a
login node (`torchvision.models.inception_v3(weights="IMAGENET1K_V1")`, `vgg16(weights="IMAGENET1K_V1")`); compute nodes have no internet.

---

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
