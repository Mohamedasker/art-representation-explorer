# Autoregressive models on ArtBench-10 + WikiArt (iGPT and VQ tokens + GPT)

Both models are trained from scratch on the train splits of both datasets jointly (123,143 paintings; with the horizontally flipped copies 246,286 sequences per
epoch), validated on both test splits (18,301 paintings), 4 x A100-40GB (one JURECA dc-gpu-devel node), bf16, global batch 256, AdamW (wd 0.1, cosine), wandb,
resumable run directories. Images: shorter side resized to 128 px, centre crop (`src/models/ar_data.py`; pixel cache in `data/ar_tokens/`, gitignored).

## iGPT (primary, slug `ar_igpt`): next-pixel prediction at 32 x 32 px

Palette of 512 colours (k-means on RGB pixels of 6,000 train images; mean quantisation error 0.031), 1024 tokens per painting, GPT with 24 layers / width 512 /
8 heads / 76.7M parameters (iGPT-S size; `scripts/26_prepare_igpt.py`, `scripts/27_train_igpt.py`, `scripts/slurm/train_igpt.sbatch`). Unconditional (no labels).

* **Final run** `runs/20261022-181653_ar_igpt_artbench_wikiart`: 14 epochs (13,468 steps), peak LR 6e-4, ~660 sequences/s (~95 min), wandb
  [5v1hczp1](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/5v1hczp1). Held-out loss **1.1147 bits/dim** (3.344 bits/pixel-token, perplexity 10.2 of 512;
  ArtBench 1.1233, WikiArt 1.1044 bits/dim; bits of the colour-quantised 32 px image, not of 8-bit pixels). The checkpoint used for the embeddings is `checkpoints/best.pt`
  (step 10,500, lowest validation loss).
* A first 40-epoch schedule (`runs/20261022-144757_ar_igpt_artbench_wikiart`, wandb x0fl99tl) reached its best validation loss at step 9,500 (1.1165 bits/dim) and
  then overfitted (train 1.04 vs val 1.14 bits/dim at step 28k), so it was stopped and replaced by the 14-epoch run with a complete cosine schedule.
* Samples (`samples/`, also in wandb): unconditional (full softmax, top-k 50) and completions of the lower half of real held-out paintings from the upper half.

## VQ tokens + GPT (optional variant, slugs `ar_gpt` and `ar_vq`): 128 px, 256 tokens

* **Tokenizer** (`scripts/21_train_vq.py`, `scripts/slurm/train_vq.sbatch`): VQ-VAE, 56.1M parameters, 3 stride-2 stages -> 16 x 16 = 256 tokens, codebook 4096 x 32 (l2-normalised
  codes, dead codes re-initialised), loss L1 + VGG16 perceptual + codebook; no adversarial loss. `runs/20261022-134439_vq_artbench_wikiart`, 40 epochs (19,240 steps, ~790 img/s, ~2 h),
  wandb [upxt79y2](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/upxt79y2). Test reconstruction: **PSNR 23.8 dB** (ArtBench 23.6, WikiArt 23.9), L1 0.047,
  perceptual 0.252, 100 % of the codes used, code perplexity 4,048. Embedding `ar_vq` = quantised latents average-pooled to 4 x 4 (512-D).
* **Transformer** (`scripts/22_tokenize_ar.py`, `scripts/23_train_ar.py`, `scripts/slurm/train_ar.sbatch`): 12 layers / width 1024 / 16 heads / 159.8M parameters, prefix tokens (dataset, style)
  with 20 % label dropout so it also works unlabeled / unconditionally. `runs/20261022-153559_ar_gpt_artbench_wikiart` (~1,790 sequences/s), wandb
  [p8oezq3z](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/p8oezq3z). Validation loss is lowest at step 6,500 (**9.46 bits/token**, perplexity 705 of 4096; ~9.48
  unlabeled); afterwards the model memorises the training tokens (train 8.76 vs val 9.58 bits/token at step 12k), so the run was stopped at step ~13k (of 57,720) and
  `checkpoints/best.pt` is used. Samples: unconditional, and per ArtBench / WikiArt style (CFG 1.5) in `samples/`.

## Layer probe (`scripts/24_probe_ar_layers.py`, `runs/<run>/layer_probe.{json,png}`)

Hidden states of every block, non-affine LayerNorm, mean over the token positions; kNN (k=20, cosine) and logistic regression on art style, trained on a 10k subset of the train split,
evaluated on the whole test split; chosen layer = highest mean of the four accuracies (ArtBench / WikiArt x kNN / LR). Mean accuracy per layer (ArtBench and WikiArt style, chance 0.105 / 0.159):

| model | layer 1 | 4 | 8 | 10 | **best** | 14 | 18 | last |
|---|---|---|---|---|---|---|---|---|
| iGPT (24 layers) | 0.275 | 0.329 | 0.367 | 0.385 | **0.388 (layer 11)** | 0.386 | 0.380 | 0.369 |
| VQ tokens + GPT (12 layers) | 0.379 | 0.421 | 0.437 | 0.437 | **0.438 (layer 9)** | - | - | 0.430 |

As in the iGPT paper the best features are in the middle of the network (a plateau from layer 11 to ~16 for iGPT), not at the output. Per-layer numbers: the json files.

## Agreement with ImageNet groups / art style (`scripts/18_imagenet_agreement.py`, `imagenet_agreement.md`; kNN, z-scored features)

| | ArtBench ImageNet kNN | ArtBench style kNN | WikiArt ImageNet kNN | WikiArt style kNN |
|---|---|---|---|---|
| `ar_igpt` | 0.344 | 0.408 | 0.363 | 0.324 |
| `ar_gpt` | 0.407 | 0.474 | 0.437 | 0.406 |
| `ar_vq` | 0.333 | 0.352 | 0.347 | 0.257 |
