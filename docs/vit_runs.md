# Supervised ViT-B/16 on the art labels (ArtBench-10 + WikiArt)

`scripts/19_train_vit.py` (slurm: `scripts/slurm/train_vit.sbatch`), exactly parallel to the supervised ResNet-50 (`scripts/10_train_cnn.py`): train splits of both
datasets jointly (123,143 images), three linear heads on the encoder output (style 29 classes, genre 10, artist 128), loss weights 1 / 0.5 / 0.5, label smoothing 0.1,
RandomResizedCrop(0.35-1) + flip, AdamW (weight decay 0.05, cosine schedule, 1000 warm-up steps), bf16, global batch 256 (4 x 64), **30 epochs (14,430 steps)**,
4 x A100-40GB (one JURECA dc-gpu-devel node), ~3,350 img/s, ~21 min per run. ViT-B/16 224 px (timm `vit_base_patch16_224`, 85.9M parameters, drop-path 0.1,
mean = std = 0.5 input normalisation).

| run | init | peak LR (batch 256) | layer-wise LR decay | wandb |
|---|---|---|---|---|
| `runs/20261022-111343_vit_scratch` | **random** (parallel to the CNN) | 2.5e-4 | none | [d0evropf](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/d0evropf) |
| `runs/20261022-111344_vit_imagenet_ft` | **ImageNet-1k** (`vit_base_patch16_224.augreg_in1k`), everything fine-tuned | 5e-5 | 0.75 | [z1390q0b](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/z1390q0b) |

## Top-1 accuracy on the complete held-out (test) splits of both datasets (18,301 images)

| weights | style (all) | style ArtBench | style WikiArt | genre (WikiArt) | artist |
|---|---|---|---|---|---|
| ResNet-50 trained here from scratch (`20261015-175907_cnn_resnet50`, 256 px, 30 ep.) | 0.655 | 0.653 | 0.657 | 0.774 | 0.771 |
| **ViT-B/16 from scratch** (`vit_supervised`) | 0.572 | 0.599 | 0.539 | 0.620 | 0.693 |
| **ViT-B/16 ImageNet init, fine-tuned** (`vit_supervised_ft`) | **0.740** | 0.749 | 0.730 | 0.820 | 0.850 |

A ViT from scratch on ~123k images is weaker than the CNN (no convolutional inductive bias; still improving when the cosine schedule ended, so more epochs would
help) while ImageNet initialisation is clearly better than both. Embeddings (final-layer CLS token, 768-D) of the three ViTs (frozen ImageNet weights, from scratch,
fine-tuned) are in `representations/<dataset>_{vit_imagenet,vit_supervised,vit_supervised_ft}_{train,test,all}.npz`.

## Agreement with ImageNet groups vs. art style (`scripts/18_imagenet_agreement.py`, full table: `imagenet_agreement.md`)

kNN accuracy (10-NN majority vote, cosine, z-scored features; 10,000 images per dataset) predicting the CLIP-assigned ImageNet group / the art style:

| | ArtBench ImageNet group | ArtBench style | WikiArt ImageNet group | WikiArt style |
|---|---|---|---|---|
| ResNet-50 ImageNet | 0.564 | 0.499 | 0.595 | 0.426 |
| ResNet-50 trained here | 0.469 | 0.717 | 0.508 | 0.674 |
| ViT-B/16 ImageNet (frozen) | 0.534 | 0.476 | 0.566 | 0.393 |
| ViT-B/16 from scratch | 0.389 | **0.833** | 0.443 | **0.820** |
| ViT-B/16 ImageNet, fine-tuned | 0.511 | 0.786 | 0.547 | 0.759 |

Supervised training on the art labels moves the transformer's embedding from object content (ImageNet groups) to style, as for the CNN; the from-scratch ViT
ends up the most style-organised (and the least object-organised) space, the fine-tuned one keeps more object information from its ImageNet start.
(The style kNN of the trained models is measured on all of the 10,000 images including their training images, so it is optimistic compared with the held-out
accuracies above.)
