# StyleGAN2 on ArtBench-10 + WikiArt (train splits jointly)

`scripts/20_train_gan.py` (slurm: `scripts/slurm/train_gan.sbatch`), model `src/models/gan.py`. Pure-PyTorch StyleGAN2 (no custom CUDA ops), unconditional, **128 px**,
123,143 training paintings (ArtBench-10 train + WikiArt train), G 22.9M / D 23.9M parameters, channels min(16384/res, 512), mapping network 2 layers, style mixing 0.9,
path-length regulariser (weight 2, every 4 steps), lazy R1 (gamma 0.5, every 16 steps), Adam lr 0.0025 (betas 0, 0.99), G EMA (half-life 40 kimg), adaptive discriminator
augmentation (target 0.6; in practice it stayed at p ~ 0 after the first 1000 kimg: with 123k diverse images the discriminator did not overfit), batch 4 x 32.
**Why this model:** robust, well understood, trains from scratch without tuning, and the discriminator gives a natural embedding; implemented without the CUDA extensions of the
official code (modulated conv in non-fused form, FIR resampling via avg-pooling / bilinear interpolation) because those may not build on JURECA. 128 px so that ~6M images fit the budget.

| | |
|---|---|
| run | `runs/20261022-131606_gan_union` (wandb [o8dbvvvb](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/o8dbvvvb)) |
| trained | **6,141 kimg** (~6.1M real images shown, ~50 epochs); planned target 8,000 kimg, stopped at 6,141 because the FID-like metric was flattening (22.1 -> 20.9 in the last 1000 kimg) and devel slots were scarce |
| hardware / time | 1 x JURECA node, 4 x A100-40GB, 5 devel jobs (2 h each) chained with `--resume`, ~5.5 h of training. Speed: 237 img/s at first, **~360 img/s** after replacing the depthwise-conv resampling by pooling / interpolation (double back-prop through depthwise convs made R1 / path length cost ~45 % of the time) and bf16 + channels-last |
| FID-like (torchvision Inception-v3 pool features, 5000 generated vs 5000 held-out real; not comparable to published FIDs) | 1000 kimg 88.1, 2000: 39.8, 3000: 28.4, 4000: 24.6, 5000: 22.1, 6000: 20.9 |
| stability | no divergence or NaN. Averages over 1000-kimg windows: loss_D 1.08 / 1.11 / 1.06, D(real) +0.81 -> +0.54 -> +0.59, D(fake) -0.67 -> -0.47 -> -0.52, R1 0.14 -> 0.32, path length 0.21 -> 0.30. Sign of D(real) ~ 0.3-0.5 (< ADA target), i.e. the discriminator never overfit |
| samples | `samples/` of the run: fixed-latent 8x8 grids of G_ema (truncation 1.0 and 0.7) every 100 kimg, also logged to wandb; `docs/gan_samples_trunc0.7.png` |
| caveats | the run was restarted at 1.6M images to switch to pooling-based resampling (identical in the interior of the image, border pixels of the generator differ slightly) and bf16; wandb dropped ~160 re-run iterations |

Samples are recognisably painterly (portraits with plausible faces, landscapes, abstract colour fields), with the usual 128-px-GAN weaknesses (distorted anatomy and buildings, blurred detail).

## Embeddings

* `gan_disc` (**required**): 512-D output of the discriminator's last hidden fc layer, eval mode, no gradients (`representations/<dataset>_gan_disc_{train,test,all}.npz`, G + D final = `gan_final.pt`).
* `gan_enc` (stretch, done): inversion encoder `runs/20261022-195550_gan_enc` (wandb [n3tuv3xl](https://wandb.ai/asker0-university-of-koblenz/art-representation-explorer/runs/n3tuv3xl),
  `scripts/25_train_gan_encoder.py`): ResNet-34 (ImageNet init) -> one latent w (512-D) of the frozen G_ema, 5 epochs (~18 min on 4 GPUs), loss = pixel L2 + VGG-16 perceptual on real
  training paintings + latent regression on generated images. Held-out (2048 test paintings) reconstruction MSE 0.105 (images in [-1, 1]), perceptual 5.09 (6.4 at the start).
  Reconstructions (`docs/gan_encoder_recon.png`, top = input) keep colour and global layout but lose detail: a single w cannot represent real paintings well (no w+, no optimisation).

## Agreement with ImageNet groups / art style (`scripts/18_imagenet_agreement.py`, 10-NN accuracy, z-scored, `imagenet_agreement.md`)

| | ArtBench group | ArtBench style | WikiArt group | WikiArt style |
|---|---|---|---|---|
| GAN discriminator | 0.393 | 0.402 | 0.423 | 0.304 |
| GAN inversion encoder | 0.393 | 0.417 | 0.416 | 0.328 |
| autoencoder | 0.261 | 0.237 | 0.267 | 0.191 |
| ResNet-50 ImageNet | 0.564 | 0.499 | 0.595 | 0.426 |
| MAE ImageNet | 0.496 | 0.528 | 0.525 | 0.454 |
| SimCLR ImageNet | 0.552 | 0.503 | 0.582 | 0.433 |

The GAN features sit between the autoencoder and the pretrained ImageNet / self-supervised models: they carry style and content information (style kNN 0.30-0.42 against 0.1 chance)
but are weaker than models trained for recognition.
