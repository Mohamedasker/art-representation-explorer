# Do the embedding spaces organise paintings by ImageNet class?

Confident paintings only (CLIP puts >= 0.35 of its ImageNet probability on one WordNet group). kNN = leave-one-out 10-NN majority vote (cosine, features z-scored per dimension); NMI = k-means (k = 17 groups) vs the labels. Groups come from CLIP ViT-L/14, so the CLIP row is not an independent judge.

## artbench (10,000 images; chance kNN for ImageNet groups 0.269, for styles 0.156)

| representation | ImageNet group kNN | ImageNet group NMI | art style kNN | art style NMI |
|---|---|---|---|---|
| cnn_supervised | 0.469 | 0.177 | 0.717 | 0.445 |
| resnet50_imagenet | 0.564 | 0.239 | 0.499 | 0.194 |
| autoencoder | 0.261 | 0.045 | 0.237 | 0.083 |
| mae_imagenet | 0.496 | 0.154 | 0.528 | 0.195 |
| mae_adapt_artbench | 0.522 | 0.193 | 0.577 | 0.278 |
| mae_adapt_wikiart | 0.526 | 0.190 | 0.566 | 0.249 |
| mae_scratch_artbench | 0.439 | 0.140 | 0.524 | 0.224 |
| mae_scratch_wikiart | 0.442 | 0.133 | 0.498 | 0.178 |
| simclr_imagenet | 0.552 | 0.222 | 0.503 | 0.198 |
| simclr_artbench_baseline | 0.498 | 0.193 | 0.553 | 0.278 |
| dinov2_vitb14 | 0.668 | 0.293 | 0.558 | 0.268 |
| clip_vitb32 | 0.641 | 0.235 | 0.654 | 0.397 |

## wikiart (10,000 images; chance kNN for ImageNet groups 0.226, for styles 0.172)

| representation | ImageNet group kNN | ImageNet group NMI | art style kNN | art style NMI |
|---|---|---|---|---|
| cnn_supervised | 0.508 | 0.184 | 0.674 | 0.401 |
| resnet50_imagenet | 0.595 | 0.258 | 0.426 | 0.176 |
| autoencoder | 0.267 | 0.056 | 0.191 | 0.074 |
| mae_imagenet | 0.525 | 0.179 | 0.454 | 0.159 |
| mae_adapt_artbench | 0.552 | 0.203 | 0.493 | 0.203 |
| mae_adapt_wikiart | 0.566 | 0.215 | 0.513 | 0.224 |
| mae_scratch_artbench | 0.468 | 0.132 | 0.421 | 0.156 |
| mae_scratch_wikiart | 0.480 | 0.160 | 0.438 | 0.170 |
| simclr_imagenet | 0.582 | 0.238 | 0.433 | 0.165 |
| simclr_wikiart_baseline | 0.537 | 0.209 | 0.511 | 0.268 |
| dinov2_vitb14 | 0.689 | 0.311 | 0.472 | 0.209 |
| clip_vitb32 | 0.666 | 0.252 | 0.562 | 0.325 |
