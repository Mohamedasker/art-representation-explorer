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
| simclr_artbench_blur_heavy | 0.454 | 0.166 | 0.471 | 0.201 |
| simclr_artbench_color_heavy | 0.510 | 0.222 | 0.514 | 0.237 |
| simclr_artbench_crop_only | 0.326 | 0.076 | 0.449 | 0.211 |
| simclr_artbench_edge_texture | 0.466 | 0.175 | 0.519 | 0.255 |
| simclr_artbench_geometry_heavy | 0.427 | 0.140 | 0.537 | 0.274 |
| simclr_artbench_no_color | 0.421 | 0.111 | 0.448 | 0.170 |
| simclr_artbench_rotation | 0.408 | 0.122 | 0.519 | 0.256 |
| vit_imagenet | 0.534 | 0.207 | 0.476 | 0.184 |
| vit_supervised | 0.389 | 0.122 | 0.833 | 0.653 |
| vit_supervised_ft | 0.511 | 0.199 | 0.786 | 0.569 |
| dinov2_vitb14 | 0.668 | 0.293 | 0.558 | 0.268 |
| dino_vitb16 | 0.585 | 0.237 | 0.561 | 0.242 |
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
| simclr_wikiart_blur_heavy | 0.497 | 0.196 | 0.406 | 0.181 |
| simclr_wikiart_color_heavy | 0.556 | 0.257 | 0.458 | 0.208 |
| simclr_wikiart_crop_only | 0.335 | 0.064 | 0.419 | 0.171 |
| simclr_wikiart_edge_texture | 0.515 | 0.189 | 0.470 | 0.232 |
| simclr_wikiart_geometry_heavy | 0.485 | 0.164 | 0.486 | 0.258 |
| simclr_wikiart_no_color | 0.477 | 0.131 | 0.415 | 0.162 |
| simclr_wikiart_rotation | 0.461 | 0.151 | 0.485 | 0.258 |
| vit_imagenet | 0.566 | 0.234 | 0.393 | 0.161 |
| vit_supervised | 0.443 | 0.116 | 0.820 | 0.668 |
| vit_supervised_ft | 0.547 | 0.191 | 0.759 | 0.528 |
| dinov2_vitb14 | 0.689 | 0.311 | 0.472 | 0.209 |
| dino_vitb16 | 0.610 | 0.254 | 0.511 | 0.210 |
| clip_vitb32 | 0.666 | 0.253 | 0.562 | 0.325 |
