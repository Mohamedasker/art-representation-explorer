# Grayscale input · wikiart (style accuracy, 10000 train -> 4000 test)

| model | lin_colour_colour | lin_gray_gray | lin_colour_to_gray | gray_drop_lin | knn_colour_colour | knn_gray_gray | cos_colour_vs_gray |
|---|---|---|---|---|---|---|---|
| imagenet | 0.422 | 0.412 | 0.342 | 0.010 | 0.460 | 0.443 | 0.943 |
| baseline | 0.473 | 0.467 | 0.461 | 0.006 | 0.520 | 0.508 | 0.984 |
| color_heavy | 0.426 | 0.428 | 0.426 | -0.001 | 0.467 | 0.463 | 0.996 |
| edge_texture | 0.456 | 0.432 | 0.429 | 0.024 | 0.469 | 0.447 | 0.976 |
| rotation | 0.458 | 0.431 | 0.384 | 0.027 | 0.474 | 0.457 | 0.942 |
| geometry_heavy | 0.461 | 0.440 | 0.399 | 0.021 | 0.484 | 0.470 | 0.904 |
| no_color | 0.402 | 0.466 | 0.428 | -0.064 | 0.406 | 0.503 | 0.652 |
| blur_heavy | 0.355 | 0.333 | 0.289 | 0.022 | 0.398 | 0.362 | 0.883 |
| crop_only | 0.387 | 0.335 | 0.149 | 0.052 | 0.419 | 0.271 | 0.072 |
