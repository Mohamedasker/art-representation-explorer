# Grayscale input · artbench (style accuracy, 10000 train -> 4000 test)

| model | lin_colour_colour | lin_gray_gray | lin_colour_to_gray | gray_drop_lin | knn_colour_colour | knn_gray_gray | cos_colour_vs_gray |
|---|---|---|---|---|---|---|---|
| imagenet | 0.517 | 0.498 | 0.462 | 0.019 | 0.488 | 0.482 | 0.950 |
| baseline | 0.522 | 0.532 | 0.516 | -0.010 | 0.539 | 0.534 | 0.992 |
| color_heavy | 0.484 | 0.488 | 0.489 | -0.004 | 0.510 | 0.507 | 0.998 |
| edge_texture | 0.517 | 0.502 | 0.498 | 0.015 | 0.499 | 0.486 | 0.984 |
| rotation | 0.503 | 0.475 | 0.465 | 0.028 | 0.491 | 0.475 | 0.963 |
| geometry_heavy | 0.515 | 0.496 | 0.482 | 0.019 | 0.501 | 0.491 | 0.924 |
| no_color | 0.462 | 0.527 | 0.459 | -0.065 | 0.427 | 0.527 | 0.580 |
| blur_heavy | 0.422 | 0.405 | 0.361 | 0.017 | 0.443 | 0.417 | 0.922 |
| crop_only | 0.417 | 0.398 | 0.140 | 0.019 | 0.422 | 0.296 | 0.086 |
