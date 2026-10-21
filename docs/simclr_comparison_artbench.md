# SimCLR augmentation comparison · artbench

## Static metrics (held-out test split; R2 of ridge regression embedding -> statistic)

| model | colour_R2 | colour_R2_sat | colour_R2_bright | colour_R2_hue | texture_R2 | top_pc_colour | style_acc | cka_baseline |
|---|---|---|---|---|---|---|---|---|
| imagenet | 0.436 | 0.781 | 0.917 | 0.345 | 0.823 | 0.127 | 0.589 | 0.729 |
| baseline | 0.359 | 0.629 | 0.889 | 0.270 | 0.795 | 0.132 | 0.592 | 1.000 |
| color_heavy | 0.167 | 0.289 | 0.720 | 0.092 | 0.663 | 0.078 | 0.553 | 0.808 |
| edge_texture | 0.429 | 0.733 | 0.923 | 0.341 | 0.849 | 0.154 | 0.558 | 0.851 |
| rotation | 0.447 | 0.838 | 0.916 | 0.351 | 0.845 | 0.172 | 0.550 | 0.821 |
| geometry_heavy | 0.497 | 0.858 | 0.934 | 0.409 | 0.871 | 0.191 | 0.572 | 0.845 |
| no_color | 0.617 | 0.830 | 0.890 | 0.563 | 0.791 | 0.262 | 0.515 | 0.517 |
| blur_heavy | 0.466 | 0.849 | 0.932 | 0.372 | 0.825 | 0.183 | 0.500 | 0.740 |
| crop_only | 0.826 | 0.951 | 0.947 | 0.799 | 0.813 | 0.224 | 0.472 | 0.357 |
