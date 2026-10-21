# SimCLR augmentation comparison · wikiart

## Static metrics (held-out test split; R2 of ridge regression embedding -> statistic)

| model | colour_R2 | colour_R2_sat | colour_R2_bright | colour_R2_hue | texture_R2 | top_pc_colour | style_acc | cka_baseline |
|---|---|---|---|---|---|---|---|---|
| imagenet | 0.424 | 0.770 | 0.874 | 0.336 | 0.785 | 0.110 | 0.529 | 0.723 |
| baseline | 0.389 | 0.707 | 0.865 | 0.301 | 0.777 | 0.131 | 0.562 | 1.000 |
| color_heavy | 0.185 | 0.309 | 0.670 | 0.117 | 0.673 | 0.086 | 0.517 | 0.824 |
| edge_texture | 0.434 | 0.784 | 0.897 | 0.344 | 0.823 | 0.139 | 0.531 | 0.849 |
| rotation | 0.450 | 0.837 | 0.896 | 0.357 | 0.822 | 0.162 | 0.530 | 0.826 |
| geometry_heavy | 0.493 | 0.878 | 0.914 | 0.403 | 0.839 | 0.182 | 0.538 | 0.851 |
| no_color | 0.570 | 0.817 | 0.856 | 0.511 | 0.752 | 0.223 | 0.485 | 0.616 |
| blur_heavy | 0.467 | 0.844 | 0.906 | 0.377 | 0.801 | 0.179 | 0.460 | 0.735 |
| crop_only | 0.766 | 0.920 | 0.905 | 0.733 | 0.767 | 0.159 | 0.447 | 0.320 |
