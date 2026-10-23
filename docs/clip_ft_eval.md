# CLIP fine-tuning: fit vs. forgetting

ViT-B-32-quickgelu / openai; top-1 accuracy in %. Near-duplicate threshold (CLIP cosine): 0.95.

| model | style zero-shot, ArtBench test | style zero-shot, WikiArt test | ArtBench test, no near-dups of WikiArt train | WikiArt test, no near-dups of ArtBench train | kNN style, ArtBench | kNN style, WikiArt | CIFAR-100 zero-shot | CIFAR-10 zero-shot |
|---|---|---|---|---|---|---|---|---|
| original CLIP | 48.7 | 23.8 | – | – | 69.3 | 63.9 | 64.5 | 88.3 |
| fine-tuned on artbench | 66.3 | 38.0 | – | 37.4 | 73.4 | 67.0 | 59.8 | 81.0 |
| fine-tuned on artbench (alpha 0.5) | 59.6 | 32.8 | – | 33.7 | 72.7 | 66.4 | 64.0 | 86.5 |
| fine-tuned on wikiart | 65.6 | 50.9 | 66.0 | – | 73.3 | 69.1 | 56.2 | 82.7 |
| fine-tuned on wikiart (alpha 0.5) | 60.1 | 39.1 | 62.9 | – | 72.6 | 67.4 | 62.6 | 86.8 |

Excluded near-duplicates: artbench test: 2689, wikiart test: 1565
