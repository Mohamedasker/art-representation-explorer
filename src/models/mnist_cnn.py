"""
The small CNN of exercise sheet 1 (``SmallCNN``), for 1 x 28 x 28 MNIST images, and a helper that returns the
intermediate representations of a batch (the activations after every conv block, the hidden fully-connected
layer and the logits).

The same class is written out in exercise_sheets/mnist_pca_cnn_tsne_umap.ipynb (students do not have this repository).
"""

from __future__ import annotations

import torch
import torch.nn as nn

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# name -> (what it is, how many numbers per image)
LAYERS = {
    "conv1": ("after conv block 1: 32 maps of 14 x 14", 32 * 14 * 14),
    "conv2": ("after conv block 2: 64 maps of 7 x 7", 64 * 7 * 7),
    "conv3": ("after conv block 3: 128 maps of 3 x 3", 128 * 3 * 3),
    "fc": ("hidden fully-connected layer (256 units, after ReLU)", 256),
    "logits": ("the 10 class scores (logits)", 10),
}


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))


class MnistCNN(nn.Module):
    """3 conv blocks (28 -> 14 -> 7 -> 3 pixels) and a two-layer classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1, self.block2, self.block3 = _block(1, 32), _block(32, 64), _block(64, 128)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.drop(self.flatten(self.block3(self.block2(self.block1(x))))))))

    @torch.no_grad()
    def intermediates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Activations of one batch (eval mode), flattened to (B, D), keyed like :data:`LAYERS`."""
        self.eval()
        a1 = self.block1(x)
        a2 = self.block2(a1)
        a3 = self.block3(a2)
        h = torch.relu(self.fc1(self.flatten(a3)))
        out = self.fc2(h)
        return {"conv1": a1.flatten(1), "conv2": a2.flatten(1), "conv3": a3.flatten(1), "fc": h, "logits": out}
