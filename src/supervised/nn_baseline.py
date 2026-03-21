"""
SL Report carry-overs: backbone architecture for Wine NN baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WineNN(nn.Module):
    """
    Multi-Layer Perceptron from SL/OL Report baseline for Wine dataset.

    Architecture: Input -> Hidden(100) -> ReLU -> Output(8)

    The OL baseline used input_dim=12, but for the UL Phase 0 audit,
    we are enforcing the canonical 11-feature contract for the raw data.
    The input_dim changes depending on the DR method used (e.g., PCA=8).
    """

    def __init__(self, input_dim: int, num_classes: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
