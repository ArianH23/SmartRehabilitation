"""Grasp classifier: a small MLP over the 74 hand/object features."""
from __future__ import annotations

import torch
import torch.nn as nn


class Net(nn.Module):
    """74 -> 64 -> 32 -> 3 MLP (ReLU, dropout after the first hidden layer).

    The layer names are kept as in the thesis code so that checkpoints trained back then still load.
    """

    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer_3 = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer_1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer_2(x))
        return self.layer_3(x)
