import torch
import torch.nn as nn


class ClassicalMLPBaseline(nn.Module):
    """Standard classical multi-layer perceptron baseline."""

    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
