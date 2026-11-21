"""Policy/value residual network."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .game import ACTION_SIZE, NUM_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """ResNet trunk with a policy head (logits over every action) and a value head.

    The value is in [-1, 1] from the point of view of the side to move.
    """

    def __init__(
        self,
        num_res_blocks: int = 2,
        num_channels: int = 32,
        in_planes: int = NUM_PLANES,
        action_size: int = ACTION_SIZE,
    ) -> None:
        super().__init__()
        self.arch = {
            "num_res_blocks": num_res_blocks,
            "num_channels": num_channels,
            "in_planes": in_planes,
            "action_size": action_size,
        }
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "AlphaZeroNet":
        return cls(num_res_blocks=cfg.num_res_blocks, num_channels=cfg.num_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # A network saved before planes were appended to the encoding (in_planes
        # 18, without the repetition planes) reads only the planes it knows.
        x = self.trunk(self.stem(x[:, :self.arch["in_planes"]]))
        return self.policy_head(x), self.value_head(x)

    @torch.no_grad()
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one encoded position. Returns (policy logits, value)."""
        device = next(self.parameters()).device
        x = torch.from_numpy(planes).unsqueeze(0).to(device)
        logits, value = self(x)
        return logits[0].float().cpu().numpy(), float(value.item())
