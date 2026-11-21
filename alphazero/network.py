"""Policy/value residual network."""

from __future__ import annotations

import copy

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
        x = self.trunk(self.stem(x))
        return self.policy_head(x), self.value_head(x)

    @torch.no_grad()
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one encoded position. Returns (policy logits, value)."""
        device = next(self.parameters()).device
        x = torch.from_numpy(planes).unsqueeze(0).to(device)
        logits, value = self(x)
        return logits[0].float().cpu().numpy(), float(value.item())


def _fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """A conv computing ``bn(conv(x))`` in eval mode, with the batch norm folded in."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      padding=conv.padding, bias=True).to(conv.weight.device)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale[:, None, None, None])
        fused.bias.copy_((bias - bn.running_mean) * scale + bn.bias)
    return fused


def inference_copy(model: AlphaZeroNet) -> AlphaZeroNet:
    """An eval-mode copy of ``model`` with every batch norm folded into its conv.

    Same outputs up to float rounding, with half the layers per forward pass:
    on CPU the batch norms took about as long as the convolutions. Use it for
    self-play and play, never for training (it has no batch norms to train).
    """
    fused = copy.deepcopy(model).eval()
    stem_conv, stem_bn = fused.stem[0], fused.stem[1]
    fused.stem[0], fused.stem[1] = _fold_bn(stem_conv, stem_bn), nn.Identity()
    for block in fused.trunk:
        block.conv1, block.bn1 = _fold_bn(block.conv1, block.bn1), nn.Identity()
        block.conv2, block.bn2 = _fold_bn(block.conv2, block.bn2), nn.Identity()
    for head in (fused.policy_head, fused.value_head):
        head[0], head[1] = _fold_bn(head[0], head[1]), nn.Identity()
    for p in fused.parameters():
        p.requires_grad_(False)
    return fused
