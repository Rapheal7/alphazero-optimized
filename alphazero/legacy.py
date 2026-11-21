"""Load checkpoints saved by the archived AlphaZeroChess2 notebook.

That notebook (``AlphaZeroAgentParallel``, files like
``model_5_20_256_parallel_chess.pt``) saves ``{'episode', 'model_state',
'optim_state'}`` for its ``ResNetAlphaZero``. The network differs from
:class:`alphazero.network.AlphaZeroNet`:

- 14 input planes: 12 piece planes, side to move, and one plane that is 1 when
  *any* castling right remains. The piece and side-to-move planes match
  :func:`alphazero.game.encode_board`, so :class:`NotebookNet` takes the usual
  input (:data:`~alphazero.game.NUM_PLANES` planes) and converts it, and works
  with ``MCTS`` and ``BatchedMCTS`` unchanged.
- Biased convolutions, no ReLU in the policy head, a 1-channel value head.

The move encoding is the same 4544-action mapping as :data:`alphazero.game.ENCODER`.

Two training quirks of the notebook are undone here:

- Its ``forward`` already returns a softmax, and the training loop applied
  another softmax on top. ``NotebookNet`` returns the raw policy logits, which
  rank moves the same way, but as a sharper distribution.
- Its value targets were assigned with the wrong sign (each position got the
  result for the player who had just moved), so the value is negated to give the
  value for the side to move, as this package expects.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ACTION_SIZE

NOTEBOOK_PLANES = 14


class _NotebookResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class NotebookNet(nn.Module):
    """The notebook's ``ResNetAlphaZero`` (same parameter names), taking the package's input planes."""

    def __init__(self, num_res_blocks: int, num_channels: int, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.arch = {"num_res_blocks": num_res_blocks, "num_channels": num_channels,
                     "action_size": action_size, "format": "notebook"}
        self.conv1 = nn.Conv2d(NOTEBOOK_PLANES, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [_NotebookResidualBlock(num_channels) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    @staticmethod
    def to_notebook_planes(x: torch.Tensor) -> torch.Tensor:
        """Package encoding (planes 0-16 are used) -> the notebook's 14 planes."""
        any_castling = x[:, 13:17].amax(dim=1, keepdim=True)
        return torch.cat([x[:, :13], any_castling], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(self.to_notebook_planes(x))))
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), -self.value_head(x)

    @torch.no_grad()
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Evaluate one encoded position. Returns (policy logits, value)."""
        device = next(self.parameters()).device
        logits, value = self(torch.from_numpy(planes).unsqueeze(0).to(device))
        return logits[0].float().cpu().numpy(), float(value.item())


def is_notebook_checkpoint(ckpt: Any) -> bool:
    return isinstance(ckpt, dict) and "model_state" in ckpt and "arch" not in ckpt


def notebook_model_from_checkpoint(ckpt: dict, path: Any) -> NotebookNet:
    state = ckpt["model_state"]
    conv1 = state.get("conv1.weight")
    policy = state.get("policy_head.3.weight")
    if conv1 is None or policy is None:
        raise ValueError(f"{path} has a 'model_state' but not the AlphaZeroChess2 ResNetAlphaZero layers")
    if conv1.shape[1] != NOTEBOOK_PLANES or policy.shape[0] != ACTION_SIZE:
        raise ValueError(
            f"{path} is from a notebook network with {conv1.shape[1]} input planes and "
            f"{policy.shape[0]} actions; only the 14-plane, {ACTION_SIZE}-action one can be loaded")
    blocks = {int(m.group(1)) for k in state if (m := re.match(r"res_blocks\.(\d+)\.", k))}
    model = NotebookNet(num_res_blocks=len(blocks), num_channels=conv1.shape[0])
    model.load_state_dict(state)
    return model
