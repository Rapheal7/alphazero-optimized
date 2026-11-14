"""Hyperparameters and presets.

Defaults are deliberately small: ``python scripts/train.py`` finishes in
a minute or two on a laptop CPU. Use ``--preset colab`` (or override individual
fields) for a real training run on a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import torch


@dataclass
class Config:
    # --- Neural network ---
    num_res_blocks: int = 2
    num_channels: int = 32

    # --- MCTS ---
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    games_per_iteration: int = 4
    # Sample moves in proportion to visit counts for this many plies, then play greedily.
    temperature_moves: int = 30
    # Games longer than this many plies are stopped and scored as a draw.
    max_moves: int = 120

    # --- Training ---
    iterations: int = 5
    train_steps_per_iteration: int = 50
    batch_size: int = 64
    buffer_size: int = 20_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay_iterations: int = 50
    lr_decay_rate: float = 0.1

    # --- Runtime ---
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def updated(self, **overrides: Any) -> "Config":
        """Return a copy with every non-None override applied."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


PRESETS: dict[str, Config] = {
    # Seconds on CPU: used by the tests and the smoke run.
    "tiny": Config(
        num_res_blocks=1,
        num_channels=16,
        num_simulations=8,
        games_per_iteration=1,
        max_moves=20,
        iterations=1,
        train_steps_per_iteration=2,
        batch_size=16,
    ),
    # Default: minutes on CPU.
    "small": Config(),
    # Hours on a Colab GPU. Close to the X1 notebook's final settings.
    "colab": Config(
        num_res_blocks=10,
        num_channels=128,
        num_simulations=200,
        games_per_iteration=25,
        max_moves=300,
        iterations=100,
        train_steps_per_iteration=200,
        batch_size=256,
        buffer_size=200_000,
        lr_decay_iterations=40,
    ),
}


def get_preset(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name])


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA when available, otherwise CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
