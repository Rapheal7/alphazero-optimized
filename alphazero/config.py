"""Hyperparameters and presets.

Defaults are deliberately small: ``python scripts/train.py`` finishes in
a minute or two on a laptop CPU. Use ``--preset colab`` (or override individual
fields) for a real training run on a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    # Games longer than this many plies are stopped and scored by material:
    # tanh(material balance in pawns / material_scale) for white, the negative for
    # black. 0 scores them as draws (which taught the old value head that long
    # games are always drawn).
    max_moves: int = 250
    material_scale: float = 5.0
    # Resign when the root search value stays below resign_threshold for
    # resign_moves of the same side's moves in a row (-1 or lower: never resign).
    # resign_check_fraction of games are played out anyway, to count how often
    # resigning would have thrown away a game that wasn't lost.
    resign_threshold: float = -0.9
    resign_moves: int = 3
    resign_check_fraction: float = 0.1

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


class DeviceUnavailableError(ValueError):
    """The requested device can't be used with this machine or PyTorch build."""


def device_problem(device: torch.device) -> str | None:
    """Why ``device`` can't be used here, or ``None`` if it can."""
    import torch

    if device.type == "cuda":
        if torch.version.cuda is None:
            return (f"this PyTorch build ({torch.__version__}) has no CUDA support; install a "
                    "CUDA build (see https://pytorch.org/get-started/locally/)")
        if not torch.cuda.is_available():
            return "PyTorch has CUDA support, but no CUDA GPU or driver was found"
    elif device.type == "mps" and not torch.backends.mps.is_available():
        return "MPS (Apple GPU) isn't available on this machine"
    return None


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA when available, otherwise CPU.

    Raises :class:`DeviceUnavailableError` (with the reason) for an explicit
    device that can't be used, instead of failing later inside PyTorch.
    """
    import torch  # deferred so importing the config does not load torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    problem = device_problem(device)
    if problem:
        raise DeviceUnavailableError(f"can't use device {name!r}: {problem}")
    return device
