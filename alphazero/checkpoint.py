"""Saving and loading training checkpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .network import AlphaZeroNet

_PATTERN = re.compile(r"checkpoint_iter_(\d+)\.pt$")


def checkpoint_path(directory: str | Path, iteration: int) -> Path:
    return Path(directory) / f"checkpoint_iter_{iteration:04d}.pt"


def save_checkpoint(
    path: str | Path,
    model: AlphaZeroNet,
    cfg: Config,
    iteration: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "arch": model.arch,
            "config": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics or {},
        },
        path,
    )
    return path


def latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := _PATTERN.search(p.name))]
    return max(found)[1] if found else None


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    # weights_only=True: checkpoints only hold tensors and plain Python values,
    # so refuse anything that would need arbitrary unpickling.
    return torch.load(Path(path), map_location=device, weights_only=True)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[AlphaZeroNet, Config, dict]:
    """Load a model for play. Also accepts checkpoints from the archived
    AlphaZeroChess2 notebook (see :mod:`alphazero.legacy`)."""
    from .legacy import is_notebook_checkpoint, notebook_model_from_checkpoint

    ckpt = load_checkpoint(path, device)
    if is_notebook_checkpoint(ckpt):
        model = notebook_model_from_checkpoint(ckpt, path).to(device).eval()
        ckpt.setdefault("iteration", ckpt.get("episode", 0))
        cfg = Config(num_res_blocks=model.arch["num_res_blocks"], num_channels=model.arch["num_channels"])
        return model, cfg, ckpt
    if not isinstance(ckpt, dict) or "arch" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} isn't a checkpoint saved by `alphazero train` or the AlphaZeroChess2 notebook")
    model = AlphaZeroNet(**ckpt["arch"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, Config.from_dict(ckpt["config"]), ckpt
