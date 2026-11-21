"""Self-play on several CPU cores at once.

Most of a CPU self-play game is single-threaded Python (tree search and
python-chess move generation), so one process can't use more than about one core
however many threads PyTorch has. :class:`SelfPlayPool` keeps worker processes
alive for the whole training run; each plays whole games with
:class:`~alphazero.batched_mcts.BatchedMCTS` on a one-thread PyTorch.

Every iteration the current weights are written once to a temporary file, and
each worker reloads them (folded for inference) when the file changes. Each game
gets its own seed, drawn from NumPy's global generator, so with a fixed
``--seed`` the games don't depend on the number of workers or on which worker
plays which game. (``--workers 1`` plays in the training process instead, without
per-game seeds.)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
import torch

from .batched_mcts import play_game
from .config import Config
from .network import AlphaZeroNet, inference_copy
from .selfplay import GameRecord


def resolve_workers(requested: int, device: torch.device, num_games: int) -> int:
    """Worker processes to use: ``requested``, or for 0 (auto) one per CPU core.

    Auto is 1 (play in this process) off the CPU: there the network is the
    bottleneck and several processes would just contend for the one GPU.
    """
    if requested <= 0:
        requested = (os.cpu_count() or 1) if device.type == "cpu" else 1
    return max(1, min(requested, num_games))


def seed_game(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- worker side --------------------------------------------------------------

_worker_model: tuple[str, AlphaZeroNet] | None = None


def _init_worker() -> None:
    torch.set_num_threads(1)


def _play(weights_path: str, cfg: Config, seed: int) -> GameRecord:
    global _worker_model
    if _worker_model is None or _worker_model[0] != weights_path:
        saved = torch.load(weights_path, map_location="cpu", weights_only=True)
        model = AlphaZeroNet(**saved["arch"])
        model.load_state_dict(saved["state_dict"])
        _worker_model = (weights_path, inference_copy(model))
    seed_game(seed)
    return play_game(_worker_model[1], cfg, cfg.mcts_batch_size)


# --- parent side --------------------------------------------------------------

class SelfPlayPool:
    """A pool of self-play worker processes. Use as a context manager."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._executor = ProcessPoolExecutor(
            workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        self._tmp = tempfile.TemporaryDirectory(prefix="alphazero-selfplay-")
        self._version = 0

    def __enter__(self) -> "SelfPlayPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._executor.shutdown(cancel_futures=True)
        self._tmp.cleanup()

    def _publish(self, model: AlphaZeroNet) -> str:
        """Write the current weights for the workers and return the file's path."""
        old = os.path.join(self._tmp.name, f"weights_{self._version}.pt")
        self._version += 1
        path = os.path.join(self._tmp.name, f"weights_{self._version}.pt")
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"arch": model.arch, "state_dict": state}, path)
        if os.path.exists(old):
            os.remove(old)  # every worker has finished the previous iteration's games
        return path

    def generate_games(
        self,
        model: AlphaZeroNet,
        cfg: Config,
        num_games: int | None = None,
        log: Callable[[str], None] = print,
    ) -> list[GameRecord]:
        """Play ``num_games`` self-play games in the workers, returned in seed order."""
        n = num_games or cfg.games_per_iteration
        path = self._publish(model)
        seeds = np.random.randint(0, 2**31 - 1, size=n)
        futures = {self._executor.submit(_play, path, cfg, int(s)): i for i, s in enumerate(seeds)}
        games: list[GameRecord | None] = [None] * n
        for done, fut in enumerate(as_completed(futures), 1):
            rec = games[futures[fut]] = fut.result()
            note = " (move limit, scored as draw)" if rec.adjudicated else ""
            log(f"  self-play game {done}/{n}: {rec.result} in {rec.num_plies} plies{note}")
        return games
