"""Training loop: self-play -> replay buffer -> gradient steps -> checkpoint."""

from __future__ import annotations

import contextlib
import random
import time
from collections import deque
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import checkpoint_path, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import Config, resolve_device
from .batched_mcts import generate_games
from .network import AlphaZeroNet
from .parallel import SelfPlayPool, resolve_workers
from .selfplay import Example


class ReplayBuffer:
    """Fixed-size FIFO of training examples (oldest are dropped first)."""

    def __init__(self, capacity: int) -> None:
        self._data: deque[Example] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def extend(self, examples: Iterable[Example]) -> None:
        self._data.extend(examples)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self._data, min(batch_size, len(self._data)))
        states, pis, zs = zip(*batch)
        return (
            torch.from_numpy(np.stack(states)).to(device),
            torch.from_numpy(np.stack(pis).astype(np.float32)).to(device),
            torch.tensor(zs, dtype=torch.float32, device=device).unsqueeze(1),
        )


def compute_loss(model: AlphaZeroNet, states, target_pis, target_zs) -> tuple[torch.Tensor, dict]:
    """AlphaZero loss: (z - v)^2 - pi . log p. Weight decay is handled by the optimizer."""
    logits, values = model(states)
    policy_loss = -(target_pis * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_zs)
    loss = policy_loss + value_loss
    return loss, {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


def train_step(model, optimizer, buffer: ReplayBuffer, batch_size: int, device) -> dict[str, float]:
    model.train()
    loss, metrics = compute_loss(model, *buffer.sample(batch_size, device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return metrics


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: Config, resume: bool = False, log: Callable[[str], None] = print) -> AlphaZeroNet:
    """Run ``cfg.iterations`` iterations of self-play + training and return the model.

    With ``resume=True`` the latest checkpoint in ``cfg.checkpoint_dir`` is loaded
    (weights, optimizer and LR schedule) and training continues from the next
    iteration. The network shape comes from the checkpoint. The replay buffer
    isn't saved, so it starts empty again.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    ckpt = None
    if resume:
        path = latest_checkpoint(cfg.checkpoint_dir)
        if path is None:
            log(f"No checkpoint in {cfg.checkpoint_dir}/, starting from scratch.")
        else:
            ckpt = load_checkpoint(path, device)
            log(f"Resuming from {path} (iteration {ckpt['iteration']}).")

    model = AlphaZeroNet(**ckpt["arch"]) if ckpt else AlphaZeroNet.from_config(cfg)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_iterations, gamma=cfg.lr_decay_rate
    )
    start = 0
    if ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start = ckpt["iteration"]

    params = sum(p.numel() for p in model.parameters())
    log(f"Device: {device} | network: {model.arch['num_res_blocks']} blocks x "
        f"{model.arch['num_channels']} channels ({params:,} params)")

    workers = resolve_workers(cfg.selfplay_workers, device, cfg.games_per_iteration)
    log(f"Self-play: batched MCTS ({cfg.mcts_batch_size} leaves per network call), "
        f"{workers} worker process{'es' if workers > 1 else ''}")

    buffer = ReplayBuffer(cfg.buffer_size)
    with SelfPlayPool(workers) if workers > 1 else contextlib.nullcontext() as pool:
        for it in range(start, cfg.iterations):
            t0 = time.time()
            log(f"Iteration {it + 1}/{cfg.iterations}")
            if pool is not None:
                games = pool.generate_games(model, cfg, log=log)
            else:
                games = generate_games(model, cfg, log=log)
            t_play = time.time() - t0
            buffer.extend(ex for g in games for ex in g.examples)

            metrics: dict[str, float] = {}
            for _ in range(cfg.train_steps_per_iteration):
                step = train_step(model, optimizer, buffer, cfg.batch_size, device)
                for k, v in step.items():
                    metrics[k] = metrics.get(k, 0.0) + v / cfg.train_steps_per_iteration
            scheduler.step()

            path = save_checkpoint(checkpoint_path(cfg.checkpoint_dir, it + 1), model, cfg, it + 1,
                                   optimizer, scheduler, metrics)
            log(f"  loss {metrics.get('loss', float('nan')):.4f} "
                f"(policy {metrics.get('policy_loss', float('nan')):.4f}, "
                f"value {metrics.get('value_loss', float('nan')):.4f}) | buffer {len(buffer)} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s "
                f"(self-play {t_play:.1f}s) | saved {path}")
    return model
