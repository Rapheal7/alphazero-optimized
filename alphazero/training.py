"""Training loop: self-play -> replay buffer -> gradient steps -> checkpoint."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import checkpoint_path, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import Config, resolve_device
from .network import AlphaZeroNet
from .selfplay import Example, generate_games, summarize_games


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
    """AlphaZero loss: (z - v)^2 - pi . log p. Weight decay is handled by the optimizer.

    The metrics also split the policy loss (a cross-entropy) into the entropy of
    the search targets, which no network can go below, and the KL divergence
    from the targets to the network's policy, which is what training reduces.
    ``value_baseline`` is the variance of the value targets: the value loss of
    predicting their mean everywhere. A value loss near it means the value head
    has learnt nothing beyond the average result.
    """
    logits, values = model(states)
    log_p = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pis * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_zs)
    loss = policy_loss + value_loss
    with torch.no_grad():
        target_entropy = -torch.special.xlogy(target_pis, target_pis).sum(dim=1).mean()
        policy_entropy = -(log_p.exp() * log_p).sum(dim=1).mean()
    return loss, {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "target_entropy": target_entropy.item(),
        "policy_kl": policy_loss.item() - target_entropy.item(),
        "policy_entropy": policy_entropy.item(),
        "value_baseline": target_zs.var(unbiased=False).item(),
    }


def format_losses(metrics: dict[str, float]) -> str:
    m = {k: metrics.get(k, float("nan")) for k in (
        "loss", "policy_loss", "target_entropy", "policy_kl", "policy_entropy",
        "value_loss", "value_baseline")}
    return (f"loss {m['loss']:.4f} (policy {m['policy_loss']:.4f} = target entropy "
            f"{m['target_entropy']:.4f} + KL {m['policy_kl']:.4f}, net entropy "
            f"{m['policy_entropy']:.4f}; value {m['value_loss']:.4f}, baseline "
            f"{m['value_baseline']:.4f})")


def steps_for(cfg: Config, new_examples: int) -> int:
    """Gradient steps for an iteration that added ``new_examples`` to the buffer."""
    if new_examples == 0:
        return 0
    if cfg.sample_reuse <= 0:
        return cfg.train_steps_per_iteration
    return max(1, min(cfg.train_steps_per_iteration,
                      math.ceil(cfg.sample_reuse * new_examples / cfg.batch_size)))


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

    buffer = ReplayBuffer(cfg.buffer_size)
    for it in range(start, cfg.iterations):
        t0 = time.time()
        log(f"Iteration {it + 1}/{cfg.iterations}")
        games = generate_games(model, cfg, log=log)
        new_examples = sum(len(g.examples) for g in games)
        buffer.extend(ex for g in games for ex in g.examples)
        summary, selfplay_metrics = summarize_games(games)
        log(f"  {summary}")

        metrics: dict[str, float] = {}
        steps = steps_for(cfg, new_examples)
        for _ in range(steps):
            step = train_step(model, optimizer, buffer, cfg.batch_size, device)
            for k, v in step.items():
                metrics[k] = metrics.get(k, 0.0) + v / steps
        metrics.update(selfplay_metrics, train_steps=steps)
        scheduler.step()

        path = save_checkpoint(checkpoint_path(cfg.checkpoint_dir, it + 1), model, cfg, it + 1,
                               optimizer, scheduler, metrics)
        log(f"  {format_losses(metrics)} | {steps} steps | buffer {len(buffer)} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s | saved {path}")
    return model
