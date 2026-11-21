"""Faster MCTS: batched leaf evaluation, tree reuse and half precision on GPU.

A drop-in alternative to :class:`alphazero.mcts.MCTS` (same ``search`` signature,
same ``Node`` tree), kept in its own module so the original stays untouched.

- **Batched leaves with virtual loss.** Each step walks up to ``batch_size``
  paths down the tree before calling the network. Every node on a pending path
  gets a temporary visit and a loss, so the next path is steered elsewhere. All
  the new leaves are then evaluated in a single forward pass, the virtual loss
  is removed and the real values are backed up.
- **Tree reuse.** After a move is played, the subtree under it becomes the next
  root, so its visits aren't thrown away.
- **Inference mode and fp16.** Network calls run under ``torch.inference_mode``,
  with float16 autocast on CUDA.

``play_game`` and ``generate_games`` mirror :mod:`alphazero.selfplay` so the
training loop can switch over by changing one import.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import chess
import numpy as np
import torch

from .config import Config
from . import selfplay
from .game import ACTION_SIZE, ENCODER, encode_board, legal_actions, terminal_value
from .mcts import Node, root_value
from .network import AlphaZeroNet
from .selfplay import GameRecord

# Loss added per pending visit, in value units ([-1, 1]).
VIRTUAL_LOSS = 1.0


class BatchedMCTS:
    def __init__(self, model: AlphaZeroNet, cfg: Config, batch_size: int = 8,
                 reuse_tree: bool = True) -> None:
        self.model = model
        self.cfg = cfg
        self.batch_size = max(1, batch_size)
        self.reuse_tree = reuse_tree
        self._root: Node | None = None
        self._root_start_fen: str | None = None
        self._root_moves: list[chess.Move] = []
        # Value of the last searched position for its side to move (see root_value).
        self.root_value = 0.0

    # --- network -----------------------------------------------------------

    def _evaluate(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate several positions in one forward pass. Returns (logits, values)."""
        device = next(self.model.parameters()).device
        x = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(device)
        autocast = (torch.autocast("cuda", dtype=torch.float16)
                    if device.type == "cuda" else contextlib.nullcontext())
        with torch.inference_mode(), autocast:
            logits, values = self.model(x)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()[:, 0]

    @staticmethod
    def _expand(node: Node, board: chess.Board, logits: np.ndarray) -> None:
        actions = legal_actions(board)
        idx = np.fromiter((a for a, _ in actions), dtype=np.int64, count=len(actions))
        legal_logits = logits[idx]
        priors = np.exp(legal_logits - legal_logits.max())
        priors /= priors.sum()
        for (action, move), p in zip(actions, priors):
            node.children[action] = Node(move, float(p))

    # --- tree --------------------------------------------------------------

    def _select(self, node: Node) -> Node:
        sqrt_n = np.sqrt(max(node.visit_count, 1))
        c = self.cfg.c_puct
        return max(
            node.children.values(),
            key=lambda ch: ch.q + c * ch.prior * sqrt_n / (1 + ch.visit_count),
        )

    def _add_dirichlet_noise(self, node: Node) -> None:
        eps = self.cfg.dirichlet_epsilon
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(node.children))
        for child, n in zip(node.children.values(), noise):
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _reused_root(self, board: chess.Board) -> Node | None:
        """The stored subtree for ``board``, if ``board`` continues the last searched game."""
        if self._root is None or board.root().fen() != self._root_start_fen:
            return None
        stack, known = board.move_stack, self._root_moves
        if len(stack) < len(known) or stack[:len(known)] != known:
            return None
        node = self._root
        for move in stack[len(known):]:
            node = node.children.get(ENCODER.encode(move))
            if node is None:
                return None
        return node if node.is_expanded() else None

    def new_root(self, board: chess.Board, add_noise: bool = False) -> Node:
        """Return an expanded root for ``board``, reusing the last tree when possible."""
        if terminal_value(board) is not None:
            raise ValueError("search() called on a finished game")
        self.model.eval()
        root = self._reused_root(board) if self.reuse_tree else None
        if root is None:
            root = Node()
            logits, _ = self._evaluate([board])
            self._expand(root, board, logits[0])
        if add_noise:
            self._add_dirichlet_noise(root)
        self._root = root
        self._root_start_fen = board.root().fen()
        self._root_moves = list(board.move_stack)
        return root

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        # `value` is for the side to move at the leaf; the leaf's stats belong
        # to the player who moved into it, so start with the opposite sign.
        v = -value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v

    @staticmethod
    def _virtual_loss(path: list[Node], sign: int) -> None:
        for n in path:
            n.visit_count += sign
            n.value_sum -= sign * VIRTUAL_LOSS

    def simulate_batch(self, root: Node, scratch: chess.Board, max_leaves: int) -> list[int]:
        """Run up to ``max_leaves`` simulations with one network call.

        ``scratch`` must hold the root position and is restored before returning.
        Returns the depth of each leaf reached (one entry per finished simulation).
        """
        pending: list[tuple[list[Node], chess.Board]] = []
        pending_ids: set[int] = set()
        depths: list[int] = []

        while len(depths) + len(pending) < max_leaves:
            node, path = root, [root]
            while node.is_expanded():
                node = self._select(node)
                scratch.push(node.move)
                path.append(node)
            depth = len(path) - 1
            leaf_board = scratch.copy()
            for _ in range(depth):
                scratch.pop()

            value = terminal_value(leaf_board)
            if value is not None:
                self._backup(path, value)
                depths.append(depth)
            elif id(node) in pending_ids:
                break  # the tree funnels back to a leaf already queued: evaluate now
            else:
                self._virtual_loss(path, +1)
                pending.append((path, leaf_board))
                pending_ids.add(id(node))

        if pending:
            logits, values = self._evaluate([b for _, b in pending])
            for (path, leaf_board), lg, v in zip(pending, logits, values):
                self._virtual_loss(path, -1)
                self._expand(path[-1], leaf_board, lg)
                self._backup(path, float(v))
                depths.append(len(path) - 1)
        return depths

    def search(self, board: chess.Board, add_noise: bool = True,
               num_simulations: int | None = None) -> np.ndarray:
        """Run simulations from ``board`` and return root visit counts per action.

        With tree reuse, visits carried over from the previous search are included,
        so the counts can add up to more than ``num_simulations``.
        """
        root = self.new_root(board, add_noise)
        scratch = board.copy()
        remaining = num_simulations or self.cfg.num_simulations
        while remaining > 0:
            remaining -= len(self.simulate_batch(root, scratch, min(self.batch_size, remaining)))
        self.root_value = root_value(root)

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return visits


# --- self-play with the batched search --------------------------------------

def play_game(model: AlphaZeroNet, cfg: Config, batch_size: int = 8) -> GameRecord:
    """Same as :func:`alphazero.selfplay.play_game`, using :class:`BatchedMCTS`.

    Without tree reuse: with playout cap randomisation, a reused subtree would
    mix a fast search's visits into the next full search's policy target.
    """
    return selfplay.play_game(model, cfg,
                              BatchedMCTS(model, cfg, batch_size=batch_size, reuse_tree=False))


def generate_games(
    model: AlphaZeroNet,
    cfg: Config,
    num_games: int | None = None,
    log: Callable[[str], None] = print,
    batch_size: int = 8,
) -> list[GameRecord]:
    return selfplay.generate_games(model, cfg, num_games, log,
                                   play=lambda m, c: play_game(m, c, batch_size))
