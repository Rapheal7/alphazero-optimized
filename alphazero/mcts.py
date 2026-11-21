"""Monte Carlo Tree Search guided by the policy/value network (PUCT)."""

from __future__ import annotations

import math

import chess
import numpy as np

from .config import Config
from .game import ACTION_SIZE, encode_board, legal_actions, terminal_value
from .network import AlphaZeroNet


class Node:
    """One edge/state in the search tree.

    ``value_sum`` is stored from the point of view of the player who made the
    move into this node, so a parent can maximise ``child.q`` directly.
    """

    __slots__ = ("move", "prior", "visit_count", "value_sum", "children")

    def __init__(self, move: chess.Move | None = None, prior: float = 0.0) -> None:
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)


def root_value(root: Node) -> float:
    """The search's value of ``root`` for its side to move: the visit-weighted
    mean of the children's Q (each stored for the root's side to move)."""
    visits = sum(child.visit_count for child in root.children.values())
    if not visits:
        return 0.0
    return sum(child.value_sum for child in root.children.values()) / visits


class MCTS:
    def __init__(self, model: AlphaZeroNet, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg
        # Value of the last searched position for its side to move (see root_value).
        self.root_value = 0.0

    def _expand(self, node: Node, board: chess.Board) -> float:
        """Add children with network priors. Returns the value for the side to move."""
        logits, value = self.model.predict(encode_board(board))
        actions = legal_actions(board)
        idx = np.fromiter((a for a, _ in actions), dtype=np.int64, count=len(actions))
        legal_logits = logits[idx]
        priors = np.exp(legal_logits - legal_logits.max())
        priors /= priors.sum()
        for (action, move), p in zip(actions, priors):
            node.children[action] = Node(move, float(p))
        return value

    def _add_dirichlet_noise(self, node: Node) -> None:
        eps = self.cfg.dirichlet_epsilon
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(node.children))
        for child, n in zip(node.children.values(), noise):
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _select(self, node: Node) -> Node:
        sqrt_n = math.sqrt(max(node.visit_count, 1))
        c = self.cfg.c_puct
        return max(
            node.children.values(),
            key=lambda ch: ch.q + c * ch.prior * sqrt_n / (1 + ch.visit_count),
        )

    def new_root(self, board: chess.Board, add_noise: bool = False) -> Node:
        """Expand and return a root for ``board``, ready for :meth:`simulate`."""
        if terminal_value(board) is not None:
            raise ValueError("search() called on a finished game")
        self.model.eval()
        root = Node()
        self._expand(root, board)
        if add_noise:
            self._add_dirichlet_noise(root)
        return root

    def simulate(self, root: Node, scratch: chess.Board) -> int:
        """Run one simulation from ``root``. ``scratch`` must hold the root position
        and is restored before returning. Returns the depth of the leaf reached."""
        node, path = root, [root]
        while node.is_expanded():
            node = self._select(node)
            scratch.push(node.move)
            path.append(node)

        value = terminal_value(scratch)
        if value is None:
            value = self._expand(node, scratch)

        # `value` is for the side to move at the leaf; the leaf's stats belong
        # to the player who moved into it, so start with the opposite sign.
        v = -value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v
        for _ in range(len(path) - 1):
            scratch.pop()
        return len(path) - 1

    def search(self, board: chess.Board, add_noise: bool = True,
               num_simulations: int | None = None) -> np.ndarray:
        """Run simulations from ``board`` and return root visit counts per action."""
        root = self.new_root(board, add_noise)
        scratch = board.copy()
        for _ in range(num_simulations or self.cfg.num_simulations):
            self.simulate(root, scratch)
        self.root_value = root_value(root)

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return visits


def select_action(visits: np.ndarray, temperature: float) -> int:
    """Pick an action from visit counts: proportional (temperature 1) or greedy (0)."""
    if temperature <= 1e-3 or visits.sum() == 0:
        best = np.flatnonzero(visits == visits.max())
        return int(np.random.choice(best))
    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()
    return int(np.random.choice(len(visits), p=probs))
