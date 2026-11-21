"""Self-play: the current network plays itself to produce training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from .config import Config
from .game import (ENCODER, adjudicated_value, encode_board, game_outcome, material_balance,
                   terminal_value)
from .mcts import MCTS, select_action
from .network import AlphaZeroNet

# (encoded board, MCTS visit distribution, final result for the side to move)
Example = tuple[np.ndarray, np.ndarray, float]


@dataclass
class GameRecord:
    examples: list[Example] = field(default_factory=list)
    result: str = "*"
    num_plies: int = 0
    # Stopped at cfg.max_moves and scored by material.
    adjudicated: bool = False
    moves: list[str] = field(default_factory=list)
    # The score the training targets were built from, for white: 1, 0 or -1 for
    # a finished game, the material score for an adjudicated one.
    white_value: float = 0.0
    material: int = 0

    def describe(self) -> str:
        note = ""
        if self.adjudicated:
            note = (f" (move limit, material {self.material:+d}, "
                    f"scored {self.white_value:+.2f} for white)")
        return f"{self.result} in {self.num_plies} plies{note}"


def play_game(model: AlphaZeroNet, cfg: Config, mcts=None) -> GameRecord:
    """Play one self-play game. ``mcts`` defaults to a plain :class:`MCTS`; any
    object with the same ``search`` method works (e.g. ``BatchedMCTS``)."""
    board = chess.Board()
    mcts = mcts or MCTS(model, cfg)
    history: list[tuple[np.ndarray, np.ndarray, bool]] = []
    record = GameRecord()

    while terminal_value(board) is None and board.ply() < cfg.max_moves:
        visits = mcts.search(board, add_noise=True)
        pi = visits / visits.sum()
        temperature = 1.0 if board.ply() < cfg.temperature_moves else 0.0
        move = ENCODER.decode(select_action(visits, temperature))
        history.append((encode_board(board), pi, board.turn))
        record.moves.append(board.san(move))
        board.push(move)

    outcome = game_outcome(board)
    record.adjudicated = outcome is None
    record.result = outcome.result() if outcome else "1/2-1/2"
    record.num_plies = board.ply()
    record.material = material_balance(board)
    if outcome is None:
        record.white_value = adjudicated_value(board, cfg.material_scale)
    elif outcome.winner is not None:
        record.white_value = 1.0 if outcome.winner == chess.WHITE else -1.0
    for planes, pi, turn in history:
        z = record.white_value if turn == chess.WHITE else -record.white_value
        record.examples.append((planes, pi, z))
    return record


def generate_games(
    model: AlphaZeroNet,
    cfg: Config,
    num_games: int | None = None,
    log: Callable[[str], None] = print,
    play: Callable[[AlphaZeroNet, Config], GameRecord] = play_game,
) -> list[GameRecord]:
    games = []
    n = num_games or cfg.games_per_iteration
    for i in range(n):
        rec = play(model, cfg)
        log(f"  self-play game {i + 1}/{n}: {rec.describe()}")
        games.append(rec)
    return games
