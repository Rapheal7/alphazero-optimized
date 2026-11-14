"""Self-play: the current network plays itself to produce training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from .config import Config
from .game import ENCODER, encode_board, terminal_value
from .mcts import MCTS, select_action
from .network import AlphaZeroNet

# (encoded board, MCTS visit distribution, final result for the side to move)
Example = tuple[np.ndarray, np.ndarray, float]


@dataclass
class GameRecord:
    examples: list[Example] = field(default_factory=list)
    result: str = "*"
    num_plies: int = 0
    adjudicated: bool = False
    moves: list[str] = field(default_factory=list)


def play_game(model: AlphaZeroNet, cfg: Config) -> GameRecord:
    board = chess.Board()
    mcts = MCTS(model, cfg)
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

    outcome = board.outcome(claim_draw=True)
    winner = outcome.winner if outcome else None
    record.adjudicated = outcome is None
    record.result = outcome.result() if outcome else "1/2-1/2"
    record.num_plies = board.ply()
    for planes, pi, turn in history:
        z = 0.0 if winner is None else (1.0 if turn == winner else -1.0)
        record.examples.append((planes, pi, z))
    return record


def generate_games(
    model: AlphaZeroNet,
    cfg: Config,
    num_games: int | None = None,
    log: Callable[[str], None] = print,
) -> list[GameRecord]:
    games = []
    n = num_games or cfg.games_per_iteration
    for i in range(n):
        rec = play_game(model, cfg)
        note = " (move limit, scored as draw)" if rec.adjudicated else ""
        log(f"  self-play game {i + 1}/{n}: {rec.result} in {rec.num_plies} plies{note}")
        games.append(rec)
    return games
