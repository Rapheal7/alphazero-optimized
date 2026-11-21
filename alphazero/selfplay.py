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
    # chess.Termination name in lower case, "move_limit" or "resignation".
    termination: str = ""
    resigned: chess.Color | None = None
    # Games played out with resigning turned off (cfg.resign_check_fraction):
    # the side that would have resigned, and at which ply.
    resign_check: bool = False
    would_resign: chess.Color | None = None
    would_resign_ply: int = 0
    # Moves searched with the full num_simulations (the ones in ``examples``).
    full_searches: int = 0

    @property
    def false_resignation(self) -> bool:
        """A resign-check game where the side that would have resigned didn't lose."""
        if self.would_resign is None:
            return False
        return self.white_value != (-1.0 if self.would_resign == chess.WHITE else 1.0)

    def describe(self) -> str:
        note = ""
        if self.adjudicated:
            note = (f" (move limit, material {self.material:+d}, "
                    f"scored {self.white_value:+.2f} for white)")
        elif self.resigned is not None:
            note = f" ({chess.COLOR_NAMES[self.resigned]} resigned)"
        if self.would_resign is not None:
            outcome = "and did not lose" if self.false_resignation else "and lost"
            note += (f" (no-resign game: {chess.COLOR_NAMES[self.would_resign]} would have "
                     f"resigned at ply {self.would_resign_ply} {outcome})")
        return f"{self.result} in {self.num_plies} plies, {len(self.examples)} examples{note}"


def play_game(model: AlphaZeroNet, cfg: Config, mcts=None) -> GameRecord:
    """Play one self-play game. ``mcts`` defaults to a plain :class:`MCTS`; any
    object with the same ``search`` method works (e.g. ``BatchedMCTS``)."""
    board = chess.Board()
    mcts = mcts or MCTS(model, cfg)
    history: list[tuple[np.ndarray, np.ndarray, bool]] = []
    record = GameRecord()
    can_resign = cfg.resign_threshold > -1
    if can_resign and np.random.random() < cfg.resign_check_fraction:
        can_resign, record.resign_check = False, True
    low_streak = {chess.WHITE: 0, chess.BLACK: 0}

    while terminal_value(board) is None and board.ply() < cfg.max_moves:
        full = np.random.random() < cfg.full_search_fraction
        sims = cfg.num_simulations if full else min(cfg.fast_simulations, cfg.num_simulations)
        visits = mcts.search(board, add_noise=full, num_simulations=sims)
        if mcts.root_value < cfg.resign_threshold:
            low_streak[board.turn] += 1
        else:
            low_streak[board.turn] = 0
        if low_streak[board.turn] >= cfg.resign_moves:
            if can_resign:
                record.resigned = board.turn
                break
            if record.resign_check and record.would_resign is None:
                record.would_resign, record.would_resign_ply = board.turn, board.ply()
        temperature = 1.0 if board.ply() < cfg.temperature_moves else 0.0
        move = ENCODER.decode(select_action(visits, temperature))
        if full:
            history.append((encode_board(board), visits / visits.sum(), board.turn))
            record.full_searches += 1
        record.moves.append(board.san(move))
        board.push(move)

    record.num_plies = board.ply()
    record.material = material_balance(board)
    outcome = game_outcome(board)
    if record.resigned is not None:
        record.termination = "resignation"
        record.result = "0-1" if record.resigned == chess.WHITE else "1-0"
        record.white_value = -1.0 if record.resigned == chess.WHITE else 1.0
    elif outcome is None:
        record.adjudicated = True
        record.termination = "move_limit"
        record.result = "1/2-1/2"
        record.white_value = adjudicated_value(board, cfg.material_scale)
    else:
        record.termination = outcome.termination.name.lower()
        record.result = outcome.result()
        if outcome.winner is not None:
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
