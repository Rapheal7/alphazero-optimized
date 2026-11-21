"""Players and matches: model vs random mover, model vs Stockfish, human vs model."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import chess
import chess.engine

from .config import Config
from .game import ENCODER, game_outcome
from .mcts import MCTS, select_action
from .network import AlphaZeroNet


class Player(Protocol):
    name: str

    def choose_move(self, board: chess.Board) -> chess.Move: ...


class RandomPlayer:
    name = "Random"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class MCTSPlayer:
    """Plays the most-visited move after search (no Dirichlet noise)."""

    def __init__(self, model: AlphaZeroNet, cfg: Config, num_simulations: int | None = None,
                 name: str = "AlphaZero") -> None:
        self.mcts = MCTS(model, cfg)
        self.num_simulations = num_simulations or cfg.num_simulations
        self.name = name

    def choose_move(self, board: chess.Board) -> chess.Move:
        visits = self.mcts.search(board, add_noise=False, num_simulations=self.num_simulations)
        return ENCODER.decode(select_action(visits, temperature=0.0))


_STOCKFISH_CANDIDATES = ("/usr/games/stockfish", "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish")


def find_stockfish(path: str | None = None) -> str | None:
    if path:
        return path if Path(path).exists() or shutil.which(path) else None
    found = shutil.which("stockfish")
    if found:
        return found
    return next((p for p in _STOCKFISH_CANDIDATES if Path(p).exists()), None)


class StockfishPlayer:
    """Stockfish via UCI. Use as a context manager so the engine process is closed."""

    def __init__(self, path: str | None = None, skill_level: int = 0, time_limit: float = 0.05) -> None:
        resolved = find_stockfish(path)
        if resolved is None:
            raise FileNotFoundError(
                "Stockfish not found. Install it (brew install stockfish / apt-get install stockfish) "
                "or pass --stockfish-path."
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(resolved)
        self.engine.configure({"Skill Level": skill_level})
        self.limit = chess.engine.Limit(time=time_limit)
        self.name = f"Stockfish(skill {skill_level})"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "StockfishPlayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def play_match(white: Player, black: Player, max_moves: int = 300,
               on_move: Callable[[chess.Board, chess.Move], None] | None = None) -> chess.Board:
    """Play one game and return the final board. Over ``max_moves`` plies counts as a draw."""
    board = chess.Board()
    while game_outcome(board) is None and board.ply() < max_moves:
        player = white if board.turn == chess.WHITE else black
        move = player.choose_move(board)
        if on_move:
            on_move(board, move)
        board.push(move)
    return board


def game_result(board: chess.Board) -> str:
    outcome = game_outcome(board)
    return outcome.result() if outcome else "1/2-1/2"


@dataclass
class MatchStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def __str__(self) -> str:
        return f"+{self.wins} ={self.draws} -{self.losses} (score {self.score:.0%} over {self.games} games)"


def evaluate(player: Player, opponent: Player, num_games: int, max_moves: int = 300,
             log: Callable[[str], None] = print) -> MatchStats:
    """Play ``num_games``, alternating colours (from the X1 notebook's evaluate_model)."""
    stats = MatchStats()
    for i in range(num_games):
        player_white = i % 2 == 0
        white, black = (player, opponent) if player_white else (opponent, player)
        board = play_match(white, black, max_moves=max_moves)
        result = game_result(board)
        if result == "1/2-1/2":
            stats.draws += 1
        elif (result == "1-0") == player_white:
            stats.wins += 1
        else:
            stats.losses += 1
        log(f"  game {i + 1}/{num_games}: {white.name} (W) vs {black.name} (B) -> {result} "
            f"in {board.ply()} plies")
    return stats


def parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    """Accept SAN (``Nf3``) or UCI (``g1f3``). Returns None if illegal or unparseable."""
    text = text.strip()
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    return None
