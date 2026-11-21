import chess
import numpy as np
import pytest
import torch

from alphazero.config import get_preset
from alphazero.game import ACTION_SIZE, ENCODER
from alphazero.network import AlphaZeroNet
from alphazero.selfplay import play_game


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = get_preset("tiny")
    return AlphaZeroNet.from_config(cfg).eval(), cfg


def _targets_match_white_value(rec):
    for planes, _, z in rec.examples:
        white_to_move = planes[12].all()
        assert z == pytest.approx(rec.white_value if white_to_move else -rec.white_value)


def test_game_stopped_at_move_limit_is_scored_by_material(tiny, monkeypatch):
    model, cfg = tiny
    # Stop after 6 plies with white a queen up, whatever was played.
    monkeypatch.setattr("alphazero.selfplay.material_balance", lambda board: 9)
    monkeypatch.setattr("alphazero.selfplay.adjudicated_value",
                        lambda board, scale: float(np.tanh(9 / scale)))
    rec = play_game(model, cfg.updated(max_moves=6))
    assert rec.adjudicated and rec.num_plies == 6
    assert rec.white_value == pytest.approx(np.tanh(9 / cfg.material_scale))
    assert "material +9" in rec.describe()
    _targets_match_white_value(rec)


def test_material_scale_zero_scores_stopped_games_as_draws(tiny):
    model, cfg = tiny
    rec = play_game(model, cfg.updated(max_moves=6, material_scale=0.0))
    assert rec.adjudicated and rec.white_value == 0.0
    assert all(z == 0.0 for _, _, z in rec.examples)


class ScriptedSearch:
    """Stands in for MCTS: every legal move gets one visit, and the root value
    for the side to move comes from ``value(board)``."""

    def __init__(self, value):
        self.value = value
        self.root_value = 0.0

    def search(self, board, add_noise=True, num_simulations=None):
        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in board.legal_moves:
            visits[ENCODER.encode(move)] = 1.0
        self.root_value = self.value(board)
        return visits


def black_is_lost(board):
    return -0.95 if board.turn == chess.BLACK else 0.5


def test_side_resigns_after_resign_moves_low_values(tiny):
    model, cfg = tiny
    cfg = cfg.updated(resign_check_fraction=0.0, resign_moves=3)
    rec = play_game(model, cfg, ScriptedSearch(black_is_lost))
    # Black is to move at board.ply() 1, 3 and 5, and resigns at the third.
    assert rec.resigned == chess.BLACK and rec.termination == "resignation"
    assert rec.result == "1-0" and rec.num_plies == 5 and rec.white_value == 1.0
    assert len(rec.examples) == 5
    _targets_match_white_value(rec)


def test_low_value_streak_must_be_unbroken(tiny):
    model, cfg = tiny
    cfg = cfg.updated(resign_check_fraction=0.0, resign_moves=3)
    flicker = lambda board: -0.95 if board.ply() % 4 == 1 else 0.0  # black: low, ok, low...
    rec = play_game(model, cfg, ScriptedSearch(flicker))
    assert rec.resigned is None


def test_resign_check_games_play_on_and_record_false_resignations(tiny):
    model, cfg = tiny
    cfg = cfg.updated(resign_check_fraction=1.0, resign_moves=3)
    rec = play_game(model, cfg, ScriptedSearch(black_is_lost))
    assert rec.resign_check and rec.resigned is None
    assert rec.would_resign == chess.BLACK and rec.would_resign_ply == 5
    assert rec.num_plies > 5
    assert rec.false_resignation == (rec.white_value != 1.0)
    assert "would have resigned at ply 5" in rec.describe()


def test_resign_threshold_minus_one_never_resigns(tiny):
    model, cfg = tiny
    cfg = cfg.updated(resign_threshold=-1.0, resign_check_fraction=0.0)
    rec = play_game(model, cfg, ScriptedSearch(lambda board: -1.0))
    assert rec.resigned is None and not rec.resign_check


def test_search_reports_root_value_for_side_to_move(tiny):
    from alphazero.batched_mcts import BatchedMCTS
    from alphazero.mcts import MCTS

    model, cfg = tiny
    # White mates with Ra8#: the search should see the position as winning.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    for search in (MCTS(model, cfg), BatchedMCTS(model, cfg)):
        search.search(board, add_noise=False, num_simulations=200)
        assert search.root_value > 0.5
