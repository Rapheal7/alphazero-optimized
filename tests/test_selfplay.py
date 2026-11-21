import chess
import numpy as np
import pytest
import torch

from alphazero.config import get_preset
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
