import chess
import numpy as np
import torch

from alphazero.config import get_preset
from alphazero.game import ENCODER
from alphazero.mcts import MCTS, select_action
from alphazero.network import AlphaZeroNet


def _untrained():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = get_preset("tiny")
    return AlphaZeroNet.from_config(cfg).eval(), cfg


def test_visit_counts_cover_only_legal_moves():
    model, cfg = _untrained()
    board = chess.Board()
    visits = MCTS(model, cfg).search(board, num_simulations=30)
    assert visits.sum() == 30
    legal = {ENCODER.encode(m) for m in board.legal_moves}
    assert set(np.flatnonzero(visits)) <= legal


def test_finds_mate_in_one_with_untrained_network():
    # Back-rank mate: Ra8#. Search alone must find it; the network knows nothing.
    model, cfg = _untrained()
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    visits = MCTS(model, cfg).search(board, add_noise=False, num_simulations=200)
    move = ENCODER.decode(select_action(visits, temperature=0.0))
    assert move == chess.Move.from_uci("a1a8")
