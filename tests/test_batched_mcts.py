import chess
import numpy as np
import torch

from alphazero.batched_mcts import BatchedMCTS, generate_games
from alphazero.config import get_preset
from alphazero.game import ENCODER
from alphazero.mcts import select_action
from alphazero.network import AlphaZeroNet


def _untrained():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = get_preset("tiny")
    return AlphaZeroNet.from_config(cfg).eval(), cfg


def _all_nodes(node):
    yield node
    for child in node.children.values():
        yield from _all_nodes(child)


def test_visit_counts_cover_only_legal_moves():
    model, cfg = _untrained()
    board = chess.Board()
    visits = BatchedMCTS(model, cfg, batch_size=8, reuse_tree=False).search(board, num_simulations=30)
    assert visits.sum() == 30
    legal = {ENCODER.encode(m) for m in board.legal_moves}
    assert set(np.flatnonzero(visits)) <= legal


def test_finds_mate_in_one_with_untrained_network():
    model, cfg = _untrained()
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    visits = BatchedMCTS(model, cfg, batch_size=8).search(board, add_noise=False, num_simulations=200)
    move = ENCODER.decode(select_action(visits, temperature=0.0))
    assert move == chess.Move.from_uci("a1a8")


def test_virtual_loss_is_fully_removed():
    model, cfg = _untrained()
    board = chess.Board()
    mcts = BatchedMCTS(model, cfg, batch_size=16, reuse_tree=False)
    mcts.search(board, add_noise=False, num_simulations=64)
    root = mcts._root
    # Every node's visits equal the visits of its children plus its own evaluation.
    for node in _all_nodes(root):
        if node is root:
            assert node.visit_count == sum(c.visit_count for c in node.children.values())
        elif node.is_expanded():
            assert node.visit_count == 1 + sum(c.visit_count for c in node.children.values())
        assert abs(node.value_sum) <= node.visit_count + 1e-6


def test_tree_is_reused_after_a_move():
    model, cfg = _untrained()
    board = chess.Board()
    mcts = BatchedMCTS(model, cfg, batch_size=8)
    visits = mcts.search(board, add_noise=False, num_simulations=40)
    move = ENCODER.decode(select_action(visits, temperature=0.0))
    carried = mcts._root.children[ENCODER.encode(move)]
    board.push(move)
    mcts.search(board, add_noise=False, num_simulations=10)
    assert mcts._root is carried
    # A position from a different game starts a fresh tree.
    mcts.search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"), num_simulations=5)
    assert mcts._root is not carried


def test_batched_self_play_produces_examples():
    model, cfg = _untrained()
    games = generate_games(model, cfg, num_games=1, log=lambda _: None)
    assert games[0].examples
    planes, pi, z = games[0].examples[0]
    assert abs(pi.sum() - 1.0) < 1e-5 and -1.0 <= z <= 1.0  # stopped games: material score
