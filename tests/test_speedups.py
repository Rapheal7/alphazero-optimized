"""The self-play speed-ups must not change results: fast draw check, fast move
encoding, folded batch norm, no board copies in BatchedMCTS, and worker processes."""

import random

import chess
import numpy as np
import pytest
import torch

from alphazero.batched_mcts import BatchedMCTS, play_game
from alphazero.config import get_preset
from alphazero.game import ACTION_SIZE, ENCODER, terminal_value
from alphazero.network import AlphaZeroNet, inference_copy
from alphazero.parallel import SelfPlayPool, resolve_workers, seed_game
from alphazero.training import train


def _reference_value(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)


def test_terminal_value_matches_python_chess_claim_draw():
    rng = random.Random(0)
    checked = draws = 0
    for g in range(20):
        # Few pieces and mostly reversible moves, so repetitions and the fifty-move
        # rule come up often; half the games start near the fifty-move limit.
        fen = "4k3/8/8/8/8/8/4P3/R3K2R w KQ - 90 1" if g % 2 else chess.STARTING_FEN
        board = chess.Board(fen)
        while True:
            value = terminal_value(board)
            assert value == _reference_value(board), board.fen()
            checked += 1
            if value is not None:
                draws += value == 0.0
                break
            moves = list(board.legal_moves)
            quiet = [m for m in moves if not board.is_zeroing(m)]
            board.push(rng.choice(quiet if quiet and rng.random() < 0.9 else moves))
    assert checked > 500 and draws > 5


def test_threefold_claim_by_next_move_is_terminal():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)
    # Black can claim: ...Ng8 would be the third occurrence of the start position.
    assert not board.is_repetition(3) and board.can_claim_threefold_repetition()
    assert terminal_value(board) == 0.0


def test_move_encoder_fast_lookup_matches_uci_table():
    for index in range(ACTION_SIZE):
        move = ENCODER.decode(index)
        assert ENCODER.encode(move) == index == ENCODER.uci_to_index[move.uci()]
    with pytest.raises(KeyError):
        ENCODER.encode(chess.Move.null())


def test_inference_copy_matches_model_and_leaves_it_untouched():
    torch.manual_seed(0)
    model = AlphaZeroNet(2, 16)
    model.train()
    for _ in range(3):  # give the batch norms non-trivial running statistics
        model(torch.randn(32, 18, 8, 8))
    model.eval()
    fused = inference_copy(model)
    x = torch.randn(5, 18, 8, 8)
    with torch.no_grad():
        for a, b in zip(model(x), fused(x)):
            assert torch.allclose(a, b, atol=1e-5)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())


def test_batched_search_leaves_the_board_unchanged():
    torch.manual_seed(0)
    cfg = get_preset("tiny")
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        board.push_uci(uci)
    fen, stack = board.fen(), list(board.move_stack)
    BatchedMCTS(AlphaZeroNet.from_config(cfg).eval(), cfg).search(board, num_simulations=40)
    assert board.fen() == fen and board.move_stack == stack


def _moves(games):
    return [g.moves for g in games]


def test_pool_games_match_in_process_play_and_follow_new_weights():
    cfg = get_preset("tiny").updated(max_moves=12)
    torch.manual_seed(0)
    first, second = AlphaZeroNet.from_config(cfg).eval(), AlphaZeroNet.from_config(cfg).eval()

    def expected(model, seeds):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # as in the workers, so float results are identical
        try:
            fused, games = inference_copy(model), []
            for s in seeds:
                seed_game(int(s))
                games.append(play_game(fused, cfg))
            return games
        finally:
            torch.set_num_threads(threads)

    with SelfPlayPool(2) as pool:
        for model in (first, second):
            np.random.seed(3)
            seeds = np.random.randint(0, 2**31 - 1, size=3)
            np.random.seed(3)
            games = pool.generate_games(model, cfg, 3, log=lambda _: None)
            assert _moves(games) == _moves(expected(model, seeds))
            assert all(g.examples and abs(g.examples[0][1].sum() - 1) < 1e-5 for g in games)


def test_pool_results_do_not_depend_on_worker_count():
    cfg = get_preset("tiny").updated(max_moves=10)
    torch.manual_seed(1)
    model = AlphaZeroNet.from_config(cfg).eval()
    results = []
    for workers in (1, 3):
        with SelfPlayPool(workers) as pool:
            np.random.seed(7)
            results.append(_moves(pool.generate_games(model, cfg, 4, log=lambda _: None)))
    assert results[0] == results[1]


def test_resolve_workers():
    cpu, gpu = torch.device("cpu"), torch.device("cuda")
    assert resolve_workers(0, gpu, 10) == 1
    assert resolve_workers(0, cpu, 1000) >= 1
    assert resolve_workers(8, cpu, 3) == 3
    assert resolve_workers(2, gpu, 10) == 2


def test_training_with_worker_processes(tmp_path):
    cfg = get_preset("tiny").updated(checkpoint_dir=str(tmp_path), seed=0,
                                     selfplay_workers=2, games_per_iteration=2, iterations=2)
    lines = []
    train(cfg, log=lines.append)
    assert (tmp_path / "checkpoint_iter_0002.pt").exists()
    assert any("2 worker processes" in line for line in lines)
    assert sum("self-play game" in line for line in lines) == 4


def test_encode_board_piece_planes_match_piece_map():
    from alphazero.game import _PIECE_PLANE, encode_board

    rng = random.Random(1)
    board = chess.Board()
    for _ in range(300):
        if board.is_game_over():
            board = chess.Board()
        expected = np.zeros((12, 8, 8), dtype=np.float32)
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            expected[_PIECE_PLANE[(piece.color, piece.piece_type)], 7 - rank, file] = 1.0
        assert np.array_equal(encode_board(board)[:12], expected)
        board.push(rng.choice(list(board.legal_moves)))
