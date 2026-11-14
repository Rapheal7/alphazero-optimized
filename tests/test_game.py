import chess
import numpy as np
import pytest

from alphazero.game import (
    ACTION_SIZE,
    ENCODER,
    NUM_PLANES,
    encode_board,
    legal_action_mask,
    terminal_value,
)

POSITIONS = [
    chess.STARTING_FEN,
    # White can promote / underpromote on b8 and by capturing on a8 / c8.
    "r1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    # Black promotion on g1, plus castling rights for both sides.
    "r3k2r/8/8/8/8/8/6p1/R3K2R b KQkq - 0 1",
    # En passant available (d5xe6).
    "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2",
]


def test_action_space_is_unique_and_stable():
    assert ACTION_SIZE == 64 * 63 + 2 * 8 * 8 * 4  # 4544
    assert len(set(ENCODER.index_to_uci)) == ACTION_SIZE


@pytest.mark.parametrize("fen", POSITIONS)
def test_every_legal_move_round_trips(fen):
    board = chess.Board(fen)
    seen = set()
    for move in board.legal_moves:
        idx = ENCODER.encode(move)
        assert ENCODER.decode(idx) == move
        seen.add(idx)
    assert len(seen) == board.legal_moves.count()  # underpromotions don't collide
    assert legal_action_mask(board).sum() == board.legal_moves.count()


def test_encode_board_planes():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (NUM_PLANES, 8, 8) and planes.dtype == np.float32
    assert planes[0].sum() == 8 and planes[6].sum() == 8  # pawns
    assert planes[0, 6].sum() == 8  # white pawns on rank 2 -> row 6
    assert planes[12].all()  # white to move
    board.push_san("e4")
    assert not encode_board(board)[12].any()


def test_terminal_values():
    assert terminal_value(chess.Board()) is None
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert terminal_value(mated) == -1.0  # side to move has been mated
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert terminal_value(stalemate) == 0.0
