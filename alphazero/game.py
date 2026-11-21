"""Chess rules glue: board encoding, move <-> action index mapping, terminal values.

The rules themselves come from python-chess. This module only translates
between ``chess.Board`` / ``chess.Move`` and the arrays the network uses.
"""

from __future__ import annotations

import chess
import numpy as np

# 12 piece planes + side to move + 4 castling rights + halfmove clock.
NUM_PLANES = 18

_PROMOTION_PIECES = "qrbn"


class MoveEncoder:
    """Fixed mapping between UCI move strings and policy indices.

    Every from-square/to-square pair gets an index (64 * 63 = 4032), and every
    possible promotion move (rank 7 -> 8 and rank 2 -> 1, to any file) gets one
    extra index per promotion piece. Underpromotions keep their own indices,
    which the X1 notebook's ``from * 64 + to`` scheme could not represent.
    Adapted from ``ChessEncoder`` in AlphaZeroChess2.
    """

    def __init__(self) -> None:
        self.index_to_uci: list[str] = []
        self.uci_to_index: dict[str, int] = {}
        files, ranks = "abcdefgh", "12345678"
        for f1 in files:
            for r1 in ranks:
                for f2 in files:
                    for r2 in ranks:
                        if (f1, r1) == (f2, r2):
                            continue
                        self._add(f"{f1}{r1}{f2}{r2}")
                        if (r1, r2) in (("7", "8"), ("2", "1")):
                            for piece in _PROMOTION_PIECES:
                                self._add(f"{f1}{r1}{f2}{r2}{piece}")
        self._moves = [chess.Move.from_uci(u) for u in self.index_to_uci]

    def _add(self, uci: str) -> None:
        self.uci_to_index[uci] = len(self.index_to_uci)
        self.index_to_uci.append(uci)

    @property
    def action_size(self) -> int:
        return len(self.index_to_uci)

    def encode(self, move: chess.Move) -> int:
        return self.uci_to_index[move.uci()]

    def decode(self, index: int) -> chess.Move:
        return self._moves[index]


ENCODER = MoveEncoder()
ACTION_SIZE = ENCODER.action_size

_PIECE_PLANE = {
    (color, piece_type): (0 if color == chess.WHITE else 6) + piece_type - 1
    for color in chess.COLORS
    for piece_type in chess.PIECE_TYPES
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as a ``(NUM_PLANES, 8, 8)`` float32 array.

    Planes 0-5: white P N B R Q K. Planes 6-11: black p n b r q k.
    Row 0 is rank 8 (the board as printed). Plane 12 is 1 when white is to move.
    Planes 13-16 are castling rights (white K/Q side, black K/Q side). Plane 17 is
    the halfmove clock divided by 100.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        planes[_PIECE_PLANE[(piece.color, piece.piece_type)], 7 - rank, file] = 1.0
    if board.turn == chess.WHITE:
        planes[12] = 1.0
    planes[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(board.has_queenside_castling_rights(chess.BLACK))
    planes[17] = min(board.halfmove_clock, 100) / 100.0
    return planes


def legal_actions(board: chess.Board) -> list[tuple[int, chess.Move]]:
    return [(ENCODER.encode(m), m) for m in board.legal_moves]


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[ENCODER.encode(move)] = True
    return mask


def game_outcome(board: chess.Board) -> chess.Outcome | None:
    """How the game has ended, or ``None`` if it hasn't.

    Threefold repetition and the fifty-move rule end the game as soon as they
    have actually happened, rather than waiting for fivefold / seventy-five
    moves. Unlike ``board.outcome(claim_draw=True)``, a position isn't a draw
    just because the side to move *could* repeat or reach move fifty with its
    next move: that would score winning positions as draws whenever a
    repeating move exists, and pull self-play towards draws.
    """
    outcome = board.outcome()
    if outcome is not None:
        return outcome
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    if board.is_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return None


_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_balance(board: chess.Board) -> int:
    """White's material minus black's, in pawns (P=1, N=B=3, R=5, Q=9)."""
    return sum(value * (len(board.pieces(piece_type, chess.WHITE))
                        - len(board.pieces(piece_type, chess.BLACK)))
               for piece_type, value in _PIECE_VALUES.items())


def adjudicated_value(board: chess.Board, material_scale: float) -> float:
    """Score for white of a game stopped before it ended: ``tanh(material / scale)``.

    A pawn up is worth about +0.2 with the default scale of 5, a queen up about
    +0.95. ``material_scale <= 0`` scores every stopped game as a draw.
    """
    if material_scale <= 0:
        return 0.0
    return float(np.tanh(material_balance(board) / material_scale))


def terminal_value(board: chess.Board) -> float | None:
    """Value of a finished game for the side to move, or ``None`` if it isn't over.

    See :func:`game_outcome` for when repetitions and the fifty-move rule count.
    """
    outcome = game_outcome(board)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    # The side to move can only have lost (it has been checkmated).
    return 1.0 if outcome.winner == board.turn else -1.0
