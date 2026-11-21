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
        # encode() is called for every legal move of every expanded node, so look
        # indices up by (from, to, promotion) arithmetic instead of move.uci().
        self._by_squares = [-1] * (64 * 64 * 7)
        for index, move in enumerate(self._moves):
            self._by_squares[self._key(move)] = index

    @staticmethod
    def _key(move: chess.Move) -> int:
        return (move.from_square * 64 + move.to_square) * 7 + (move.promotion or 0)

    def _add(self, uci: str) -> None:
        self.uci_to_index[uci] = len(self.index_to_uci)
        self.index_to_uci.append(uci)

    @property
    def action_size(self) -> int:
        return len(self.index_to_uci)

    def encode(self, move: chess.Move) -> int:
        index = self._by_squares[self._key(move)]
        if index < 0:
            raise KeyError(move.uci())
        return index

    def decode(self, index: int) -> chess.Move:
        return self._moves[index]


ENCODER = MoveEncoder()
ACTION_SIZE = ENCODER.action_size

_PIECE_PLANE = {
    (color, piece_type): (0 if color == chess.WHITE else 6) + piece_type - 1
    for color in chess.COLORS
    for piece_type in chess.PIECE_TYPES
}


_PIECE_ORDER = sorted(_PIECE_PLANE, key=_PIECE_PLANE.get)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as a ``(NUM_PLANES, 8, 8)`` float32 array.

    Planes 0-5: white P N B R Q K. Planes 6-11: black p n b r q k.
    Row 0 is rank 8 (the board as printed). Plane 12 is 1 when white is to move.
    Planes 13-16 are castling rights (white K/Q side, black K/Q side). Plane 17 is
    the halfmove clock divided by 100.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    # Unpack the 12 piece bitboards (bit i = square i = a1, b1, ... h8) in one go,
    # then flip the ranks so row 0 is rank 8. Much faster than board.piece_map().
    bitboards = np.array([board.pieces_mask(pt, color) for color, pt in _PIECE_ORDER],
                         dtype="<u8")  # little-endian, so bit i is square i
    bits = np.unpackbits(bitboards.view(np.uint8), bitorder="little").reshape(12, 8, 8)
    planes[:12] = bits[:, ::-1, :]
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


def _repetition_possible(board: chess.Board) -> bool:
    """Cheap necessary condition for ``board.can_claim_threefold_repetition()``.

    A threefold claim (now, or after one legal move) needs some position since the
    last capture or pawn move to have occurred at least twice already, and equal
    positions have equal occupancy. If no occupancy repeats in that window, the
    claim is impossible and python-chess's slow check (replay the history, then
    try every legal move) can be skipped. That check took about a third of
    self-play time.
    """
    window = min(board.halfmove_clock, len(board.move_stack))
    seen = {board.occupied}
    for state in board._stack[len(board._stack) - window:]:
        if state.occupied in seen:
            return True
        seen.add(state.occupied)
    return False


def terminal_value(board: chess.Board) -> float | None:
    """Value of a finished game for the side to move, or ``None`` if it isn't over.

    Same result as ``board.outcome(claim_draw=True)``: threefold repetition and
    the fifty-move rule end the game straight away rather than at fivefold /
    seventy-five moves. The repetition check is skipped when it can't succeed.
    """
    outcome = board.outcome()
    if outcome is None:
        if board.can_claim_fifty_moves() or (
                _repetition_possible(board) and board.can_claim_threefold_repetition()):
            return 0.0
        return None
    if outcome.winner is None:
        return 0.0
    # The side to move can only have lost (it has been checkmated).
    return 1.0 if outcome.winner == board.turn else -1.0
