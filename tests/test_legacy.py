"""Notebook checkpoints: the notebook's own network (copied verbatim below) must load
into NotebookNet and give the same policy ranking and the sign-corrected value."""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.checkpoint import load_model
from alphazero.game import ACTION_SIZE, encode_board
from alphazero.legacy import NotebookNet


# --- from notebooks/archive/AlphaZeroChess2.ipynb ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ResNetAlphaZero(nn.Module):
    def __init__(self, action_size, num_residual_blocks, num_hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(14, num_hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_hidden_channels) for _ in range(num_residual_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden_channels, 2, kernel_size=1), nn.BatchNorm2d(2),
            nn.Flatten(), nn.Linear(2 * 8 * 8, action_size))
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden_channels, 1, kernel_size=1), nn.BatchNorm2d(1),
            nn.Flatten(), nn.Linear(1 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        return F.softmax(self.policy_head(x), dim=1), self.value_head(x)


def notebook_encode(board):
    """The notebook's _encode_state_numba, in plain Python."""
    a = np.zeros((14, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        a["PNBRQKpnbrqk".index(piece.symbol()), 7 - square // 8, square % 8] = 1.0
    a[12] = 1.0 if board.turn == chess.WHITE else 0.0
    a[13] = 1.0 if any(board.has_kingside_castling_rights(c) or board.has_queenside_castling_rights(c)
                       for c in chess.COLORS) else 0.0
    return a
# -----------------------------------------------------------------------------


def _save_notebook_checkpoint(path, blocks=2, channels=8):
    torch.manual_seed(0)
    net = ResNetAlphaZero(ACTION_SIZE, blocks, channels)
    # Non-trivial BatchNorm statistics, so a layer mix-up would show.
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    net.eval()
    torch.save({"episode": 5, "model_state": net.state_dict(), "optim_state": {}}, path)
    return net


def test_notebook_checkpoint_loads_and_matches_the_notebook_network(tmp_path):
    path = tmp_path / "model_5_2_8_parallel_chess.pt"
    reference = _save_notebook_checkpoint(path)
    model, cfg, ckpt = load_model(path)
    assert isinstance(model, NotebookNet)
    assert (cfg.num_res_blocks, cfg.num_channels, ckpt["iteration"]) == (2, 8, 5)

    for fen in [chess.STARTING_FEN, "r3k2r/8/8/8/8/8/8/4K3 b kq - 3 20",
                "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"]:
        board = chess.Board(fen)
        logits, value = model.predict(encode_board(board))
        with torch.no_grad():
            ref_probs, ref_value = reference(torch.from_numpy(notebook_encode(board)).unsqueeze(0))
        ours = torch.softmax(torch.from_numpy(logits), dim=0)
        assert torch.allclose(ours, ref_probs[0], atol=1e-6)
        assert abs(value + ref_value.item()) < 1e-5  # sign corrected


def test_notebook_model_plays_through_uci(tmp_path, monkeypatch):
    import io
    import time

    from alphazero.uci import UCIEngine

    _save_notebook_checkpoint(tmp_path / "model_5_2_8_parallel_chess.pt")
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    engine = UCIEngine(checkpoint_dir=tmp_path / "none", device="cpu", out=out)
    engine.handle("setoption name Checkpoint value model_5_2_8_parallel_chess.pt")
    engine.handle("go nodes 8")
    end = time.monotonic() + 30
    while not any(l.startswith("bestmove") for l in out.getvalue().splitlines()):
        assert time.monotonic() < end
        time.sleep(0.01)
    assert "loaded" in out.getvalue() and "iteration 5" in out.getvalue()
