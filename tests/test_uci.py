import io
import time

import chess
import numpy as np
import torch

from alphazero.uci import UCIEngine, parse_go, q_to_centipawns


def _engine(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    out = io.StringIO()
    # An empty checkpoint dir means an untrained network: enough to test the protocol.
    return UCIEngine(checkpoint_dir=tmp_path, device="cpu", out=out), out


def _lines(out):
    return out.getvalue().splitlines()


def _wait_for_bestmove(out, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        moves = [line for line in _lines(out) if line.startswith("bestmove")]
        if moves:
            return moves[-1]
        time.sleep(0.01)
    raise AssertionError("no bestmove")


def test_handshake(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("uci")
    lines = _lines(out)
    assert lines[0].startswith("id name ")
    assert any(line.startswith("option name Simulations") for line in lines)
    assert lines[-1] == "uciok"
    engine.handle("isready")
    assert _lines(out)[-1] == "readyok"


def test_go_nodes_returns_legal_move(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("position startpos moves e2e4 e7e5")
    engine.handle("go nodes 30")
    best = _wait_for_bestmove(out).split()[1]
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(best) in board.legal_moves
    info = [line for line in _lines(out) if line.startswith("info depth")][-1]
    assert " nodes 30 " in info


def test_finds_and_reports_mate_in_one(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    engine.handle("go nodes 300")
    assert _wait_for_bestmove(out).split()[1] == "a1a8"
    assert any("score mate 1" in line for line in _lines(out))


def test_infinite_waits_for_stop(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("isready")
    engine.handle("position startpos")
    engine.handle("go infinite")
    time.sleep(0.3)
    assert not any(line.startswith("bestmove") for line in _lines(out))
    engine.handle("stop")
    assert _lines(out)[-1].startswith("bestmove")


def test_finished_game_returns_null_move(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    engine.handle("go movetime 100")
    assert _lines(out)[-1] == "bestmove 0000"


def test_illegal_move_keeps_the_legal_prefix(tmp_path):
    engine, out = _engine(tmp_path)
    engine.handle("position startpos moves e2e4 e2e4")
    assert engine.board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/")
    assert "illegal move" in _lines(out)[-1]


def test_setoption(tmp_path):
    engine, _ = _engine(tmp_path)
    engine.handle("setoption name Move Overhead value 250")
    engine.handle("setoption name Simulations value 77")
    assert engine.move_overhead_ms == 250
    assert engine.default_simulations == 77
    engine.handle("setoption name Simulations value 0")
    assert engine.simulations is None


def test_parse_go():
    white = chess.WHITE
    assert parse_go(["nodes", "500"], white, 50, 100).simulations == 500
    assert parse_go(["depth", "12"], white, 50, 100).simulations == 50
    assert parse_go([], white, 50, 100).simulations == 50
    assert parse_go(["movetime", "1000"], white, 50, 100).seconds == 0.9
    clock = parse_go(["wtime", "60000", "btime", "1000", "winc", "1000"], white, 50, 100)
    assert clock.seconds == (60000 / 20 + 500 - 100) / 1000
    assert clock.simulations is None
    black = parse_go(["wtime", "60000", "btime", "1000", "movestogo", "1"], chess.BLACK, 50, 100)
    assert black.seconds == 0.5  # capped at half of black's remaining clock
    assert parse_go(["infinite"], white, 50, 100).infinite


def test_q_to_centipawns():
    assert q_to_centipawns(0.0) == 0
    assert q_to_centipawns(0.5) > 0 > q_to_centipawns(-0.5)
    assert q_to_centipawns(1.0) == -q_to_centipawns(-1.0)


def _play_one_move(engine, out):
    engine.handle("go nodes 4")
    return _wait_for_bestmove(out)


def test_missing_checkpoint_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine, out = _engine(tmp_path)
    engine.handle("setoption name Checkpoint value model_5_20_256_parallel_chess.pt")
    assert _play_one_move(engine, out).startswith("bestmove ")
    assert any("not found" in line and "untrained" in line for line in _lines(out))


def test_notebook_checkpoint_is_rejected_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The archived notebooks save {'episode', 'model_state', 'optim_state'}.
    torch.save({"episode": 5, "model_state": {"conv1.weight": torch.zeros(1)}, "optim_state": {}},
               tmp_path / "model_5_20_256_parallel_chess.pt")
    engine, out = _engine(tmp_path)
    engine.handle("setoption name Checkpoint value model_5_20_256_parallel_chess.pt")
    assert _play_one_move(engine, out).startswith("bestmove ")
    assert any("archived notebooks" in line for line in _lines(out))


def test_bare_checkpoint_name_is_found_in_checkpoint_dir(tmp_path, monkeypatch):
    from alphazero.checkpoint import save_checkpoint
    from alphazero.config import get_preset
    from alphazero.network import AlphaZeroNet

    cfg = get_preset("tiny")
    save_checkpoint(tmp_path / "ckpts" / "mine.pt", AlphaZeroNet.from_config(cfg), cfg, 3)
    monkeypatch.chdir(tmp_path)  # the GUI's working directory: no file here
    out = io.StringIO()
    engine = UCIEngine(checkpoint_dir=tmp_path / "ckpts", device="cpu", out=out)
    engine.handle("setoption name Checkpoint value mine.pt")
    assert _play_one_move(engine, out).startswith("bestmove ")
    assert any("loaded" in line and "iteration 3" in line for line in _lines(out))
