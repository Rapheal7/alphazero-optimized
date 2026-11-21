"""End-to-end smoke test: a tiny self-play + training run, checkpoint, resume, evaluate."""

from alphazero.checkpoint import latest_checkpoint, load_model
from alphazero.cli import main
from alphazero.config import get_preset
from alphazero.training import train


def test_tiny_training_run_checkpoints_and_resumes(tmp_path):
    cfg = get_preset("tiny").updated(checkpoint_dir=str(tmp_path), seed=0)
    train(cfg, log=lambda _: None)
    first = latest_checkpoint(tmp_path)
    assert first is not None and first.name == "checkpoint_iter_0001.pt"

    model, saved_cfg, ckpt = load_model(first)
    assert ckpt["iteration"] == 1
    assert saved_cfg.num_channels == cfg.num_channels
    assert all(k in ckpt["metrics"] for k in ("loss", "policy_loss", "value_loss"))

    train(cfg.updated(iterations=2), resume=True, log=lambda _: None)
    assert latest_checkpoint(tmp_path).name == "checkpoint_iter_0002.pt"


def test_cli_train_then_evaluate(tmp_path, capsys):
    ck = str(tmp_path)
    assert main(["train", "--preset", "tiny", "--checkpoint-dir", ck, "--seed", "1"]) == 0
    assert main(["evaluate", "--checkpoint-dir", ck, "--games", "2",
                 "--simulations", "4", "--max-moves", "10"]) == 0
    assert "Result vs random" in capsys.readouterr().out


def test_cli_reports_unavailable_device(tmp_path, capsys):
    import torch

    from alphazero.config import device_problem

    if device_problem(torch.device("cuda")) is None:
        import pytest
        pytest.skip("CUDA is available here")
    assert main(["train", "--preset", "tiny", "--checkpoint-dir", str(tmp_path), "--device", "cuda"]) == 2
    assert "can't use device 'cuda'" in capsys.readouterr().err


def test_loss_metrics_split_policy_loss_into_entropy_and_kl():
    import torch

    from alphazero.network import AlphaZeroNet
    from alphazero.training import compute_loss

    cfg = get_preset("tiny")
    torch.manual_seed(0)
    model = AlphaZeroNet.from_config(cfg)
    states = torch.randn(4, 20, 8, 8)
    pis = torch.softmax(torch.randn(4, model.arch["action_size"]), dim=1)
    pis[0] = 0.0
    pis[0, 7] = 1.0  # a one-hot target has zero entropy
    zs = torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
    _, m = compute_loss(model, states, pis, zs)
    assert abs(m["policy_loss"] - (m["target_entropy"] + m["policy_kl"])) < 1e-5
    assert m["policy_kl"] >= 0 and m["target_entropy"] > 0 and m["policy_entropy"] > 0
    assert abs(m["value_baseline"] - 0.5) < 1e-6


def test_steps_follow_sample_reuse():
    from alphazero.training import steps_for

    cfg = get_preset("small").updated(batch_size=64, train_steps_per_iteration=50, sample_reuse=4)
    assert steps_for(cfg, 0) == 0
    assert steps_for(cfg, 10) == 1
    assert steps_for(cfg, 320) == 20  # 4 * 320 / 64
    assert steps_for(cfg, 100_000) == 50  # capped
    assert steps_for(cfg.updated(sample_reuse=0), 10) == 50


def test_iteration_summary_counts_results():
    import chess

    from alphazero.selfplay import GameRecord, summarize_games

    games = [
        GameRecord(white_value=1.0, termination="checkmate", examples=[(None, None, 1.0)] * 2),
        GameRecord(white_value=-1.0, termination="resignation", resigned=chess.WHITE),
        GameRecord(termination="threefold_repetition", examples=[(None, None, 0.0)] * 2),
        GameRecord(adjudicated=True, white_value=0.3, termination="move_limit",
                   resign_check=True, would_resign=chess.BLACK),
    ]
    text, m = summarize_games(games)
    assert "white +1 =1 -1, 1 at move limit" in text
    assert m["draw_rate"] == 0.25 and m["move_limit_rate"] == 0.25 and m["resign_rate"] == 0.25
    assert m["draw_target_fraction"] == 0.5 and m["false_resignations"] == 1.0


def test_cli_evaluate_against_another_checkpoint(tmp_path, capsys):
    ck = str(tmp_path)
    assert main(["train", "--preset", "tiny", "--checkpoint-dir", ck, "--seed", "1"]) == 0
    first = str(latest_checkpoint(tmp_path))
    assert main(["evaluate", "--checkpoint", first, "--opponent", "checkpoint",
                 "--opponent-checkpoint", first, "--games", "2",
                 "--simulations", "4", "--max-moves", "10"]) == 0
    out = capsys.readouterr().out
    assert "Opponent (B)" in out and "Result vs checkpoint" in out
