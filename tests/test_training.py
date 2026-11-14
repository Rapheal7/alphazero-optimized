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
