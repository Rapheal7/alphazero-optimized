"""AlphaZero-style chess: self-play reinforcement learning with MCTS and a ResNet."""

from .config import Config, get_preset

__all__ = ["AlphaZeroNet", "Config", "get_preset"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # Loaded on first use so `alphazero uci` starts without importing torch.
    if name == "AlphaZeroNet":
        from .network import AlphaZeroNet

        return AlphaZeroNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
