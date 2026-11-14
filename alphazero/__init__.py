"""AlphaZero-style chess: self-play reinforcement learning with MCTS and a ResNet."""

from .config import Config, get_preset
from .network import AlphaZeroNet

__all__ = ["AlphaZeroNet", "Config", "get_preset"]
__version__ = "0.1.0"
