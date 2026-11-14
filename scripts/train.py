"""Run `alphazero train`. See `python scripts/train.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

from alphazero.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
