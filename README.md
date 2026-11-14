# alphazero-optimized

An AlphaZero-style chess engine in PyTorch. The network learns only from games against itself: Monte Carlo Tree Search (MCTS), guided by a residual policy/value network, plays each move, and the network trains on the search results and the final scores.

The defaults run small on a laptop CPU, so you can see the whole loop work in about a minute. The `colab` preset is for a real run on a GPU.

## Layout

```
alphazero/
  config.py      Config dataclass + presets (tiny / small / colab), device selection
  game.py        Board encoding (18x8x8), move <-> action index (4544 actions), terminal values
  network.py     AlphaZeroNet: ResNet trunk, policy head (logits) and value head (tanh)
  mcts.py        PUCT tree search with Dirichlet noise at the root
  selfplay.py    Self-play games -> (state, visit distribution, result) examples
  training.py    Replay buffer, loss, training loop, LR schedule, resume
  checkpoint.py  Save / load / find the latest checkpoint
  evaluation.py  Players (model, random, Stockfish), matches, scoring
  cli.py         `alphazero train | play | evaluate`
scripts/         train.py, play.py, evaluate.py: thin wrappers around the CLI
tests/           pytest: encoding, MCTS, end-to-end tiny training run
notebooks/
  AlphaZeroChess_Colab.ipynb   Clone + install + train / evaluate / play in Colab
  archive/                     The original exploratory notebooks, unchanged
```

## Setup

Python 3.10+.

```bash
git clone https://github.com/Rapheal7/alphazero-optimized.git
cd alphazero-optimized
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e ".[dev]"  (adds the `alphazero` command)
```

Stockfish is optional. You only need it to evaluate against Stockfish: `brew install stockfish` or `sudo apt-get install stockfish`.

## Run

```bash
# Train (default "small" preset: ~1 minute on CPU, checkpoints in ./checkpoints)
python scripts/train.py

# Seconds-long smoke run
python scripts/train.py --preset tiny

# Bigger run, or carry on from the latest checkpoint
python scripts/train.py --preset colab --iterations 50
python scripts/train.py --resume --iterations 10

# Override anything
python scripts/train.py --simulations 100 --games 8 --res-blocks 4 --channels 64 --device cuda

# Evaluate the latest checkpoint against a random mover or Stockfish
python scripts/evaluate.py --games 4
python scripts/evaluate.py --opponent stockfish --skill-level 0 --games 4

# Play against it in the terminal (SAN like Nf3 or UCI like g1f3)
python scripts/play.py --color white --simulations 200
```

After `pip install -e .`, the same commands are available as `alphazero train`, `alphazero play` and `alphazero evaluate`, and also as `python -m alphazero ...`. Add `--help` to any command for the full list of options.

### Presets

| preset  | network            | sims/move | games/iter | iterations | intended for          |
|---------|--------------------|-----------|------------|------------|-----------------------|
| `tiny`  | 1 block × 16 ch    | 8         | 1          | 1          | tests, smoke runs     |
| `small` | 2 blocks × 32 ch   | 50        | 4          | 5          | default, laptop CPU   |
| `colab` | 10 blocks × 128 ch | 200       | 25         | 100        | GPU (Colab)           |

Self-play games longer than `--max-moves` plies are stopped and scored as a draw. This keeps CPU runs short, because untrained networks rarely finish a game.

## Colab

Open `notebooks/AlphaZeroChess_Colab.ipynb` in Colab and set the runtime to GPU. It clones this repo, installs the package, saves checkpoints to Google Drive (optional) and runs `alphazero train --resume`, so you can re-run the cell after a disconnect. It can also evaluate against random or Stockfish and let you play against the model.

## Tests

```bash
pytest -q
```

The tests check that the move encoding covers every legal move with no collisions (including underpromotions, castling and en passant), and that MCTS finds a mate in one with an untrained network. They also run a tiny end-to-end loop: self-play, a training step, a checkpoint, resume, then evaluation through the CLI.

## How it works

- **Encoding:** 12 piece planes, side to move, 4 castling-rights planes and the halfmove clock, from White's point of view. The policy covers every from/to square pair, plus a separate index for each promotion piece.
- **Search:** PUCT selection `Q + c_puct · P · √N_parent / (1 + N)`. The network gives priors and a value at each new leaf, and finished games use their true result. Values flip sign at each ply on the way back up. Dirichlet noise is added at the root during self-play only.
- **Training targets:** the root visit distribution (policy) and the game result from the point of view of the side to move (value). Loss is `(z − v)² − π·log p`, using Adam with weight decay and a step LR schedule.
