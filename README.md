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
  batched_mcts.py  Faster search: batched leaves with virtual loss, tree reuse, fp16 on GPU
  selfplay.py    Self-play games -> (state, visit distribution, result) examples
  training.py    Replay buffer, loss, training loop, LR schedule, resume
  checkpoint.py  Save / load / find the latest checkpoint
  evaluation.py  Players (model, random, Stockfish), matches, scoring
  uci.py         UCI engine front end for chess GUIs (En Croissant, Cute Chess, Arena)
  cli.py         `alphazero train | play | evaluate | uci`
scripts/         train.py, play.py, evaluate.py, uci.py: thin wrappers around the CLI
                 alphazero-uci: launcher to point a chess GUI at
tests/           pytest: encoding, MCTS, UCI protocol, end-to-end tiny training run
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

### Installing to a folder you choose

By default pip installs into the active environment. To put the package somewhere else:

```bash
# Install into a folder of your choice (package plus its bin/ commands)
pip install --target path/to/dir .
PYTHONPATH=path/to/dir path/to/dir/bin/alphazero --help

# Or make the virtual environment wherever you want it
python -m venv path/to/env && path/to/env/bin/pip install .

# Or just build the wheel into a folder, without installing it
pip wheel --no-deps . -w path/to/dist
```

`--target` installs the dependencies (PyTorch included) into that folder too; add `--no-deps` if they're already installed.

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

## Use it in a chess GUI (UCI)

The engine speaks UCI, so any UCI GUI can load it. In [En Croissant](https://encroissant.org): **Engines → Add new → Local**, then pick the launcher:

- macOS / Linux: `scripts/alphazero-uci` in this repo. It uses the repo's `.venv` if there is one.
- Windows: run `pip install -e .`, then pick `.venv\Scripts\alphazero-uci.exe`.

It loads the latest checkpoint in `checkpoints/` (or `./checkpoints` if the GUI starts it somewhere that has one). To pick a checkpoint, set the engine's **Checkpoint** option to a `.pt` path, or pass arguments to the launcher: `scripts/alphazero-uci --checkpoint checkpoints/checkpoint_iter_0050.pt --device cpu`. Without a checkpoint it plays with an untrained network.

Checkpoints from the archived AlphaZeroChess2 notebook (`model_<iter>_<blocks>_<channels>_parallel_chess.pt`) load too. `alphazero/legacy.py` rebuilds that notebook's 14-plane network and fixes two of its training quirks (see the module docstring). A relative **Checkpoint** value is looked up in the working directory, then `checkpoints/`, then the repo root. If a checkpoint can't be loaded, the engine says why in an `info string` and plays with an untrained network.

| UCI option | Meaning |
|---|---|
| Checkpoint | `.pt` file to load (empty: latest in `checkpoints/`) |
| Device | auto, cpu, cuda or mps |
| Simulations | Simulations for `go depth N` or a bare `go` (0: the checkpoint's `num_simulations`) |
| Move Overhead | Milliseconds kept back from each timed move for GUI/IPC lag |

MCTS has no search depth, so `go depth N` runs the **Simulations** budget. `go nodes N` runs exactly N simulations, `go movetime`, `wtime`/`btime`/`winc`/`binc`/`movestogo` search for a share of the clock, and `go infinite` (the GUI's analysis mode) runs until `stop`. Each `info` line reports `nodes` as simulations, `depth` as the average leaf depth, and `score cp` converted from the MCTS value.

You can also talk to it by hand:

```bash
python scripts/uci.py
uci
position startpos moves e2e4
go movetime 1000
quit
```

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

### Faster search (`batched_mcts.py`)

`BatchedMCTS` takes the same arguments as `MCTS` and has the same `search()` method, but it's faster:

- **Batched leaves with virtual loss:** each step walks up to `batch_size` paths before calling the network. Each node on a pending path gets a temporary visit and loss, so the next path goes somewhere else. Then all the new leaves are evaluated in one forward pass.
- **Tree reuse:** after a move, the subtree under it becomes the next root, so its visits carry over. That means the visit counts can add up to more than `num_simulations`.
- **Inference mode, and fp16 autocast on CUDA**, for the network calls.

Its `play_game` / `generate_games` match `selfplay.py`, so to use it in training, change the import in `training.py`:

```python
from .batched_mcts import generate_games
```

On a laptop CPU with the `colab`-size network, `batch_size=8` runs about 2x as many simulations per second as `MCTS` (roughly 1,100 vs 530). Larger batches can be slower on CPU, because PyTorch's CPU convolutions slow down sharply at batch 16 on some machines. On a GPU, try 16 to 64.
