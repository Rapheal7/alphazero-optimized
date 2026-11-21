# Self-play speed on the CPU

Self-play is almost all of a CPU training run; the gradient steps take well
under a second per iteration with the presets here. This page covers where
self-play time went, what changed, how it compares with public AlphaZero-style
code, and whether a C++ port of the search would pay off.

Measured on an Apple M4 (4 performance + 6 efficiency cores), PyTorch 2.15 CPU,
python-chess 1.11. Another long engine match was running on the same machine the
whole time (load average 17 to 40 on 10 cores), so wall-clock numbers are noisy.
Single-process comparisons use CPU time, run before and after in alternation,
best of 3.

## Where the time went (before)

`cProfile` of one self-play game, untrained network:

| network, sims/move | search | network | python-chess | tree code |
|---|---|---|---|---|
| 2x32, 50 (`small`) | sequential `MCTS` | 38% | 48% | 6% |
| 2x32, 50 | `BatchedMCTS` | 17% | 61% | 9% |
| 10x128, 50 | sequential `MCTS` | 75% | 17% | 3% |
| 10x128, 50 | `BatchedMCTS` | 58% | 30% | 5% |

One python-chess call stood out: `can_claim_threefold_repetition` alone was
about 35% of the time on the small network. `terminal_value` called
`board.outcome(claim_draw=True)` at every leaf. That replays the game back to
the last irreversible move and then plays every legal move to see whether it
would repeat a position a third time.

## What changed

| change | where | effect on results |
|---|---|---|
| Training self-play uses `BatchedMCTS` (8 leaves per network call, virtual loss) | `training.py` | same search; leaf order differs within a batch |
| Self-play runs in a pool of worker processes, one PyTorch thread each, kept for the whole run | `parallel.py`, `--workers` | none: each game has its own seed, so games don't depend on the pool size (tested) |
| Draw check skips the repetition replay when no position since the last capture or pawn move has repeated (equal positions have equal occupancy) | `game.terminal_value` | identical to `outcome(claim_draw=True)` (tested on random games) |
| Move -> index by square arithmetic instead of `move.uci()` and a dict | `game.MoveEncoder.encode` | identical (tested for all 4544 actions) |
| Board planes from the 12 piece bitboards instead of `piece_map()` | `game.encode_board` | identical (tested) |
| No `board.copy()` per leaf: encoding and legal moves are read before unwinding | `batched_mcts.py` | identical |
| Batch norm folded into the convolutions for self-play | `network.inference_copy` | equal within 1e-6 |
| Self-play builds a fresh tree each move (`reuse_tree`, off by default) | `config.py` | matches the AlphaZero pseudocode and lc0's self-play; see below |

`BatchedMCTS.play_game` used to keep the tree between moves. In self-play that
doesn't save time, because every move still runs `num_simulations` new
simulations. It does change the training targets: the carried-over visits were
spent before that node got root Dirichlet noise, and the visit distribution
then mixes two searches. The AlphaZero pseudocode builds a new root every move,
and lc0 turns tree reuse off in self-play, so `reuse_tree` defaults to off. The
UCI engine still reuses its tree.

## Before and after

Plies of self-play per CPU-second, single process, one PyTorch thread
(8 games of at most 60 plies each, best of 3):

| network, sims/move | old training path (sequential `MCTS`) | old `BatchedMCTS` | new sequential | new batched |
|---|---|---|---|---|
| 2x32, 50 | 17.4 | 22.9 | 21.1 | **33.4** (1.9x) |
| 6x64, 100 | 7.3 | 8.4 | 8.3 | **11.6** (1.6x) |

Training now uses the batched search and spreads games over cores. Wall clock,
20 games, 2x32 network, run back to back on the loaded machine:

| run | old training path (sequential, 4 torch threads) | new (10 workers) |
|---|---|---|
| 1 | 143 s (7.8 plies/s) | 17.7 s (62.7 plies/s) |
| 2 | 241 s (4.6 plies/s) | 14.3 s (77.8 plies/s) |

That's 8x to 17x on a busy machine; the old path suffers more there because its
four PyTorch threads fight the other load for cores. The cleanest old number
(37 plies/s, measured before the other match started) is still under half of
the new path's 63 to 78 plies/s measured with the machine fully loaded. With
the machine idle, expect roughly the per-core gain times the number of
performance cores, plus about half a core for each efficiency core.

Reproduce with `python scripts/bench_selfplay.py` (see `--help`).

## How this compares with public AlphaZero-style code

| | AlphaZero pseudocode (DeepMind, 2018) | OpenSpiel AlphaZero | mctx (DeepMind) | Leela Chess Zero | this repo now |
|---|---|---|---|---|---|
| Parallel self-play | 5000 actors, always the latest network | Python: one process per actor, checkpoints reloaded from disk; C++: threads | a batch of games searched together on the accelerator | `--parallelism` games at once | worker processes; weights reloaded from a file each iteration |
| Batched network calls | across actors (TPUs) | C++ evaluator batches across threads (off by default); Python: none | every simulation step, across the whole batch | minibatches of leaves with virtual loss; mux backend batches across games | 8 leaves per call with virtual loss, within a game |
| Network cache | no | LRU cache of outputs | no | NN cache keyed by position hash | no (few transpositions in 50 to 200 simulation trees) |
| Tree reuse in self-play | no | no | n/a | no (engine play: yes) | no (UCI play: yes) |
| Exploration constant | grows with log of visits (c_base 19652, c_init 1.25) | fixed (2 in C++) | per policy | grows with log of visits, FPU reduction | fixed `c_puct` 1.5 |
| Batch norm at inference | n/a | no | n/a | folded into convs on load | folded into convs |
| Search language | n/a | C++ or Python | JAX (game must be JAX too) | C++ | Python |

Adopted: parallel actors fed from a weights file (OpenSpiel, Python), folded
batch norm (lc0), a fresh self-play tree per move (pseudocode, lc0), and
batched leaves with virtual loss (lc0; already in `BatchedMCTS`).

Not adopted yet, roughly in order of training progress per CPU-hour:

- **Fewer simulations with Gumbel search** (mctx `gumbel_muzero_policy`): Gumbel
  top-k with sequential halving at the root improves the policy even with 2 to
  16 simulations, against AlphaZero's 800. This is the biggest lever for "train
  longer in the same time". It changes the search and its policy target, so it
  needs its own PR and a strength test.
- **Playout cap randomization** (KataGo): a full search on a fraction of moves,
  recorded as policy targets; cheap searches elsewhere.
- **Resignation with some games played out** (pseudocode, lc0, OpenSpiel C++
  cutoff): cuts long lost games. Untrained networks rarely reach that, so it
  matters later in a run.
- **Log-growing c_puct and FPU reduction**: about strength, not speed.
- **Batching across games in one process** (mctx, lc0 mux): on this CPU a
  forward pass costs about the same per position from batch 8 up, so worker
  processes win. On a GPU, batching across games is the right design.

## Would a C++ port of the search pay off?

After the changes, one process with one PyTorch thread spends its time like this
(cProfile, which inflates the Python share a little):

| network, sims/move | network | python-chess | tree code + encoding | everything outside the network |
|---|---|---|---|---|
| 2x32, 50 | 41% | 26% | 26% | 59% |
| 6x64, 100 | 69% | 14% | 14% | 31% |
| 10x128, 200 | 86% | 6% | 6% | 14% |

A C++ tree search and move generator (bitboards, as in SlyChess, plus an
`(N, 18, 8, 8)` encoder, called from Python with pybind11) would make that last
column close to zero. By Amdahl's law the best case is:

| network | upper bound on speedup |
|---|---|
| 2x32 | about 2.3x |
| 6x64 | about 1.4x |
| 10x128 | about 1.15x |

The tiny default network is the only place it would really help, and you
wouldn't train that for strength. For networks big enough to learn chess, the
network forward pass is already the cost, and PyTorch already runs it in C++.
Not recommended now. Fewer simulations (Gumbel) or running the forward passes
on the M4 GPU (MPS) with batching across games would do more.
