"""Benchmark self-play speed: plies per second for each way of playing games.

    python scripts/bench_selfplay.py                       # small preset network
    python scripts/bench_selfplay.py --res-blocks 6 --channels 64 --simulations 100
    python scripts/bench_selfplay.py --modes parallel --workers 4 8 10

Modes: ``sequential`` (alphazero.selfplay, one leaf per network call),
``batched`` (BatchedMCTS in this process) and ``parallel`` (BatchedMCTS in a
SelfPlayPool). Every mode plays the same number of games with an untrained
network, each capped at ``--max-moves`` plies. For ``parallel`` the pool start-up
(spawning workers, importing torch) is timed separately: training keeps one pool
for the whole run, so the warm time is what each iteration costs.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alphazero import batched_mcts, selfplay  # noqa: E402
from alphazero.config import get_preset  # noqa: E402
from alphazero.network import AlphaZeroNet  # noqa: E402


def _report(label: str, games, seconds: float, cpu: float | None = None) -> None:
    plies = sum(g.num_plies for g in games)
    line = f"{label:22s} {seconds:7.1f}s  {plies:5d} plies  {plies / seconds:7.1f} plies/s"
    if cpu is not None:  # CPU time is less disturbed by other programs than wall time
        line += f"  | cpu {cpu:6.1f}s  {plies / cpu:6.1f} plies/cpu-s"
    print(line, flush=True)


def _timed(fn):
    t, c = time.perf_counter(), time.process_time()
    result = fn()
    return result, time.perf_counter() - t, time.process_time() - c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", default="small")
    p.add_argument("--res-blocks", type=int)
    p.add_argument("--channels", type=int)
    p.add_argument("--simulations", type=int)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--max-moves", type=int, default=60)
    p.add_argument("--mcts-batch", type=int)
    p.add_argument("--modes", nargs="+", default=["sequential", "batched", "parallel"])
    p.add_argument("--workers", type=int, nargs="+", default=[4, 8, 10])
    p.add_argument("--threads", type=int, help="torch threads in this process")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = get_preset(args.preset).updated(
        num_res_blocks=args.res_blocks, num_channels=args.channels,
        num_simulations=args.simulations, max_moves=args.max_moves,
        mcts_batch_size=args.mcts_batch)
    torch.manual_seed(0)
    model = AlphaZeroNet.from_config(cfg).eval()
    print(f"{cfg.num_res_blocks}x{cfg.num_channels} net, {cfg.num_simulations} sims/move, "
          f"{args.games} games of <= {cfg.max_moves} plies, torch threads {torch.get_num_threads()}")
    quiet = lambda _: None  # noqa: E731

    if "sequential" in args.modes:
        np.random.seed(1)
        games, wall, cpu = _timed(lambda: selfplay.generate_games(model, cfg, args.games, log=quiet))
        _report("sequential", games, wall, cpu)
    if "batched" in args.modes:
        np.random.seed(1)
        games, wall, cpu = _timed(lambda: batched_mcts.generate_games(model, cfg, args.games, log=quiet))
        _report(f"batched ({cfg.mcts_batch_size} leaves)", games, wall, cpu)
    if "parallel" in args.modes:
        from alphazero.parallel import SelfPlayPool

        for w in args.workers:
            t = time.perf_counter()
            with SelfPlayPool(w) as pool:
                pool.generate_games(model, cfg.updated(max_moves=1), w, log=quiet)  # start workers
                start = time.perf_counter() - t
                np.random.seed(1)
                t = time.perf_counter()
                games = pool.generate_games(model, cfg, args.games, log=quiet)
                _report(f"parallel x{w}", games, time.perf_counter() - t)
            print(f"{'':22s} (first iteration also paid ~{start:.1f}s pool start-up)")


if __name__ == "__main__":
    main()
