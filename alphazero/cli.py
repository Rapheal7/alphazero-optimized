"""Command-line interface: ``alphazero train | play | evaluate``."""

from __future__ import annotations

import argparse
import sys

from typing import TYPE_CHECKING

from .config import PRESETS, Config, get_preset, resolve_device

if TYPE_CHECKING:
    from .network import AlphaZeroNet

# chess, torch and the model code are imported inside the commands, so that
# `alphazero uci` answers the GUI's first "uci" without waiting for torch to load.


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", help="Checkpoint file (default: latest in --checkpoint-dir)")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--simulations", type=int, help="MCTS simulations per move")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")


def _load(args: argparse.Namespace) -> tuple[AlphaZeroNet, Config]:
    from .checkpoint import latest_checkpoint, load_model
    from .network import AlphaZeroNet

    device = resolve_device(args.device)
    path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    if path is None:
        print(f"No checkpoint found in {args.checkpoint_dir}/; using an untrained network.")
        cfg = Config()
        return AlphaZeroNet.from_config(cfg).to(device).eval(), cfg
    model, cfg, ckpt = load_model(path, device)
    print(f"Loaded {path} (iteration {ckpt['iteration']}).")
    return model, cfg


def cmd_train(args: argparse.Namespace) -> int:
    from .training import train

    cfg = get_preset(args.preset).updated(
        iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        num_res_blocks=args.res_blocks,
        num_channels=args.channels,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )
    train(cfg, resume=args.resume)
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    import chess

    from .evaluation import MCTSPlayer, parse_human_move

    model, cfg = _load(args)
    ai = MCTSPlayer(model, cfg, args.simulations)
    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()
    print("Enter moves in SAN (Nf3) or UCI (g1f3). Type 'quit' to stop.\n")
    while not board.is_game_over(claim_draw=True):
        print(board.unicode(borders=True, orientation=human_color), "\n")
        if board.turn == human_color:
            try:
                text = input("Your move: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text.lower() in {"quit", "exit", "q"}:
                return 0
            move = parse_human_move(board, text)
            if move is None:
                print("Illegal or unrecognised move. Legal:",
                      " ".join(board.san(m) for m in board.legal_moves))
                continue
        else:
            move = ai.choose_move(board)
            print(f"AlphaZero plays {board.san(move)}")
        board.push(move)
    print(board.unicode(borders=True, orientation=human_color))
    outcome = board.outcome(claim_draw=True)
    print(f"\nGame over: {outcome.result()} ({outcome.termination.name.lower()})")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import MCTSPlayer, RandomPlayer, StockfishPlayer, evaluate

    model, cfg = _load(args)
    player = MCTSPlayer(model, cfg, args.simulations)
    if args.opponent == "random":
        stats = evaluate(player, RandomPlayer(), args.games, args.max_moves)
    else:
        try:
            opponent = StockfishPlayer(args.stockfish_path, args.skill_level, args.stockfish_time)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
        with opponent:
            stats = evaluate(player, opponent, args.games, args.max_moves)
    print(f"Result vs {args.opponent}: {stats}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphazero", description="AlphaZero-style chess engine.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Self-play + training loop")
    t.add_argument("--preset", default="small", choices=sorted(PRESETS),
                   help="tiny = seconds, small = minutes on CPU (default), colab = GPU run")
    t.add_argument("--iterations", type=int)
    t.add_argument("--games", type=int, help="Self-play games per iteration")
    t.add_argument("--simulations", type=int, help="MCTS simulations per move")
    t.add_argument("--res-blocks", type=int)
    t.add_argument("--channels", type=int)
    t.add_argument("--train-steps", type=int, help="Gradient steps per iteration")
    t.add_argument("--batch-size", type=int)
    t.add_argument("--max-moves", type=int, help="Plies before a self-play game is scored as a draw")
    t.add_argument("--lr", type=float)
    t.add_argument("--checkpoint-dir", default="checkpoints")
    t.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    t.add_argument("--seed", type=int)
    t.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("play", help="Play against a trained model in the terminal")
    _add_model_args(p)
    p.add_argument("--color", choices=["white", "black"], default="white")
    p.set_defaults(func=cmd_play)

    e = sub.add_parser("evaluate", help="Match a model against a random mover or Stockfish")
    _add_model_args(e)
    e.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    e.add_argument("--games", type=int, default=4)
    e.add_argument("--max-moves", type=int, default=200)
    e.add_argument("--stockfish-path")
    e.add_argument("--skill-level", type=int, default=0, help="Stockfish skill level 0-20")
    e.add_argument("--stockfish-time", type=float, default=0.05, help="Seconds per Stockfish move")
    e.set_defaults(func=cmd_evaluate)

    u = sub.add_parser("uci", help="Run as a UCI engine for chess GUIs (En Croissant, Cute Chess...)")
    u.add_argument("--checkpoint", help="Checkpoint file (default: latest in --checkpoint-dir)")
    u.add_argument("--checkpoint-dir",
                   help="Default: ./checkpoints if it exists, else checkpoints/ in the repo")
    u.add_argument("--simulations", type=int,
                   help="MCTS simulations for 'go' without a time limit (default: from checkpoint)")
    u.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    u.set_defaults(func=cmd_uci)
    return parser


def cmd_uci(args: argparse.Namespace) -> int:
    from .uci import run_engine

    return run_engine(args)


def main(argv: list[str] | None = None) -> int:
    from .config import DeviceUnavailableError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DeviceUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
