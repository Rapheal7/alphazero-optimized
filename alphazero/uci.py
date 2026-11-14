"""UCI (Universal Chess Interface) front end, so chess GUIs such as En Croissant,
Cute Chess or Arena can load the trained network as an engine.

Run it with ``alphazero uci`` (or ``alphazero-uci`` / ``python scripts/uci.py``)
and point the GUI at that command. The GUI talks to the engine over stdin/stdout:

    uci                    -> id / option lines, then uciok
    isready                -> loads the checkpoint (first time only), then readyok
    position startpos moves e2e4 e7e5
    go movetime 1000       -> info lines while searching, then bestmove
    stop / quit

MCTS has no notion of search depth, so ``go depth N`` runs the number of
simulations set by the ``Simulations`` option. ``go nodes N`` runs exactly N
simulations, and the time controls (``movetime``, ``wtime``/``btime``...) and
``go infinite`` search until the time is up or the GUI sends ``stop``.

Torch is only imported when the model is first needed, so the engine answers
``uci`` straight away and the GUI does not time out while it starts.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import chess

if TYPE_CHECKING:
    from .config import Config
    from .mcts import MCTS, Node

ENGINE_NAME = "AlphaZero Optimized"
ENGINE_AUTHOR = "Rapheal7"

DEFAULT_MOVE_OVERHEAD_MS = 100
INFO_INTERVAL_S = 0.5
_DEVICES = ("auto", "cpu", "cuda", "mps")


def q_to_centipawns(q: float) -> int:
    """Map an MCTS value in [-1, 1] to centipawns (the curve Leela Chess Zero uses)."""
    q = max(-0.999, min(0.999, q))
    return round(111.714640912 * math.tan(1.5620688421 * q))


def default_checkpoint_dir() -> Path:
    """``./checkpoints`` if it exists, otherwise ``checkpoints/`` at the repo root.

    GUIs start engines from their own working directory, so a relative path alone
    would usually miss the checkpoints written by ``alphazero train``.
    """
    cwd = Path("checkpoints")
    if cwd.is_dir():
        return cwd
    return Path(__file__).resolve().parents[1] / "checkpoints"


@dataclass
class SearchLimits:
    """What to search for after ``go``: a simulation budget, a time budget, or both."""

    simulations: int | None = None
    seconds: float | None = None
    infinite: bool = False


def parse_go(tokens: list[str], turn: chess.Color, default_simulations: int,
             move_overhead_ms: int) -> SearchLimits:
    """Turn the arguments of a ``go`` command into :class:`SearchLimits`."""
    args: dict[str, int] = {}
    flags: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"infinite", "ponder"}:
            flags.add(token)
        elif token == "searchmoves":
            break  # not supported; the rest of the line is a move list
        elif i + 1 < len(tokens):
            try:
                args[token] = int(tokens[i + 1])
                i += 1
            except ValueError:
                pass
        i += 1

    limits = SearchLimits(infinite=bool(flags))
    if "nodes" in args:
        limits.simulations = max(1, args["nodes"])
    if "movetime" in args:
        limits.seconds = max(args["movetime"] - move_overhead_ms, 10) / 1000
    else:
        own_time = args.get("wtime" if turn == chess.WHITE else "btime")
        if own_time is not None:
            increment = args.get("winc" if turn == chess.WHITE else "binc", 0)
            moves_to_go = max(args.get("movestogo", 20), 1)
            budget = own_time / moves_to_go + increment / 2 - move_overhead_ms
            # Never plan to use more than half the clock on one move.
            budget = min(budget, own_time / 2)
            limits.seconds = max(budget, 10) / 1000
    if limits.simulations is None and limits.seconds is None and not limits.infinite:
        limits.simulations = default_simulations  # bare "go" or "go depth N"
    return limits


class UCIEngine:
    """Reads UCI commands, runs MCTS on a background thread, writes replies."""

    def __init__(self, checkpoint: str | None = None, checkpoint_dir: str | Path | None = None,
                 device: str = "auto", simulations: int | None = None,
                 out: TextIO | None = None) -> None:
        self.out = out or sys.stdout
        self.checkpoint = checkpoint or ""
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
        self.device = device
        self.simulations = simulations  # None: use the checkpoint's own setting
        self.move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS

        self.board = chess.Board()
        self._mcts: MCTS | None = None
        self._cfg: Config | None = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ponder_limits: SearchLimits | None = None
        self._deadline: float | None = None

    # ---------------------------------------------------------------- output

    def send(self, line: str) -> None:
        with self._write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def info_string(self, text: str) -> None:
        self.send(f"info string {text}")

    # ----------------------------------------------------------------- model

    def _ensure_model(self) -> None:
        if self._mcts is not None:
            return
        from .checkpoint import latest_checkpoint, load_model
        from .config import Config, resolve_device
        from .mcts import MCTS
        from .network import AlphaZeroNet

        device = resolve_device(self.device)
        model = cfg = None
        if self.checkpoint:
            path = self._resolve_checkpoint(self.checkpoint)
            if path is None:
                tried = ", ".join(str(p) for p in self._checkpoint_candidates(self.checkpoint))
                self.info_string(f"checkpoint {self.checkpoint} not found (looked in: {tried}); "
                                 "using an untrained network")
        else:
            path = latest_checkpoint(self.checkpoint_dir)
            if path is None:
                self.info_string(f"no checkpoint in {self.checkpoint_dir}; using an untrained network")
        if path is not None:
            # A bad checkpoint must not take the engine (and the GUI's game) down with it.
            try:
                model, cfg, ckpt = load_model(path, device)
                self.info_string(f"loaded {path} (iteration {ckpt['iteration']}) on {device}")
            except Exception as e:  # noqa: BLE001
                self.info_string(f"could not load checkpoint {path}: {e}; using an untrained network")
        if model is None:
            cfg = Config()
            model = AlphaZeroNet.from_config(cfg).to(device).eval()
        self._cfg = cfg
        self._mcts = MCTS(model, cfg)

    def _checkpoint_candidates(self, name: str) -> list[Path]:
        """Where a Checkpoint value may point. GUIs start engines in their own working
        directory, so a bare file name is also looked up in the checkpoint folder and
        at the repo root."""
        path = Path(name).expanduser()
        if path.is_absolute():
            return [path]
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [Path.cwd() / path, self.checkpoint_dir / path, repo_root / path]
        return list(dict.fromkeys(p.resolve() for p in candidates))

    def _resolve_checkpoint(self, name: str) -> Path | None:
        return next((p for p in self._checkpoint_candidates(name) if p.is_file()), None)

    def _reset_model(self) -> None:
        self._stop_search()
        self._mcts = None
        self._cfg = None

    @property
    def default_simulations(self) -> int:
        if self.simulations is not None:
            return self.simulations
        return self._cfg.num_simulations if self._cfg else 50

    # -------------------------------------------------------------- commands

    def handle(self, line: str) -> bool:
        """Handle one command line. Returns False when the engine should exit."""
        tokens = line.split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "uci":
            self.cmd_uci()
        elif cmd == "isready":
            self._ensure_model()
            self.send("readyok")
        elif cmd == "setoption":
            self.cmd_setoption(args)
        elif cmd == "ucinewgame":
            self._stop_search()
            self.board = chess.Board()
        elif cmd == "position":
            self._stop_search()
            self.cmd_position(args)
        elif cmd == "go":
            self.cmd_go(args)
        elif cmd == "stop":
            self._stop_search()
        elif cmd == "ponderhit":
            self.cmd_ponderhit()
        elif cmd == "d":
            self.send(str(self.board))
            self.send(f"Fen: {self.board.fen()}")
        elif cmd == "quit":
            self._stop_search()
            return False
        # Anything else (debug, register, unknown commands) is ignored, as UCI asks.
        return True

    def cmd_uci(self) -> None:
        self.send(f"id name {ENGINE_NAME}")
        self.send(f"id author {ENGINE_AUTHOR}")
        self.send("option name Checkpoint type string default <empty>")
        self.send(f"option name Device type combo default {self.device} "
                  + " ".join(f"var {d}" for d in _DEVICES))
        sims = self.simulations if self.simulations is not None else 0
        self.send(f"option name Simulations type spin default {sims} min 0 max 1000000")
        self.send(f"option name Move Overhead type spin default {self.move_overhead_ms} "
                  "min 0 max 5000")
        self.send("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        # setoption name <id> [value <x>]; both may contain spaces.
        if "name" not in args:
            return
        rest = args[args.index("name") + 1:]
        if "value" in rest:
            split = rest.index("value")
            name, value = " ".join(rest[:split]), " ".join(rest[split + 1:])
        else:
            name, value = " ".join(rest), ""
        key = name.lower()
        if key == "checkpoint":
            self.checkpoint = "" if value in {"", "<empty>"} else value
            self._reset_model()
        elif key == "device":
            if value not in _DEVICES:
                self.info_string(f"unknown device {value!r}")
                return
            self.device = value
            self._reset_model()
        elif key == "simulations":
            try:
                n = int(value)
            except ValueError:
                return
            self.simulations = n if n > 0 else None  # 0: use the checkpoint's setting
        elif key == "move overhead":
            try:
                self.move_overhead_ms = max(0, int(value))
            except ValueError:
                pass

    def cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        if "moves" in args:
            split = args.index("moves")
            spec, moves = args[:split], args[split + 1:]
        else:
            spec, moves = args, []
        try:
            if spec[0] == "startpos":
                board = chess.Board()
            elif spec[0] == "fen":
                board = chess.Board(" ".join(spec[1:]))
            else:
                return
        except ValueError as e:
            self.info_string(f"invalid FEN: {e}")
            return
        for uci in moves:
            try:
                board.push_uci(uci)
            except ValueError:
                self.info_string(f"illegal move {uci} in position command; ignoring the rest")
                break
        self.board = board

    def cmd_go(self, args: list[str]) -> None:
        self._stop_search()
        self._ensure_model()
        if self.board.is_game_over(claim_draw=True):
            self.info_string("position is already decided; no legal search")
            self.send("bestmove 0000")
            return
        limits = parse_go(args, self.board.turn, self.default_simulations, self.move_overhead_ms)
        self._ponder_limits = None
        if "ponder" in args:
            # Search freely until ponderhit, then switch to the real time budget.
            self._ponder_limits = limits
            limits = SearchLimits(infinite=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._search, args=(self.board.copy(), limits), daemon=True)
        self._thread.start()

    def cmd_ponderhit(self) -> None:
        limits = self._ponder_limits
        self._ponder_limits = None
        if limits is None or self._thread is None:
            return
        if limits.seconds is not None:
            self._deadline = time.monotonic() + limits.seconds
        elif not limits.infinite:
            self._stop.set()

    def _stop_search(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        self._ponder_limits = None

    # ---------------------------------------------------------------- search

    def _search(self, board: chess.Board, limits: SearchLimits) -> None:
        assert self._mcts is not None
        start = time.monotonic()
        self._deadline = start + limits.seconds if limits.seconds is not None else None
        root = self._mcts.new_root(board)
        scratch = board.copy()
        sims = depth_total = seldepth = 0
        next_info = start + INFO_INTERVAL_S

        while not self._stop.is_set():
            if limits.simulations is not None and sims >= limits.simulations:
                break
            if self._deadline is not None and time.monotonic() >= self._deadline:
                break
            leaf_depth = self._mcts.simulate(root, scratch)
            sims += 1
            depth_total += leaf_depth
            seldepth = max(seldepth, leaf_depth)
            if time.monotonic() >= next_info:
                self._send_info(board, root, sims, depth_total, seldepth, start)
                next_info += INFO_INTERVAL_S

        # UCI: with "infinite" (or while pondering) bestmove must wait for "stop".
        while (limits.infinite or self._ponder_limits is not None) and not self._stop.is_set():
            if self._deadline is not None and time.monotonic() >= self._deadline:
                break
            time.sleep(0.01)

        self._send_info(board, root, sims, depth_total, seldepth, start)
        pv = self._principal_variation(root)
        best = pv[0] if pv else next(iter(root.children.values())).move
        line = f"bestmove {best.uci()}"
        if len(pv) > 1:
            line += f" ponder {pv[1].uci()}"
        self.send(line)

    @staticmethod
    def _principal_variation(root: Node) -> list[chess.Move]:
        pv, node = [], root
        while node.children:
            child = max(node.children.values(), key=lambda c: c.visit_count)
            if child.visit_count == 0:
                break
            pv.append(child.move)
            node = child
        return pv

    def _send_info(self, board: chess.Board, root: Node, sims: int, depth_total: int,
                   seldepth: int, start: float) -> None:
        pv = self._principal_variation(root)
        if not pv:
            return
        best = max(root.children.values(), key=lambda c: c.visit_count)
        after = board.copy(stack=False)
        after.push(best.move)
        # best.q is from the point of view of the side to move at the root.
        score = "mate 1" if after.is_checkmate() else f"cp {q_to_centipawns(best.q)}"
        elapsed = max(time.monotonic() - start, 1e-6)
        depth = max(1, round(depth_total / sims)) if sims else 1
        self.send(
            f"info depth {depth} seldepth {max(seldepth, depth)} multipv 1 score {score} "
            f"nodes {sims} nps {int(sims / elapsed)} time {int(elapsed * 1000)} "
            f"pv {' '.join(m.uci() for m in pv)}"
        )

    # ------------------------------------------------------------------ loop

    def run(self, stdin: TextIO | None = None) -> None:
        stdin = stdin or sys.stdin
        for line in stdin:
            if not self.handle(line.strip()):
                break
        self._stop_search()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``alphazero-uci`` command."""
    from .cli import main as cli_main

    return cli_main(["uci", *(sys.argv[1:] if argv is None else argv)])


def run_engine(args: Any) -> int:
    UCIEngine(
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        simulations=args.simulations,
    ).run()
    return 0
