"""Deterministic games and the Dave's Agent Tools mailbox client."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
import shutil
import threading
import time
from typing import Callable

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
    (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6),
)


@dataclass
class TicTacToe:
    """Authoritative human-X / Fluffy-O game engine."""
    board: list[str] = field(default_factory=lambda: [""] * 9)
    turn: str = "human"
    result: str | None = None

    def winner(self) -> str | None:
        for a, b, c in WIN_LINES:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return "draw" if all(self.board) else None

    def human_move(self, cell: int) -> None:
        if self.result or self.turn != "human":
            raise ValueError("It is not your turn.")
        if not 0 <= cell < 9 or self.board[cell]:
            raise ValueError("That square is not available.")
        self.board[cell] = "X"
        self.result = self.winner()
        if self.result is None:
            self.turn = "fluffy"

    def fluffy_move(self) -> int:
        if self.result or self.turn != "fluffy":
            raise ValueError("It is not Fluffy's turn.")
        choices = []
        for cell, value in enumerate(self.board):
            if not value:
                self.board[cell] = "O"
                choices.append((self._minimax(False, 0), cell))
                self.board[cell] = ""
        best = max(score for score, _ in choices)
        cell = random.choice([cell for score, cell in choices if score == best])
        self.board[cell] = "O"
        self.result = self.winner()
        if self.result is None:
            self.turn = "human"
        return cell

    def _minimax(self, maximizing: bool, depth: int) -> int:
        outcome = self.winner()
        if outcome == "O":
            return 10 - depth
        if outcome == "X":
            return depth - 10
        if outcome == "draw":
            return 0
        values = []
        mark = "O" if maximizing else "X"
        for cell, value in enumerate(self.board):
            if not value:
                self.board[cell] = mark
                values.append(self._minimax(not maximizing, depth + 1))
                self.board[cell] = ""
        return (max if maximizing else min)(values)

    def public_state(self) -> dict:
        return {"game": "tic_tac_toe", "board": list(self.board),
                "turn": self.turn, "result": self.result}

    def close(self) -> None:
        return None


CHESS_DIFFICULTIES = {
    "puppy": {"label": "Puppy", "skill": 0, "move_time": 0.05},
    "friendly": {"label": "Friendly", "skill": 4, "move_time": 0.10},
    "clever": {"label": "Clever", "skill": 9, "move_time": 0.18},
    "strong": {"label": "Strong", "skill": 16, "move_time": 0.35},
}


class ChessGame:
    """Authoritative human-White chess game backed by desktop Stockfish."""

    def __init__(self, difficulty: str = "friendly", stockfish_path: str | None = None):
        try:
            import chess
            import chess.engine
        except ImportError as exc:
            raise RuntimeError(
                "Chess needs python-chess; run: python -m pip install python-chess"
            ) from exc
        self.chess = chess
        self.board = chess.Board()
        self.turn = "human"
        self.result = None
        self.difficulty = (
            difficulty if difficulty in CHESS_DIFFICULTIES else "friendly"
        )
        configured_path = stockfish_path or os.environ.get("STOCKFISH_PATH", "")
        executable = configured_path or shutil.which("stockfish")
        if not executable:
            raise RuntimeError(
                "Stockfish was not found; set STOCKFISH_PATH to stockfish.exe"
            )
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(executable)
        except Exception as exc:
            raise RuntimeError(f"Stockfish could not start: {exc}") from exc
        settings = CHESS_DIFFICULTIES[self.difficulty]
        if "Skill Level" in self.engine.options:
            self.engine.configure({"Skill Level": settings["skill"]})
        self.last_move = None
        self.last_move_san = None
        self.last_event = None
        self.event_move_san = None
        self.evaluation_cp = self._evaluate()

    def _evaluate(self) -> int:
        analysis = self.engine.analyse(
            self.board,
            self.chess.engine.Limit(time=0.04),
        )
        score = analysis["score"].pov(self.chess.WHITE)
        return int(score.score(mate_score=100000) or 0)

    def _outcome(self) -> str | None:
        if not self.board.is_game_over(claim_draw=True):
            return None
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return "draw"
        return "human_win" if outcome.winner else "fluffy_win"

    def human_move(self, move_text: str) -> None:
        if self.result or self.turn != "human":
            raise ValueError("It is not your turn.")
        try:
            move = self.chess.Move.from_uci(str(move_text).lower())
        except ValueError as exc:
            raise ValueError("That is not a valid chess move.") from exc
        if move not in self.board.legal_moves:
            raise ValueError("That move is not legal in this position.")
        was_capture = self.board.is_capture(move)
        previous_eval = self.evaluation_cp
        san = self.board.san(move)
        self.board.push(move)
        self.last_move, self.last_move_san = move.uci(), san
        self.result = self._outcome()
        self.evaluation_cp = self._evaluate() if self.result is None else previous_eval
        swing = self.evaluation_cp - previous_eval
        if self.board.is_check():
            self.last_event = "human_gives_check"
        elif was_capture:
            self.last_event = "human_capture"
        elif swing <= -150:
            self.last_event = "human_mistake"
        elif swing >= 150:
            self.last_event = "human_strong_move"
        else:
            self.last_event = None
        self.event_move_san = san if self.last_event else None
        if self.result is None:
            self.turn = "fluffy"

    def fluffy_move(self) -> str:
        if self.result or self.turn != "fluffy":
            raise ValueError("It is not Fluffy's turn.")
        settings = CHESS_DIFFICULTIES[self.difficulty]
        human_event = self.last_event
        human_event_san = self.event_move_san
        result = self.engine.play(
            self.board,
            self.chess.engine.Limit(time=settings["move_time"]),
        )
        move = result.move
        was_capture = self.board.is_capture(move)
        san = self.board.san(move)
        self.board.push(move)
        self.last_move, self.last_move_san = move.uci(), san
        self.result = self._outcome()
        self.evaluation_cp = self._evaluate() if self.result is None else self.evaluation_cp
        if self.board.is_check():
            self.last_event = "fluffy_gives_check"
            self.event_move_san = san
        elif was_capture:
            self.last_event = "fluffy_capture"
            self.event_move_san = san
        else:
            # If Fluffy's reply is ordinary, retain a notable human event so
            # the commentator can still react once the complete turn ends.
            self.last_event = human_event
            self.event_move_san = human_event_san
        if self.result is None:
            self.turn = "human"
        return move.uci()

    def public_state(self) -> dict:
        return {
            "game": "chess",
            "fen": self.board.fen(),
            "turn": self.turn,
            "result": self.result,
            "difficulty": self.difficulty,
            "difficulty_label": CHESS_DIFFICULTIES[self.difficulty]["label"],
            "last_move": self.last_move,
            "last_move_san": self.last_move_san,
            "event": self.last_event,
            "event_move_san": self.event_move_san,
            "evaluation_cp": self.evaluation_cp,
            "in_check": self.board.is_check(),
            "legal_moves": [
                move.uci() for move in self.board.legal_moves
            ] if self.turn == "human" and self.result is None else [],
        }

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass


class GameServerClient:
    """Authenticated client for the web mailbox; it owns no game rules."""
    def __init__(self, base_url: str, secret: str, timeout: float = 3.0):
        import requests
        self.url = base_url.rstrip("/") + "/fluffy-brain.php"
        self.secret = secret
        self.timeout = timeout
        self.session = requests.Session()

    def exchange(self, state: dict, consume_move: bool = False) -> dict:
        response = self.session.post(
            self.url,
            headers={"X-Fluffy-Game-Secret": self.secret},
            json={"state": state, "consume_move": consume_move},
            timeout=(1.0, self.timeout),
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "game server rejected request")))
        return payload

    def close(self) -> None:
        self.session.close()


class FluffyGameCoordinator:
    """Non-blocking desktop state machine for phone-requested games."""
    def __init__(self, server: GameServerClient, is_idle: Callable[[], bool],
                 speak: Callable[[str], None],
                 on_game_lock: Callable[[bool], None],
                 commentary: Callable[[dict, Callable[[str], None]], bool] | None = None,
                 poll_seconds: float = 1.0, disconnect_seconds: float = 30.0,
                 queue_seconds: float = 60.0):
        self.server, self.is_idle, self.speak = server, is_idle, speak
        self.on_game_lock, self.commentary = on_game_lock, commentary
        self.poll_seconds, self.disconnect_seconds = poll_seconds, disconnect_seconds
        self.queue_seconds = queue_seconds
        self.phase, self.game, self.session_id = "normal", None, None
        self.requested_game, self.difficulty = None, None
        self.queued_at = self.last_player_seen = 0.0
        self.last_error = self.recent_result = None
        self._next_poll, self._pending_reply, self._polling = 0.0, None, False
        self._next_idle_check = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, **kwargs):
        secret = os.environ.get("FLUFFY_GAME_SECRET", "")
        if not secret:
            return None
        base = os.environ.get("FLUFFY_GAME_SERVER", "https://davesagenttools.com")
        return cls(GameServerClient(base, secret), **kwargs)

    @property
    def active(self) -> bool:
        return self.phase not in {"normal", "finished"}

    def context(self) -> dict:
        state = (
            self.game.public_state()
            if self.game
            else {"game": self.requested_game}
        )
        state.update({"phase": self.phase, "recent_result": self.recent_result})
        return state

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            reply, self._pending_reply = self._pending_reply, None
        if reply is not None:
            self._handle_reply(reply, now)
        if self.phase == "queued":
            if now - self.queued_at > self.queue_seconds:
                self._abort("I could not become ready in time, so the game was cancelled.")
            elif now >= self._next_idle_check and self.is_idle():
                self._start_game()
            else:
                self._next_idle_check = now + 0.5
        if self.active and self.last_player_seen and now - self.last_player_seen > self.disconnect_seconds:
            self._abort("The game disconnected. Open the page again when you want a new game.")
        if now >= self._next_poll and not self._polling:
            self._next_poll = now + self.poll_seconds
            self._polling = True
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        try:
            state = self.context() if self.phase != "normal" else {"phase": "normal"}
            reply = self.server.exchange(state, self.phase == "human_turn")
            with self._lock:
                self._pending_reply = reply
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self._polling = False

    def _handle_reply(self, reply: dict, now: float) -> None:
        remote = reply.get("session") or {}
        session_id = remote.get("id")
        if session_id:
            self.last_player_seen = now
        requested_game = remote.get("requested_game")
        if self.phase == "normal" and requested_game in {"tic_tac_toe", "chess"}:
            self.requested_game = str(requested_game)
            self.difficulty = str(remote.get("difficulty") or "friendly")
            self.session_id, self.phase, self.queued_at = str(session_id), "queued", now
            self.on_game_lock(True)
            game_name = "chess" if requested_game == "chess" else "tic-tac-toe"
            self.speak(
                f"{game_name.capitalize()} is queued. I will start as soon as I am idle."
            )
            return
        if self.phase == "finished":
            self._close_game()
            self.phase, self.game, self.session_id = "normal", None, None
            self.requested_game, self.difficulty = None, None
            return
        if self.session_id and session_id != self.session_id:
            return
        if self.phase == "human_turn" and reply.get("move") is not None:
            try:
                move = reply["move"]
                if self.requested_game == "tic_tac_toe":
                    move = int(move)
                self.game.human_move(move)
            except (ValueError, TypeError) as exc:
                self.speak(str(exc))
                return
            if self.game.result:
                self._finish()
            else:
                self.phase = "fluffy_turn"
                self.game.fluffy_move()
                if self.game.result:
                    self._finish()
                else:
                    self._maybe_comment()
                    self.phase = "human_turn"

    def _start_game(self) -> None:
        try:
            if self.requested_game == "chess":
                self.game = ChessGame(self.difficulty or "friendly")
                introduction = (
                    f"I am ready for {self.game.difficulty} chess. "
                    "You are White and you move first."
                )
            else:
                self.game = TicTacToe()
                introduction = "I am ready. You are crosses and you go first."
        except Exception as exc:
            print(f"[GAME] could not start {self.requested_game}: {exc}")
            self._abort(
                "I could not start that game. Please check the desktop game setup."
            )
            return
        self.phase = "human_turn"
        self.speak(introduction)

    def _maybe_comment(self) -> None:
        if (
            self.requested_game == "chess"
            and self.game is not None
            and self.game.last_event is not None
            and self.commentary is not None
        ):
            self.commentary(self.context(), self.speak)

    def _finish(self) -> None:
        self.recent_result, self.phase = self.game.result, "finishing"
        fallback = {"O": "Good game. I win this one.",
                    "X": "Good game. You beat me this time.",
                    "human_win": "Checkmate. You beat me. Good game.",
                    "fluffy_win": "Checkmate. Good game. I win this one.",
                    "draw": "Good game. That one was a draw."}.get(self.game.result, "Good game.")
        def done(text: str) -> None:
            self.speak(text)
            self.phase = "finished"
            self.on_game_lock(False)
        if not self.commentary or not self.commentary(self.context(), done):
            done(fallback)

    def _abort(self, message: str) -> None:
        self.speak(message)
        self._close_game()
        self.phase, self.game, self.session_id = "normal", None, None
        self.requested_game, self.difficulty = None, None
        self.on_game_lock(False)

    def _close_game(self) -> None:
        if self.game is not None:
            self.game.close()

    def close(self) -> None:
        self._close_game()
        self.server.close()
