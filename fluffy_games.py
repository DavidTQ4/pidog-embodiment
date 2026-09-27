"""Deterministic games and the Dave's Agent Tools mailbox client."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
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
        state = self.game.public_state() if self.game else {"game": "tic_tac_toe"}
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
        if self.phase == "normal" and remote.get("requested_game") == "tic_tac_toe":
            self.session_id, self.phase, self.queued_at = str(session_id), "queued", now
            self.on_game_lock(True)
            self.speak("Tic-tac-toe is queued. I will start as soon as I am idle.")
            return
        if self.phase == "finished":
            self.phase, self.game, self.session_id = "normal", None, None
            return
        if self.session_id and session_id != self.session_id:
            return
        if self.phase == "human_turn" and reply.get("move") is not None:
            try:
                self.game.human_move(int(reply["move"]))
            except (ValueError, TypeError) as exc:
                self.speak(str(exc))
                return
            if self.game.result:
                self._finish()
            else:
                self.phase = "fluffy_turn"
                self.game.fluffy_move()
                self._finish() if self.game.result else setattr(self, "phase", "human_turn")

    def _start_game(self) -> None:
        self.game, self.phase = TicTacToe(), "human_turn"
        self.speak("I am ready. You are crosses and you go first.")

    def _finish(self) -> None:
        self.recent_result, self.phase = self.game.result, "finishing"
        fallback = {"O": "Good game. I win this one.",
                    "X": "Good game. You beat me this time.",
                    "draw": "Good game. That one was a draw."}.get(self.game.result, "Good game.")
        def done(text: str) -> None:
            self.speak(text)
            self.phase = "finished"
            self.on_game_lock(False)
        if not self.commentary or not self.commentary(self.context(), done):
            done(fallback)

    def _abort(self, message: str) -> None:
        self.speak(message)
        self.phase, self.game, self.session_id = "normal", None, None
        self.on_game_lock(False)

    def close(self) -> None:
        self.server.close()
