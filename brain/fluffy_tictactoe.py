"""Hardware-independent game authority for the desktop Fluffy game worker.

The web relay submits only a square and a revision. Robot reactions are chosen
locally; no website-supplied speech, URLs or motion parameters are accepted.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple


LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
PREFERENCE = (4, 0, 2, 6, 8, 1, 3, 5, 7)
SQUARE_NAMES = ("top left", "top middle", "top right", "middle left",
                "centre", "middle right", "bottom left", "bottom middle",
                "bottom right")


def outcome(board: Tuple[str, ...]) -> Optional[str]:
    """Return X, O, draw, or None for an unfinished board."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if all(board) else None


def placed(board: Tuple[str, ...], square: int, player: str) -> Tuple[str, ...]:
    return board[:square] + (player,) + board[square + 1:]


@lru_cache(maxsize=20000)
def score(board: Tuple[str, ...], player: str) -> int:
    result = outcome(board)
    if result is not None:
        # Prefer earlier wins and postpone losses, without sacrificing a draw.
        return {"O": 1 + board.count(""), "X": -1 - board.count(""),
                "draw": 0}[result]
    values = [score(placed(board, i, player), "X" if player == "O" else "O")
              for i in PREFERENCE if not board[i]]
    return (max if player == "O" else min)(values)


def best_move(board: Tuple[str, ...]) -> int:
    """Choose O's best move on a valid, unfinished board."""
    if outcome(board) is not None:
        raise ValueError("game is already finished")
    return max((i for i in PREFERENCE if not board[i]),
               key=lambda i: score(placed(board, i, "O"), "X"))


@dataclass(frozen=True)
class Reaction:
    speech: str
    action: str


@dataclass
class Game:
    """One game: human X starts, desktop O responds in the same transaction.

    The worker must serialize calls and persist the resulting snapshot before
    acknowledging a relay message. A repeated revision cannot replay a move.
    """

    board: Tuple[str, ...] = field(default_factory=lambda: ("",) * 9)
    revision: int = 0
    stopped: bool = False

    def snapshot(self) -> dict:
        return {"board": list(self.board), "revision": self.revision,
                "outcome": outcome(self.board), "stopped": self.stopped,
                "turn": "X" if not self.stopped and outcome(self.board) is None else None}

    def stop(self) -> dict:
        """Latch this game stopped; starting again requires a new game."""
        if not self.stopped:
            self.stopped = True
            self.revision += 1
        return self.snapshot()

    def play(self, square: int, revision: int) -> Tuple[dict, Reaction]:
        if self.stopped:
            raise ValueError("game is stopped")
        if type(revision) is not int or revision != self.revision:
            raise ValueError("stale or invalid revision; refresh the board")
        if type(square) is not int or not 0 <= square < 9:
            raise ValueError("square must be an integer from 0 to 8")
        if outcome(self.board) is not None:
            raise ValueError("game is already finished")
        if self.board[square]:
            raise ValueError("square is occupied")
        board = placed(self.board, square, "X")
        move = None
        if outcome(board) is None:
            move = best_move(board)
            board = placed(board, move, "O")
        result = outcome(board)
        announcement = "" if move is None else f"I choose {SQUARE_NAMES[move]}. "
        if result == "O":
            reaction = Reaction(announcement + "Three in a row! Good game!", "wag_tail")
        elif result == "X":
            reaction = Reaction("You win! Well played!", "nod")
        elif result == "draw":
            reaction = Reaction(announcement + "A draw! You're a worthy opponent.", "nod")
        else:
            reaction = Reaction(announcement + "Your turn!", "think")
        self.board = board
        self.revision += 1
        return self.snapshot(), reaction
