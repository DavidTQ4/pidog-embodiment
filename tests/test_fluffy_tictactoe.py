import pytest

from brain.fluffy_tictactoe import Game, best_move, outcome, placed


def test_fluffy_never_loses_against_any_human_sequence():
    def visit(board):
        result = outcome(board)
        assert result != "X"
        if result is not None:
            return
        for square in range(9):
            if not board[square]:
                after = placed(board, square, "X")
                assert outcome(after) != "X"
                if outcome(after) is None:
                    after = placed(after, best_move(after), "O")
                visit(after)
    visit(("",) * 9)


def test_takes_immediate_win():
    assert best_move(("O", "O", "", "X", "X", "", "X", "", "")) == 2


@pytest.mark.parametrize("square", [-1, 9, True, "1", None, 1.5])
def test_invalid_inputs_do_not_change_state(square):
    game = Game()
    before = game.snapshot()
    with pytest.raises(ValueError):
        game.play(square, 0)
    assert game.snapshot() == before


def test_retries_cannot_replay_moves():
    game = Game()
    snapshot, reaction = game.play(0, 0)
    assert snapshot["board"].count("X") == 1
    assert snapshot["board"].count("O") == 1
    assert reaction.action in {"think", "nod", "wag_tail"}
    with pytest.raises(ValueError, match="revision"):
        game.play(1, 0)
    with pytest.raises(ValueError, match="occupied"):
        game.play(0, 1)
    assert game.snapshot() == snapshot


def test_stop_is_latched_and_idempotent():
    game = Game()
    stopped = game.stop()
    assert game.stop() == stopped
    with pytest.raises(ValueError, match="stopped"):
        game.play(0, stopped["revision"])


def test_finished_game_rejects_further_moves():
    game = Game(board=("X", "O", "X", "X", "O", "O", "O", "X", "X"))
    with pytest.raises(ValueError, match="finished"):
        game.play(0, 0)
