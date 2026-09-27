import time

import pytest

from brain.fluffy_game_worker import answer
from brain.fluffy_tictactoe import Game


def state():
    return {'id': 'test', 'expires': time.time() + 7200, 'stopped': False,
            'moves': [], 'board': [''] * 9, 'revision': 0,
            'pending': {'square': 0, 'created': time.time()}}


def test_rebuild_after_restart_and_duplicate_delivery():
    s = state()
    payload, reaction = answer(s)
    assert answer(s) == (payload, reaction)
    s.update(moves=[0], board=payload['board'], revision=1,
             pending={'square': 1, 'created': time.time()})
    payload, _ = answer(s)
    game = Game()
    game.play(0, 0)
    snapshot, _ = game.play(1, 1)
    assert payload['board'] == snapshot['board']


@pytest.mark.parametrize('change', [{'stopped': True}, {'expires': 0}, {'pending': None},
    {'pending': {'square': 0, 'created': 0}}])
def test_inactive_work_never_produces_reaction(change):
    s = state()
    s.update(change)
    assert answer(s) is None


def test_corrupt_history_is_rejected():
    s = state()
    s['board'][1] = 'X'
    with pytest.raises(ValueError, match='history'):
        answer(s)
