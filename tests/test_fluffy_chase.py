import cv2
import numpy as np
import pytest
from fluffy_chase import TargetChase, red_target


def frame(x=320, width=60):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (x-width//2, 200), (x+width//2, 260), (0, 0, 255), -1)
    return image


def controller(result='accepted'):
    calls = []
    chase = TargetChase(lambda y,p: calls.append(('head',y,p)) or True,
                        lambda a: (calls.append(('move',a)) or (result,None,None)),
                        lambda: calls.append(('halt',)), lambda: True, interval=0.1)
    chase.start(0)
    return chase,calls


def test_detect_and_reject_ambiguous():
    assert red_target(frame())[2] == 61
    image = frame(200)
    image |= frame(440)
    assert red_target(image) is None
    assert red_target(np.zeros((480,640,3), dtype=np.uint8)) is None


def test_confirmed_target_moves_forward():
    chase,calls = controller()
    for now in (0.1,0.2,0.3):
        chase.update(frame(),now)
    assert [c for c in calls if c[0]=='move'] == [('move','forward')]


@pytest.mark.parametrize('image', [np.zeros((480,640,3),dtype=np.uint8),frame(width=310)])
def test_loss_and_close_target_halt(image):
    chase,calls = controller()
    chase.update(image,0.1)
    assert not chase.active
    assert calls == [('halt',)]


def test_camera_stall_and_stop_do_not_repeat_halt():
    chase,calls = controller()
    chase.watchdog(1.1)
    chase.stop('stop')
    chase.update(frame(),2)
    assert calls == [('halt',)]


def test_obstacle_disarms():
    chase,calls = controller('blocked')
    for now in (0.1,0.2,0.3,0.4):
        chase.update(frame(),now)
    assert not chase.active
    assert calls[-1] == ('halt',)
    assert len([c for c in calls if c[0]=='move']) == 1


def test_off_centre_target_turns_without_forward():
    chase,calls = controller()
    for i in range(1,25):
        chase.update(frame(520),i*0.1)
    assert ('move','turn_right') in calls
    assert ('move','forward') not in calls


def test_voice_phrase_wired_on_both_sides():
    import ast
    from pathlib import Path
    for filename,name in [('body/nox_voice.py','PHRASE_TO_COMMAND'),('pidog_yolo_vlm.py','VOICE_COMMANDS')]:
        tree=ast.parse(Path(filename).read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast.literal_eval(node.value)['chase target']=='chase_target'
