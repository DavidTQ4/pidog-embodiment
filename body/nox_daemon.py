#!/usr/bin/env python3
"""
nox_daemon.py — Nox's persistent body controller.
Runs as a systemd service. Holds GPIO, exposes a Unix socket API.
Skips broken sensors (ultrasonic, IMU).
"""

import os
import sys
import json
import time
import signal
import socket
import threading
import traceback
import math
from pathlib import Path

# Audio config for HifiBerry DAC (auto-detect card number)
os.environ["SDL_AUDIODRIVER"] = "alsa"

def _find_hifiberry_card():
    """Auto-detect HifiBerry DAC ALSA card number."""
    import subprocess as _sp
    try:
        result = _sp.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "hifiberry" in line.lower() and "card" in line.lower():
                card = line.split("card ")[1].split(":")[0]
                print(f"[nox] HifiBerry DAC found at card {card}", flush=True)
                return f"plughw:{card},0"
    except Exception as e:
        print(f"[nox] HifiBerry detection error: {e}", flush=True)
    # Fallback: try card 3 (typical Pi 4 with HifiBerry)
    print("[nox] HifiBerry not found, falling back to plughw:3,0", flush=True)
    return "plughw:3,0"

# nox.env may pin AUDIODEV for robots whose speaker isn't auto-detectable
_PLAYBACK_DEVICE = os.environ.get("AUDIODEV") or _find_hifiberry_card()
os.environ["AUDIODEV"] = _PLAYBACK_DEVICE

SOCKET_PATH = "/tmp/nox.sock"
PHOTO_DIR = "/tmp"
def _find_piper():
    """pip installs the piper CLI to ~/.local/bin OR /usr/local/bin depending on how it was run."""
    local = os.path.expanduser("~/.local/bin/piper")
    if os.path.exists(local):
        return local
    import shutil
    return shutil.which("piper") or local

PIPER_BIN = os.environ.get("PIPER_BIN") or _find_piper()
PIPER_MODEL = os.environ.get("PIPER_MODEL") or os.path.expanduser("~/.local/share/piper-voices/de_DE-thorsten-high.onnx")
SOUNDS_DIR = os.path.expanduser("~/pidog/sounds")

# ─── Ultrasonic distance sensor (separate from PiDog to avoid Process hang) ───
ultrasonic = None
ultrasonic_distance = -1.0
ultrasonic_sample_at = 0.0
ultrasonic_lock = threading.Lock()
ULTRASONIC_MAX_AGE_SECONDS = 0.75
HEAD_TRACKING_SPEED = 98

def _ultrasonic_bg_thread():
    """Background thread to continuously read ultrasonic distance."""
    global ultrasonic, ultrasonic_distance, ultrasonic_sample_at
    from robot_hat import Pin as RHPin
    from robot_hat.modules import Ultrasonic as US
    import time as t2
    try:
        echo = RHPin("D0")
        trig = RHPin("D1")
        us = US(trig, echo, timeout=0.02)
        print("[nox] Ultrasonic sensor initialized! ✅", flush=True)
        while True:
            try:
                d = us.read(times=3)
                with ultrasonic_lock:
                    ultrasonic_distance = d if d > 0 else -1.0
                    ultrasonic_sample_at = time.monotonic()
            except:
                pass
            t2.sleep(0.1)  # 10Hz reading
    except Exception as e:
        print(f"[nox] Ultrasonic init failed: {e}", flush=True)

# ─── Patch: Skip ultrasonic to prevent init hang ───
import pidog.pidog as _pidog_mod
_orig_init = _pidog_mod.Pidog.__init__

def _patched_init(self, *args, **kwargs):
    """Patched init that skips sensory_process_start (ultrasonic hangs)."""
    # Temporarily replace sensory_process_start with a no-op
    _orig_sensory = self.__class__.sensory_process_start
    self.__class__.sensory_process_start = lambda self_: None
    try:
        _orig_init(self, *args, **kwargs)
    finally:
        self.__class__.sensory_process_start = _orig_sensory

_pidog_mod.Pidog.__init__ = _patched_init

from pidog import Pidog

try:
    from pidog import preset_actions as _preset_actions
except Exception:
    _preset_actions = None

# I2C reachability diagnostics (issue #12) — stdlib only, ships next to us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nox_i2c_diag import mcu_diag  # noqa: E402

# ─── Global state ───
dog = None
camera_lock = threading.Lock()
dog_lock = threading.Lock()
# Serialises the decision to admit a leg action.  The SDK has its own motion
# buffers, but do_action() does not reject work while those buffers are busy.
motion_admission_lock = threading.RLock()
current_leg_action = None
motion_halt_latched = False
running = True

# ─── Servo Idle Management ───
_last_activity = time.time()
_idle_state = "active"  # active -> resting -> sleeping
IDLE_REST_SECS = 60    # lie down after 60s idle
IDLE_SLEEP_SECS = 120  # disable servos after 120s idle
def _is_sleep_hours():
    """Check if current time is in sleep hours (23:30-06:30)."""
    from datetime import datetime
    now = datetime.now()
    hour, minute = now.hour, now.minute
    if hour == 23 and minute >= 30:
        return True
    if hour < 7:
        if hour < 6 or (hour == 6 and minute <= 30):
            return True
    return False



def _mark_activity(internal=False):
    global _last_activity, _idle_state, _servo_pwm_disabled
    if internal:
        return  # Behavior engine auto-commands don't prevent sleep
    _last_activity = time.time()
    if _idle_state != "active":
        was_sleeping = _servo_pwm_disabled
        _idle_state = "active"
        _servo_pwm_disabled = False
        print(f"[nox] Activity detected, waking up{' (re-enabling servos)' if was_sleeping else ''}", flush=True)

def _idle_watchdog():
    global _idle_state, _servo_pwm_disabled
    while running:
        elapsed = time.time() - _last_activity
        if _idle_state == "active" and elapsed > IDLE_REST_SECS:
            _idle_state = "resting"
            print(f"[nox] Idle {int(elapsed)}s → lying down to save servos", flush=True)
            try:
                with dog_lock:
                    dog.do_action("lie", speed=50)
                    dog.rgb_strip.set_mode("breath", [0, 0, 0] if _is_sleep_hours() else [0, 0, 80], bps=0.3)
            except Exception as e:
                print(f"[nox] Idle lie failed: {e}", flush=True)
        elif _idle_state == "resting" and elapsed > IDLE_SLEEP_SECS:
            _idle_state = "sleeping"
            _servo_pwm_disabled = True
            print(f"[nox] Idle {int(elapsed)}s → deep sleep (servos off, LEDs dimmed)", flush=True)
            try:
                with dog_lock:
                    dog.rgb_strip.set_mode("breath", [0, 0, 0] if _is_sleep_hours() else [0, 0, 15], bps=0.15)
            except Exception as e:
                print(f"[nox] Deep sleep failed: {e}", flush=True)
        time.sleep(5)



# --- Servo Smoothing (Sprint 1) -----------------------------------------------
def _ease_in_out_cubic(t):
    """Cubic ease-in-out: smooth acceleration and deceleration. t in [0,1]."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


class SmoothServo:
    """EMA-filtered head tracking with deadband."""
    DEADBAND_DEG = 2.0   # ignore changes smaller than 2 degrees total
    # Autonomous tracking uses this immediate EMA path. A higher alpha
    # lets the head counter-steer a fast body turn without reintroducing the
    # blocking 350 ms eased movement used by manual head commands.
    EMA_ALPHA = 0.45

    def __init__(self):
        self._current = [0.0, 0.0, 0.0]  # yaw, roll, pitch
        self._target = [0.0, 0.0, 0.0]

    def update_target(self, yaw, roll, pitch):
        """Set new target. Returns True if change exceeds deadband."""
        new = [float(yaw), float(roll), float(pitch)]
        delta = sum(abs(new[i] - self._current[i]) for i in range(3))
        if delta < self.DEADBAND_DEG:
            return False
        self._target = new
        return True

    def ema_step(self):
        """One EMA step toward target. Returns interpolated [yaw, roll, pitch]."""
        for i in range(3):
            self._current[i] += self.EMA_ALPHA * (self._target[i] - self._current[i])
        return list(self._current)

    def snap_to(self, yaw, roll, pitch):
        """Hard-set current position (after eased move completes)."""
        self._current = [float(yaw), float(roll), float(pitch)]
        self._target = list(self._current)

    def get_current(self):
        return list(self._current)


_smooth_head = SmoothServo()
_servo_pwm_disabled = False  # True when sleeping (no head commands accepted)

def init_dog():
    """Initialize PiDog with broken sensors skipped."""
    global dog
    print("[nox] Initializing PiDog...", flush=True)
    dog = Pidog()
    time.sleep(0.5)
    # Wake up
    dog.do_action('stand', speed=60)
    time.sleep(1)
    dog.rgb_strip.set_mode('breath', [0, 0, 0] if _is_sleep_hours() else [128, 0, 255], bps=0.8)
    # Health check: report what's working
    _hw_status = []
    if hasattr(dog, 'music') and dog.music is not None:
        _hw_status.append("audio:pygame")
    else:
        _hw_status.append("audio:aplay-fallback")
    if hasattr(dog, 'pitch'):
        _hw_status.append("imu:ok")
    else:
        _hw_status.append("imu:unavailable")
    if hasattr(dog, 'dual_touch'):
        _hw_status.append("touch:ok")
    if hasattr(dog, 'ears'):
        _hw_status.append("ears:ok")
    print(f"[nox] Hardware: {', '.join(_hw_status)}", flush=True)
    print(f"[nox] Audio device: {_PLAYBACK_DEVICE}", flush=True)
    dead = _dead_action_threads()
    if dead:
        print(f"[nox] WARNING: action thread(s) already dead after init: "
              f"{', '.join(dead)} — the dog will NOT move!", flush=True)
    else:
        print("[nox] Action threads healthy (legs/head/tail consumers running)", flush=True)
    with dog_lock:
        diag = i2c_status(force=True)
    if diag.get("responding"):
        print(f"[nox] I2C: robot_hat MCU at {diag['mcu_addr']} responding", flush=True)
    else:
        print(f"[nox] WARNING: {diag.get('error')} — the dog will NOT move!", flush=True)
        print(f"[nox]   fix: {diag.get('hint')}", flush=True)
    print("[nox] PiDog ready. Nox lebt! ⚡", flush=True)


_battery_adc = None

def read_battery_voltage():
    """Battery voltage with fallback around a broken SDK path.

    robot_hat 2.5.2a1 ships a get_battery_voltage() that raises
    NameError: '_adc_obj' is not defined (issue #12). The measurement itself
    works — battery sits on ADC channel A4 behind a 1:3 divider — so on any
    SDK failure we read the ADC directly. Caller must hold dog_lock.
    """
    global _battery_adc
    try:
        v = round(dog.get_battery_voltage(), 2)
    except Exception:
        from robot_hat import ADC
        if _battery_adc is None:
            _battery_adc = ADC("A4")
        v = round(_battery_adc.read_voltage() * 3, 2)
    if v <= 0.0:
        # robot_hat swallows I2C errors and returns False per byte, and
        # (False << 8) + False == 0. An exact zero is far more often "the MCU
        # never answered" than "the rail is at 0 V" (issue #12: the dog ran on
        # battery alone and still read 0.0). Ask the bus before believing it.
        diag = i2c_status()
        if not diag.get("responding", True):
            raise RuntimeError(f"I2C: {diag.get('error')} | fix: {diag.get('hint')}")
    return v


_i2c_cache = {"ts": 0.0, "diag": None}
I2C_DIAG_TTL = 30.0  # seconds; a failed probe spawns i2cdetect, so cache it


def _mcu_i2c_object():
    """The robot_hat I2C object the SDK really drives the servos through, so the
    probe hits the address the SDK resolved — not one we guess."""
    global _battery_adc
    try:
        return dog.legs.servo_list[0]
    except Exception:
        pass
    try:
        from robot_hat import ADC
        if _battery_adc is None:
            _battery_adc = ADC("A4")
        return _battery_adc
    except Exception:
        return None


def _probe_mcu():
    """(resolved_address, raw) — one byte read through robot_hat's own retry
    path. An int means the MCU ACKed; False means every attempt failed and the
    library hid it. Caller must hold dog_lock."""
    obj = _mcu_i2c_object()
    if obj is None:
        return None, None
    try:
        from robot_hat import I2C
        raw = I2C.read(obj, 1)  # bypass ADC.read()'s combining, keep the retry wrapper
    except Exception:
        raw = False
    return getattr(obj, "address", None), raw


def i2c_status(force=False):
    """Cached verdict on whether the robot_hat MCU answers on I2C (issue #12).
    Caller must hold dog_lock."""
    now = time.time()
    cached = _i2c_cache["diag"]
    if cached is not None and not force and now - _i2c_cache["ts"] < I2C_DIAG_TTL:
        return cached
    addr, raw = _probe_mcu()
    diag = mcu_diag(addr, raw)
    _i2c_cache["ts"], _i2c_cache["diag"] = now, diag
    return diag


def _servo_power_missing():
    """True when the battery rail reads ~0 V, i.e. no pack is powering the servos.

    The Pi can run from USB-C alone, so the daemon, the bridge and every HTTP
    endpoint stay perfectly healthy while the servos have no power at all.
    do_action() then still succeeds -- it only writes PWM values -- and the dog
    reports ok:true without moving a millimetre (issue #12). Returns False when
    the ADC cannot be read: unknown is not the same as absent, and a broken ADC
    must not produce a false "no power" verdict. Caller must hold dog_lock.
    """
    try:
        return read_battery_voltage() <= 1.0
    except Exception:
        return False


def cmd_status():
    """System status."""
    import shutil
    info = {"hostname": os.uname().nodename, "uptime_s": int(float(open("/proc/uptime").read().split()[0]))}
    total, used, free = shutil.disk_usage("/")
    info["disk_free_gb"] = round(free / (1024**3), 1)
    try:
        with dog_lock:
            info["i2c"] = i2c_status()
    except Exception as e:
        info["i2c"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        with dog_lock:
            info["battery_v"] = read_battery_voltage()
    except Exception as e:
        # Keep "error" for compatibility, but expose WHY it failed: a working
        # battery with a failing ADC read looks identical otherwise (issue #12).
        info["battery_v"] = "error"
        info["battery_error"] = f"{type(e).__name__}: {e}"
    try:
        info["motion"] = cmd_motion_status()
    except Exception as e:
        info["motion"] = {
            "ok": False,
            "busy": True,
            "error": f"{type(e).__name__}: {e}",
        }
    return info


def _cmd_move(action, steps=3, speed=80, internal=False):
    """Queue a movement action and acknowledge without waiting for execution."""
    _mark_activity(internal=internal)
    if _idle_state == "sleeping":
        # Re-init servos before moving
        pass  # PiDog re-enables on do_action

    # SDK do_action() swallows unknown actions (prints and returns), so we
    # must route ourselves. Probe the CLASS for a property: instance hasattr
    # would execute the ActionDict property just to probe it.
    if isinstance(getattr(type(dog.actions_dict), action, None), property):
        with dog_lock:
            # Validate the consumers and controller before adding frames. The
            # old implementation slept for 1.5 seconds while holding this
            # lock, which prevented head counter-steering during a leg turn.
            dead = _dead_action_threads()
            if dead:
                return {"ok": False, "action": action,
                        "error": f"action thread(s) dead: {', '.join(dead)} — "
                                 "commands queue but are never executed",
                        "hint": "sudo systemctl restart nox-body; then check: "
                                "journalctl -u nox-body | grep -i exception"}
            i2c = i2c_status()
            if not i2c.get("responding", True):
                return {"ok": False, "action": action,
                        "error": f"servo controller unreachable — {i2c.get('error')}",
                        "hint": i2c.get("hint"), "i2c": i2c}
            servo_power_missing = _servo_power_missing()
            dog.do_action(action, step_count=int(steps), speed=int(speed))
            queued = _action_buffer_depth()

        # The hardware lock is deliberately released immediately after the
        # SDK accepts the frames. Head commands can now be queued while the
        # independent leg consumer executes the gait. Completion remains
        # authoritative through is_legs_done() plus the leg-buffer check.
        result = {
            "ok": True,
            "action": action,
            "execution": "queued",
            "non_blocking": True,
            "queued_motion_frames": queued,
        }
        if servo_power_missing:
            result["warning"] = (
                "battery rail reads 0.0 V - the servos have no power, so this "
                "action was queued in software only and the dog will not move")
            result["hint"] = ("check the battery: pack plugged in and the PiDog "
                              "power switch ON. The Pi keeps running from USB-C, "
                              "which is why everything else looks healthy")
        return result

    with dog_lock:
        # pant/bark/howling & friends live in pidog.preset_actions, not in
        # ActionDict (issue #12: 'ActionDict' object has no attribute 'pant').
        preset = getattr(_preset_actions, action, None) if _preset_actions else None
        if callable(preset):
            try:
                preset(dog)
                return {"ok": True, "action": action, "via": "preset"}
            except Exception as e:
                return {"ok": False, "action": action,
                        "error": f"preset action failed: {type(e).__name__}: {e}"}
        return {"ok": False, "action": action,
                "error": f"action '{action}' not supported by this pidog SDK",
                "supported": _supported_actions()}


def _dead_action_threads():
    """Names of SDK action-consumer threads that have died. Empty = healthy."""
    dead = []
    for name in ("legs_thread", "head_thread", "tail_thread"):
        t = getattr(dog, name, None)
        if t is not None and not t.is_alive():
            dead.append(name)
    return dead


def _action_buffer_depth():
    """Total motion frames waiting in the SDK's action buffers."""
    total = 0
    for name in ("legs_action_buffer", "head_action_buffer", "tail_action_buffer"):
        buf = getattr(dog, name, None)
        if buf is not None:
            total += len(buf)
    return total


def _legs_motion_state_locked():
    """Inspect leg completion while dog_lock is held by the caller."""
    try:
        legs_done = bool(dog.is_legs_done())
        state_error = None
    except Exception as e:
        # Unknown must be treated as busy; accepting a command would recreate
        # the unbounded queue failure this interface is intended to prevent.
        legs_done = False
        state_error = f"{type(e).__name__}: {e}"
    buffer = getattr(dog, "legs_action_buffer", None)
    buffered_frames = len(buffer) if buffer is not None else 0
    dead = _dead_action_threads()
    return {
        "legs_done": legs_done,
        "busy": (not legs_done) or buffered_frames > 0,
        "buffered_leg_frames": buffered_frames,
        "threads_dead": dead,
        "state_error": state_error,
    }


def cmd_motion_status():
    """Return authoritative SDK leg state without adding an action."""
    global current_leg_action
    with motion_admission_lock:
        with dog_lock:
            state = _legs_motion_state_locked()
        if not state["busy"]:
            current_leg_action = None
        state.update({
            "ok": (
                state["state_error"] is None
                and "legs_thread" not in state["threads_dead"]
            ),
            "current_action": current_leg_action,
            "halt_latched": motion_halt_latched,
        })
        return state


def _remember_leg_action(action, result):
    global current_leg_action
    if result.get("ok") and isinstance(
        getattr(type(dog.actions_dict), action, None),
        property,
    ):
        current_leg_action = action


def cmd_move(action, steps=3, speed=80, internal=False):
    """Legacy movement entry point, serialised but allowed to queue."""
    if motion_halt_latched and action in {
        "forward", "backward", "turn_left", "turn_right", "trot"
    }:
        return {
            "ok": True,
            "accepted": False,
            "halt_latched": True,
            "action": action,
            "reason": "locomotion is latched off; arm_motion is required",
        }
    with motion_admission_lock:
        result = _cmd_move(action, steps, speed, internal)
        _remember_leg_action(action, result)
        return result


def _ultrasonic_state():
    """Return one coherent distance sample for motion-safety decisions."""
    with ultrasonic_lock:
        distance = ultrasonic_distance
        sample_at = ultrasonic_sample_at
    age = time.monotonic() - sample_at if sample_at > 0 else None
    valid = distance > 0 and age is not None and age <= ULTRASONIC_MAX_AGE_SECONDS
    return {
        "distance_cm": round(distance, 1) if distance > 0 else None,
        "distance_valid": valid,
        "distance_age_s": round(age, 3) if age is not None else None,
    }


def cmd_move_if_idle(
    action,
    steps=3,
    speed=80,
    internal=False,
    min_distance_cm=None,
):
    """Atomically require idle legs and, when requested, safe clearance."""
    # Do not let concurrent HTTP requests become a second queue while another
    # action is being admitted.  A caller can retry after receiving busy:true.
    if not motion_admission_lock.acquire(blocking=False):
        return {
            "ok": True,
            "accepted": False,
            "busy": True,
            "legs_done": False,
            "buffered_leg_frames": None,
            "current_action": current_leg_action,
            "action": action,
            "note": "another movement request is being admitted",
        }
    try:
        if motion_halt_latched and action in {
            "forward", "backward", "turn_left", "turn_right", "trot"
        }:
            return {
                "ok": True,
                "accepted": False,
                "busy": False,
                "halt_latched": True,
                "action": action,
                "reason": "locomotion is latched off; arm_motion is required",
            }
        with dog_lock:
            state = _legs_motion_state_locked()
        if "legs_thread" in state["threads_dead"]:
            return {
                **state,
                "ok": False,
                "accepted": False,
                "busy": True,
                "action": action,
                "error": "leg action thread is not healthy",
            }
        if state["state_error"] is not None:
            return {
                **state,
                "ok": False,
                "accepted": False,
                "busy": True,
                "action": action,
                "error": f"cannot determine leg state: {state['state_error']}",
            }
        if state["busy"]:
            return {
                **state,
                "ok": True,
                "accepted": False,
                "action": action,
            }

        if min_distance_cm is not None:
            try:
                minimum = float(min_distance_cm)
            except (TypeError, ValueError):
                return {
                    **state,
                    "ok": False,
                    "accepted": False,
                    "action": action,
                    "error": "min_distance_cm must be a number",
                }
            if not 5.0 <= minimum <= 400.0:
                return {
                    **state,
                    "ok": False,
                    "accepted": False,
                    "action": action,
                    "error": "min_distance_cm must be between 5 and 400",
                }
            clearance = _ultrasonic_state()
            if not clearance["distance_valid"]:
                return {
                    **state,
                    **clearance,
                    "ok": True,
                    "accepted": False,
                    "safety_stop": True,
                    "action": action,
                    "min_distance_cm": minimum,
                    "reason": "ultrasonic reading is missing or stale",
                }
            if clearance["distance_cm"] <= minimum:
                return {
                    **state,
                    **clearance,
                    "ok": True,
                    "accepted": False,
                    "safety_stop": True,
                    "action": action,
                    "min_distance_cm": minimum,
                    "reason": "obstruction inside safety clearance",
                }

        result = _cmd_move(action, steps, speed, internal)
        _remember_leg_action(action, result)
        result["accepted"] = bool(result.get("ok"))
        with dog_lock:
            result.update(_legs_motion_state_locked())
        result["current_action"] = current_leg_action
        if min_distance_cm is not None:
            result.update(_ultrasonic_state())
            result["min_distance_cm"] = float(min_distance_cm)
        return result
    finally:
        motion_admission_lock.release()


def cmd_servo_test():
    """Split issue #12 in half: write servo angles DIRECTLY and synchronously,
    bypassing the SDK's buffer/thread machinery. If the dog moves here but not
    via /action, the consumer threads are the problem; if it doesn't move
    here either, the failure is below the SDK (robot_hat / MCU / power).

    Phase 2 goes one level deeper: a single-servo wiggle through a fresh
    robot_hat Servo object — the shortest possible path to the MCU, bypassing
    even the SDK's Robot class. Also reports who this process runs as:
    SunFounder examples run under sudo, the systemd units do not, and that
    privilege gap is a prime suspect when writes succeed but nothing moves.
    """
    import pwd
    import grp
    try:
        euid = os.geteuid()
        report_id = {
            "user": pwd.getpwuid(euid).pw_name,
            "euid": euid,
            "groups": sorted(grp.getgrgid(g).gr_name for g in os.getgroups()),
            "path": os.environ.get("PATH", ""),
        }
    except Exception as e:
        report_id = {"error": str(e)}
    report = {"process": report_id,
              "threads_dead": _dead_action_threads(),
              "buffered_frames": _action_buffer_depth()}
    with dog_lock:
        # Ask the bus first: if the MCU does not answer, both phases below
        # "succeed" (robot_hat swallows the errors) and nothing moves.
        report["i2c"] = i2c_status(force=True)
        try:
            frames, part = dog.actions_dict["sit"]
            report["phase1"] = "sit+stand poses via legs.servo_move (SDK Robot class)"
            dog.legs.servo_move(list(frames[-1]), 60)
            time.sleep(1.0)
            frames, part = dog.actions_dict["stand"]
            dog.legs.servo_move(list(frames[-1]), 60)
            time.sleep(1.0)
            report["phase1_ok"] = True
        except Exception as e:
            report["phase1_ok"] = False
            report["phase1_error"] = f"{type(e).__name__}: {e}"
        try:
            from robot_hat import Servo
            legs_pins = getattr(type(dog), "DEFAULT_LEGS_PINS", None) or [2]
            pin = legs_pins[0]
            report["phase2"] = f"single-servo wiggle on P{pin} via raw robot_hat Servo"
            s = Servo(f"P{pin}")
            for angle in (-20, 20, 0):
                s.angle(angle)
                time.sleep(0.4)
            report["phase2_ok"] = True
        except Exception as e:
            report["phase2_ok"] = False
            report["phase2_error"] = f"{type(e).__name__}: {e}"
    report["ok"] = bool(report.get("phase1_ok") or report.get("phase2_ok"))
    if not report["i2c"].get("responding", True):
        report["ok"] = False
        report["verdict"] = report["i2c"].get("error")
        report["hint"] = report["i2c"].get("hint")
        report["question"] = "nothing will have moved: the MCU never received the writes"
    else:
        report["question"] = ("phase1: did the dog sit+stand? "
                              "phase2: did ONE front leg wiggle left-right?")
    return report


def _supported_actions():
    """Everything cmd_move can execute: ActionDict properties + preset functions."""
    dict_actions = [n for n, v in vars(type(dog.actions_dict)).items()
                    if isinstance(v, property)]
    preset_fns = []
    if _preset_actions:
        preset_fns = [n for n in dir(_preset_actions)
                      if not n.startswith("_")
                      and callable(getattr(_preset_actions, n))
                      and getattr(getattr(_preset_actions, n), "__module__", "")
                      == _preset_actions.__name__]
    return sorted(set(dict_actions + preset_fns))


def cmd_head(yaw=0, roll=0, pitch=0, smooth=True, internal=False):
    """Move head with smooth easing + deadband filter."""
    _mark_activity(internal=internal)
    if _servo_pwm_disabled:
        return {"ok": False, "error": "servos sleeping", "hint": "send wake first"}
    yaw, roll, pitch = float(yaw), float(roll), float(pitch)
    # Deadband: skip if change is too small
    if not _smooth_head.update_target(yaw, roll, pitch):
        return {"ok": True, "head": [yaw, roll, pitch], "skipped": "deadband"}
    if not smooth:
        # Direct move (for resets/wake)
        with dog_lock:
            dog.head_move(
                [[yaw, roll, pitch]],
                immediately=True,
                speed=HEAD_TRACKING_SPEED,
            )
            time.sleep(0.3)
        _smooth_head.snap_to(yaw, roll, pitch)
        return {"ok": True, "head": [yaw, roll, pitch]}
    # Smooth eased interpolation (S-curve)
    start = _smooth_head.get_current()
    target = [yaw, roll, pitch]
    steps = 6
    duration = 0.35  # seconds total
    step_delay = duration / steps
    with dog_lock:
        for s in range(1, steps + 1):
            t = _ease_in_out_cubic(s / steps)
            pos = [start[i] + (target[i] - start[i]) * t for i in range(3)]
            dog.head_move(
                [pos],
                immediately=True,
                speed=HEAD_TRACKING_SPEED,
            )
            time.sleep(step_delay)
    _smooth_head.snap_to(yaw, roll, pitch)
    return {"ok": True, "head": [yaw, roll, pitch]}



def cmd_head_ema(yaw=0, roll=0, pitch=0, internal=False):
    """EMA-only head update for autonomous tracking (no easing, just smooth filter)."""
    _mark_activity(internal=internal)
    if _servo_pwm_disabled:
        return {"ok": False, "error": "servos sleeping"}
    yaw, roll, pitch = float(yaw), float(roll), float(pitch)
    if not _smooth_head.update_target(yaw, roll, pitch):
        return {"ok": True, "skipped": "deadband"}
    pos = _smooth_head.ema_step()
    with dog_lock:
        dog.head_move(
            [pos],
            immediately=True,
            speed=HEAD_TRACKING_SPEED,
        )
    return {"ok": True, "head": pos}

_VALID_RGB_STYLES = {"monochromatic", "breath", "boom", "bark", "speak", "listen"}

def cmd_rgb(r=128, g=0, b=255, mode="breath", bps=0.8):
    """Set RGB LEDs."""
    if not isinstance(mode, str) or mode not in _VALID_RGB_STYLES and mode != "off":
        mode = "breath"
    with dog_lock:
        color = [int(r), int(g), int(b)]
        if mode == "off":
            dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
        else:
            dog.rgb_strip.set_mode(mode, color, bps=float(bps))
    return {"ok": True, "rgb": [r, g, b], "mode": mode}


# ─── Persistent Camera ───
_camera_ready = False
_camera_init_lock = threading.Lock()

def _ensure_camera():
    """Initialize camera once and keep it running."""
    global _camera_ready
    if _camera_ready:
        return True
    with _camera_init_lock:
        if _camera_ready:
            return True
        try:
            from vilib import Vilib
            Vilib.camera_start(vflip=False, hflip=False)
            time.sleep(2)
            
            # start ViLibs continous MJPEG stream
            Vilib.display(local=False, web=True)
            
            _camera_ready = True
            print("[nox] Camera initialized with MJPEG stream on port 9000", flush=True)
            return True
        except Exception as e:
            print(f"[nox] Camera init failed: {e}", flush=True)
            return False


def cmd_photo(path=None):
    """Take a photo using persistent camera."""
    if path is None:
        path = os.path.join(PHOTO_DIR, "nox_snap.jpg")
    basename = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path) or PHOTO_DIR

    with camera_lock:
        if not _ensure_camera():
            return {"ok": False, "error": "camera not available"}
        from vilib import Vilib
        Vilib.take_photo(basename, dirname)

    actual = os.path.join(dirname, basename + ".jpg")
    if os.path.exists(actual):
        return {"ok": True, "photo": actual}
    return {"ok": False, "error": f"photo file not created: {actual}"}


def cmd_speak(text):
    """TTS via Piper + aplay/PiDog sound_effect. Fully async to avoid TCP timeout."""
    _mark_activity()
    # Guard: empty or whitespace-only text crashes Piper
    if not text or not text.strip():
        return {"ok": False, "error": "empty text"}
    # Fail loudly instead of returning ok while the async pipeline dies (issue #5):
    # piper voice models are NOT installed by pip — the user must download one.
    if not os.path.exists(PIPER_MODEL):
        return {"ok": False, "error": (
            f"piper voice model not found: {PIPER_MODEL} — download a voice from "
            "https://huggingface.co/rhasspy/piper-voices and set PIPER_MODEL in nox.env"
        )}
    if not os.path.exists(PIPER_BIN):
        return {"ok": False, "error": (
            f"piper binary not found: {PIPER_BIN} — pip3 install piper-tts, "
            "or set PIPER_BIN in nox.env"
        )}

    def _tts_pipeline(speak_text):
        import subprocess as sp
        # Use unique wav per call to avoid race conditions
        wav = f"/tmp/nox_speak_{int(time.time()*1000) % 100000}.wav"
        try:
            safe_text = speak_text.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
            proc = sp.run(
                f'echo "{safe_text}" | {PIPER_BIN} --model {PIPER_MODEL} --output_file {wav}',
                shell=True, capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0:
                print(f"[nox] Piper TTS failed: {proc.stderr}", flush=True)
                return
            # Play
            played = False
            try:
                with dog_lock:
                    if hasattr(dog, 'music') and dog.music is not None:
                        dog.music.sound_play(wav)
                        played = True
            except Exception as e:
                print(f"[nox] pygame playback failed: {e}", flush=True)
            if not played:
                try:
                    sp.run(["aplay", "-D", _PLAYBACK_DEVICE, wav],
                           capture_output=True, timeout=60)
                except Exception as e2:
                    print(f"[nox] aplay also failed: {e2}", flush=True)
        finally:
            # Cleanup temp wav (after short delay for playback to finish)
            try:
                time.sleep(0.5)
                os.remove(wav)
            except:
                pass

    import threading
    threading.Thread(target=_tts_pipeline, args=(text,), daemon=True).start()
    return {"ok": True, "spoke": text}


def cmd_sound(name):
    """Play built-in sound (wav via pygame/aplay, mp3 via ffplay/pygame)."""
    import subprocess
    for ext in ['', '.wav', '.mp3']:
        path = os.path.join(SOUNDS_DIR, name + ext)
        if os.path.exists(path):
            played = False
            # Try PiDog's pygame mixer first (handles wav and mp3)
            try:
                with dog_lock:
                    if hasattr(dog, 'music') and dog.music is not None:
                        dog.music.sound_play(path)
                        played = True
            except Exception as e:
                print(f"[nox] pygame sound failed: {e}", flush=True)
            # Fallback: aplay for wav, ffplay for mp3
            if not played:
                try:
                    if path.endswith('.mp3'):
                        # Try ffplay (from ffmpeg), mpg123, or sox in order
                        for player in [["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                                       ["mpg123", "-q", path],
                                       ["sox", path, "-d"]]:
                            try:
                                subprocess.run(player, capture_output=True, timeout=15)
                                played = True
                                break
                            except FileNotFoundError:
                                continue
                    else:
                        subprocess.run(["aplay", "-D", _PLAYBACK_DEVICE, path],
                                       capture_output=True, timeout=15)
                        played = True
                except Exception as e2:
                    return {"ok": False, "error": f"playback failed: {e2}"}
            if not played:
                return {"ok": False, "error": "no suitable audio player found (install ffmpeg or mpg123)"}
            return {"ok": True, "sound": name}
    try:
        available = [f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3'))]
    except:
        available = []
    return {"ok": False, "error": "not found", "available": available}


def cmd_combo(sequence):
    """Run action sequence. Format: action1:steps:speed,action2:steps:speed"""
    results = []
    with dog_lock:
        for part in sequence.split(","):
            parts = part.strip().split(":")
            action = parts[0]
            steps = int(parts[1]) if len(parts) > 1 else 3
            speed = int(parts[2]) if len(parts) > 2 else 80
            dog.do_action(action, step_count=steps, speed=speed)
            time.sleep(1.2)
            results.append(action)
    return {"ok": True, "combo": results}


def cmd_wake():
    """Wake up sequence."""
    _mark_activity()
    with dog_lock:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.do_action('wag_tail', step_count=5, speed=80)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [0, 0, 0] if _is_sleep_hours() else [128, 0, 255], bps=0.8)
    return {"ok": True}


def cmd_keep_awake():
    """Renew the controller activity lease without moving any servos."""
    _mark_activity()
    return {"ok": True, "idle_state": _idle_state}


def cmd_sleep():
    """Sleep sequence."""
    with dog_lock:
        dog.do_action('lie', speed=50)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [0, 0, 0] if _is_sleep_hours() else [0, 0, 80], bps=0.3)
    return {"ok": True}


def cmd_reset():
    """Reset to neutral standing."""
    with dog_lock:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.head_move([[0, 0, 0]], immediately=True, speed=60)
        dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
    return {"ok": True}


# ─── Sensor Commands ───
def cmd_sensors():
    """Complete sensor readout."""
    result = {"ts": time.time()}
    
    # Battery
    try:
        with dog_lock:
            v = read_battery_voltage()
            result["battery_v"] = round(v, 2)
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            result["charging"] = v > 8.35
    except Exception as e:
        result["battery_error"] = str(e)
    
    # IMU
    try:
        with dog_lock:
            result["imu"] = {
                "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
                "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
            }
    except Exception as e:
        result["imu_error"] = str(e)
    
    # Touch
    try:
        with dog_lock:
            touch = dog.dual_touch.read()
            result["touch"] = touch
    except Exception as e:
        result["touch_error"] = str(e)
    
    # Sound Direction
    try:
        with dog_lock:
            detected = dog.ears.isdetected()
            result["sound"] = {
                "detected": detected,
                "direction": dog.ears.read() if detected else None,
            }
    except Exception as e:
        result["sound_error"] = str(e)
    
    # Ultrasonic (from background thread)
    result.update(_ultrasonic_state())
    
    # System
    import shutil
    total, used, free = shutil.disk_usage("/")
    try:
        mem_lines = open("/proc/meminfo").readlines()
        mem_avail = int(mem_lines[2].split()[1]) // 1024
    except:
        mem_avail = 0
    result["system"] = {
        "hostname": os.uname().nodename,
        "uptime_s": int(float(open("/proc/uptime").read().split()[0])),
        "disk_free_gb": round(free / (1024**3), 1),
        "mem_available_mb": mem_avail,
    }
    
    return result


def cmd_body_state():
    """Current body state: servo angles, posture."""
    result = {}
    
    with dog_lock:
        result["leg_angles"] = list(dog.leg_current_angles) if hasattr(dog, 'leg_current_angles') else None
        result["head_angles"] = list(dog.head_current_angles) if hasattr(dog, 'head_current_angles') else None
        result["tail_angles"] = list(dog.tail_current_angles) if hasattr(dog, 'tail_current_angles') else None
        
        # Posture estimation
        try:
            la = dog.leg_current_angles
            if la:
                if all(abs(a) < 50 for a in la):
                    result["posture"] = "lying"
                elif la[4] > 60 and la[6] < -60:
                    result["posture"] = "sitting"
                else:
                    result["posture"] = "standing"
        except:
            result["posture"] = "unknown"
        
        try:
            v = read_battery_voltage()
            result["battery_v"] = round(v, 2)
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            result["charging"] = v > 8.35
        except:
            pass
    
    return result


def cmd_imu():
    """Raw IMU data."""
    with dog_lock:
        return {
            "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
            "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
        }


def cmd_touch():
    """Touch sensor state."""
    with dog_lock:
        touch = dog.dual_touch.read()
        return {
            "touch": touch,
            "touched": touch != "N",
            "side": {"N": "none", "L": "left", "R": "right", "LS": "slide-left", "RS": "slide-right"}.get(touch, touch)
        }


def cmd_ears():
    """Sound direction sensor."""
    with dog_lock:
        detected = dog.ears.isdetected()
        return {
            "detected": detected,
            "direction_deg": dog.ears.read() if detected else None,
        }


def cmd_scan():
    """Scan surroundings by sweeping head."""
    positions = [(-40, 0, 0), (0, 0, 0), (40, 0, 0)]
    with dog_lock:
        for yaw, roll, pitch in positions:
            dog.head_move([[yaw, roll, pitch]], immediately=True, speed=80)
            time.sleep(0.8)
        dog.head_move([[0, 0, 0]], immediately=True, speed=80)
    return {"ok": True, "scanned": ["left", "center", "right"]}



def cmd_scan_sweep(angles=None, settle_ms=200, samples=3):
    """Sweep head across angles and read ultrasonic distance at each position.
    Returns {angle: distance_cm} map for obstacle detection.
    Adapted from HoundMind ScanningService pattern."""
    if angles is None:
        angles = [-45, -30, -15, 0, 15, 30, 45]
    _mark_activity()
    result = {}
    with dog_lock:
        for angle in angles:
            dog.head_move([[float(angle), 0, 0]], immediately=True, speed=70)
            time.sleep(settle_ms / 1000.0)
            # Read multiple ultrasonic samples, take median
            readings = []
            for _ in range(samples):
                with ultrasonic_lock:
                    d = ultrasonic_distance
                if d > 0:
                    readings.append(d)
                time.sleep(0.04)
            if readings:
                readings.sort()
                result[str(angle)] = round(readings[len(readings) // 2], 1)
            else:
                result[str(angle)] = -1
        # Return head to center
        dog.head_move([[0, 0, 0]], immediately=True, speed=70)
    return {"ok": True, "scan": result, "timestamp": time.time()}


def _clear_action_buffers(names):
    """Clear queued SDK frames and return the number discarded."""
    cleared_frames = 0
    for name in names:
        buffer = getattr(dog, name, None)
        if buffer is not None and hasattr(buffer, "clear"):
            cleared_frames += len(buffer)
            buffer.clear()
    return cleared_frames


def cmd_halt(speed=40):
    """Stop queued locomotion and settle into a stable standing posture.

    Head and tail buffers are deliberately left alone so visual tracking can
    continue after a spoken stop.  This is an urgent local command and does not
    wait for the desktop controller or network tunnel.
    """
    global current_leg_action, _idle_state, motion_halt_latched
    speed = max(20, min(60, int(speed)))
    print(f"[nox] HALT requested; stabilising at speed {speed}", flush=True)
    motion_halt_latched = True
    with motion_admission_lock:
        with dog_lock:
            cleared_frames = _clear_action_buffers(("legs_action_buffer",))
            try:
                dog.do_action("stand", speed=speed)
                current_leg_action = "stand"
            except Exception as exc:
                return {
                    "ok": False,
                    "halted": False,
                    "error": f"failed to queue stable stand: {exc}",
                    "cleared_motion_frames": cleared_frames,
                }
    _mark_activity()
    return {
        "ok": True,
        "halted": True,
        "posture": "stand",
        "speed": speed,
        "cleared_motion_frames": cleared_frames,
    }


def cmd_lie_down(speed=40):
    """Discard queued locomotion and enter a controlled lying posture."""
    global current_leg_action, _idle_state, motion_halt_latched
    speed = max(20, min(60, int(speed)))
    print(f"[nox] LIE DOWN requested at speed {speed}", flush=True)
    motion_halt_latched = True
    with motion_admission_lock:
        with dog_lock:
            cleared_frames = _clear_action_buffers(("legs_action_buffer",))
            try:
                dog.do_action("lie", speed=speed)
                current_leg_action = "lie"
            except Exception as exc:
                return {
                    "ok": False,
                    "lying_down": False,
                    "error": f"failed to queue lie posture: {exc}",
                    "cleared_motion_frames": cleared_frames,
                }
    _idle_state = "resting"
    return {
        "ok": True,
        "lying_down": True,
        "posture": "lie",
        "speed": speed,
        "cleared_motion_frames": cleared_frames,
    }


def cmd_emergency_stop():
    """Emergency stop: immediately cease all movement and lie down.
    Adapted from HoundMind SafetyModule pattern."""
    global _idle_state, _servo_pwm_disabled, current_leg_action, motion_halt_latched
    print("[nox] EMERGENCY STOP triggered!", flush=True)
    motion_halt_latched = True
    cleared_frames = 0
    with dog_lock:
        # Clear pending SDK frames before adding the single safe posture.  This
        # cannot undo a servo frame already executing, but it prevents the
        # remaining backlog from running first.
        cleared_frames = _clear_action_buffers((
            "legs_action_buffer",
            "head_action_buffer",
            "tail_action_buffer",
        ))
        try:
            dog.do_action("lie", speed=100)
            current_leg_action = "lie"
        except Exception:
            pass
        try:
            dog.rgb_strip.set_mode("boom", [255, 0, 0], bps=2.0)
        except Exception:
            pass
    _idle_state = "resting"
    return {
        "ok": True,
        "emergency": True,
        "cleared_motion_frames": cleared_frames,
    }


def cmd_arm_motion():
    """Explicitly clear the local halt latch before identity following."""
    global motion_halt_latched
    motion_halt_latched = False
    _mark_activity()
    print("[nox] Locomotion halt latch ARMED/OFF", flush=True)
    return {"ok": True, "motion_armed": True, "halt_latched": False}


def cmd_three_way_scan():
    """Quick 3-direction scan: left, forward, right.
    Faster than full sweep — for real-time obstacle avoidance during patrol."""
    _mark_activity()
    result = {}
    with dog_lock:
        # Forward
        dog.head_move([[0, 0, 0]], immediately=True, speed=80)
        time.sleep(0.15)
        readings = []
        for _ in range(3):
            with ultrasonic_lock:
                d = ultrasonic_distance
            if d > 0:
                readings.append(d)
            time.sleep(0.03)
        result["forward"] = round(sorted(readings)[len(readings) // 2], 1) if readings else -1

        # Left
        dog.head_move([[40, 0, 0]], immediately=True, speed=80)
        time.sleep(0.15)
        readings = []
        for _ in range(3):
            with ultrasonic_lock:
                d = ultrasonic_distance
            if d > 0:
                readings.append(d)
            time.sleep(0.03)
        result["left"] = round(sorted(readings)[len(readings) // 2], 1) if readings else -1

        # Right
        dog.head_move([[-40, 0, 0]], immediately=True, speed=80)
        time.sleep(0.15)
        readings = []
        for _ in range(3):
            with ultrasonic_lock:
                d = ultrasonic_distance
            if d > 0:
                readings.append(d)
            time.sleep(0.03)
        result["right"] = round(sorted(readings)[len(readings) // 2], 1) if readings else -1

        # Return to center
        dog.head_move([[0, 0, 0]], immediately=True, speed=80)

    return {"ok": True, "scan": result, "timestamp": time.time()}


# ─── Command dispatcher ───
COMMANDS = {
    "status": lambda args: cmd_status(),
    "servo_test": lambda args: cmd_servo_test(),
    "move": lambda args: cmd_move(args.get("action", "stand"), args.get("steps", 3), args.get("speed", 80), internal=args.get("_internal", False)),
    "move_if_idle": lambda args: cmd_move_if_idle(args.get("action", "stand"), args.get("steps", 3), args.get("speed", 80), internal=args.get("_internal", False), min_distance_cm=args.get("min_distance_cm")),
    "motion_status": lambda args: cmd_motion_status(),
    "head": lambda args: cmd_head(args.get("yaw", 0), args.get("roll", 0), args.get("pitch", 0), args.get("smooth", True), internal=args.get("_internal", False)),
    "head_ema": lambda args: cmd_head_ema(args.get("yaw", 0), args.get("roll", 0), args.get("pitch", 0), internal=args.get("_internal", False)),
    "rgb": lambda args: cmd_rgb(args.get("r", 128), args.get("g", 0), args.get("b", 255), args.get("mode", "breath"), args.get("bps", 0.8)),
    "photo": lambda args: cmd_photo(args.get("path")),
    "speak": lambda args: cmd_speak(args.get("text", "")),
    "sound": lambda args: cmd_sound(args.get("name", "single_bark_1")),
    "combo": lambda args: cmd_combo(args.get("sequence", "stand:1:60")),
    "wake": lambda args: cmd_wake(),
    "keep_awake": lambda args: cmd_keep_awake(),
    "sleep": lambda args: cmd_sleep(),
    "reset": lambda args: cmd_reset(),
    "ping": lambda args: {"pong": True, "ts": time.time()},
    "sensors": lambda args: cmd_sensors(),
    "body_state": lambda args: cmd_body_state(),
    "imu": lambda args: cmd_imu(),
    "touch": lambda args: cmd_touch(),
    "ears": lambda args: cmd_ears(),
    "scan": lambda args: cmd_scan(),
    "scan_sweep": lambda args: cmd_scan_sweep(args.get("angles"), args.get("settle_ms", 200), args.get("samples", 3)),
    "three_way_scan": lambda args: cmd_three_way_scan(),
    "halt": lambda args: cmd_halt(args.get("speed", 40)),
    "lie_down": lambda args: cmd_lie_down(args.get("speed", 40)),
    "arm_motion": lambda args: cmd_arm_motion(),
    "emergency_stop": lambda args: cmd_emergency_stop(),
}


def handle_client(conn):
    """Handle a single client connection."""
    try:
        data = conn.recv(4096).decode('utf-8').strip()
        if not data:
            return

        try:
            request = json.loads(data)
        except json.JSONDecodeError:
            # Simple text command: "move sit" or "speak Hallo"
            parts = data.split(None, 1)
            cmd_name = parts[0]
            if len(parts) > 1:
                # Try to parse remaining as key=value or just pass as first arg
                request = {"cmd": cmd_name}
                remaining = parts[1]
                # Simple arg parsing
                if cmd_name == "move":
                    request["action"] = remaining.split()[0]
                elif cmd_name == "speak":
                    request["text"] = remaining
                elif cmd_name == "sound":
                    request["name"] = remaining
                elif cmd_name == "rgb":
                    rgb_parts = remaining.split()
                    if len(rgb_parts) >= 3:
                        request["r"], request["g"], request["b"] = int(rgb_parts[0]), int(rgb_parts[1]), int(rgb_parts[2])
                    if len(rgb_parts) >= 4:
                        request["mode"] = rgb_parts[3]
                elif cmd_name == "head":
                    h_parts = remaining.split()
                    if len(h_parts) >= 3:
                        request["yaw"], request["roll"], request["pitch"] = float(h_parts[0]), float(h_parts[1]), float(h_parts[2])
                elif cmd_name == "combo":
                    request["sequence"] = remaining
                elif cmd_name == "photo":
                    request["path"] = remaining
            else:
                request = {"cmd": cmd_name}

        cmd_name = request.get("cmd", request.get("command", ""))
        # Log external commands so `journalctl -u nox-body` shows what arrived.
        # "status" is polled continuously by the bridge and would drown the log.
        if cmd_name != "status" and not request.get("_internal"):
            print(f"[nox] cmd: {json.dumps(request)[:200]}", flush=True)
        if cmd_name in COMMANDS:
            result = COMMANDS[cmd_name](request)
            response = json.dumps(result)
        else:
            response = json.dumps({"error": f"unknown command: {cmd_name}", "available": list(COMMANDS.keys())})

        conn.sendall((response + "\n").encode('utf-8'))
    except Exception as e:
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))
        except:
            pass
        traceback.print_exc()
    finally:
        conn.close()


def socket_server():
    """Unix socket server for receiving commands."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    server.listen(5)
    server.settimeout(1.0)
    print(f"[nox] Socket server listening on {SOCKET_PATH}", flush=True)

    while running:
        try:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"[nox] Socket error: {e}", flush=True)

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)


# Also listen on TCP for remote access from Nox's Pi
def tcp_server(port=9999):
    """TCP server for remote commands from Nox's Pi."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"[nox] TCP server listening on port {port}", flush=True)

    while running:
        try:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"[nox] TCP error: {e}", flush=True)

    server.close()


def shutdown(signum, frame):
    """Clean shutdown."""
    global running
    print(f"\n[nox] Shutting down (signal {signum})...", flush=True)
    running = False
    try:
        with dog_lock:
            dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
            dog.do_action('lie', speed=50)
            time.sleep(1)
            dog.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    init_dog()

    # Start servers
    sock_thread = threading.Thread(target=socket_server, daemon=True)
    sock_thread.start()

    tcp_thread = threading.Thread(target=tcp_server, daemon=True)
    tcp_thread.start()

    print("[nox] All systems go. Waiting for commands...", flush=True)
    
    # Start idle watchdog thread
    idle_thread = threading.Thread(target=_idle_watchdog, daemon=True)
    idle_thread.start()

    # Start ultrasonic background thread
    us_thread = threading.Thread(target=_ultrasonic_bg_thread, daemon=True)
    us_thread.start()

    # Keep main thread alive
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(signal.SIGINT, None)
