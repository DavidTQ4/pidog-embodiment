#!/usr/bin/env python3
"""Local, offline voice-command service for PiDog.

The Raspberry Pi owns microphone capture and safety-critical stop/lie actions.
Non-safety commands are placed in nox_brain_bridge's voice inbox for the
desktop YOLO controller.  No cloud service or desktop connection is required
for a spoken stop.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

from vosk import KaldiRecognizer, Model, SetLogLevel


SAMPLE_RATE = 16000
WAKE_WORDS = ("nox", "knox", "knocks")
PHRASE_TO_COMMAND = {
    "select me": "select_me",
    "arm head": "arm_head",
    "track me": "arm_head",
    "follow me": "follow_me",
    "stop": "stop",
    "emergency stop": "stop",
    "lie down": "lie_down",
    "lay down": "lie_down",
    "stop and lie down": "stop_and_lie_down",
    "stop and lay down": "stop_and_lie_down",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("NOX_VOSK_MODEL", ""),
        help="Path to an unpacked Vosk model (or set NOX_VOSK_MODEL).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NOX_AUDIO_DEVICE", "plughw:2,0"),
        help="ALSA capture device passed to arecord.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=float(os.environ.get("NOX_VOICE_GAIN", "1.0")),
        help="Software input gain; start at 1.0 and increase only if needed.",
    )
    parser.add_argument(
        "--bridge",
        default=os.environ.get("NOX_BRIDGE_URL", "http://127.0.0.1:8888"),
    )
    parser.add_argument(
        "--daemon-host",
        default=os.environ.get("NOX_DAEMON_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--daemon-port",
        type=int,
        default=int(os.environ.get("NOX_DAEMON_PORT", "9999")),
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=float(os.environ.get("NOX_VOICE_DEBOUNCE", "1.5")),
    )
    parser.add_argument(
        "--stop-without-wake",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NOX_STOP_WITHOUT_WAKE", "1") != "0",
        help="Accept bare 'stop' as a fail-safe.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=float(os.environ.get("NOX_VOICE_MIN_CONFIDENCE", "0.60")),
        help="Minimum mean Vosk confidence for wake-word commands.",
    )
    parser.add_argument(
        "--bare-stop-minimum-confidence",
        type=float,
        default=float(os.environ.get("NOX_BARE_STOP_MIN_CONFIDENCE", "0.72")),
        help="Higher minimum confidence for a stop without the wake word.",
    )
    return parser.parse_args()


def discover_model(configured: str) -> Path:
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"configured Vosk model does not exist: {path}")

    home = Path.home()
    candidates = [
        home / "robot-hat/examples/vosk-model-small-en-us-0.15",
        home / "robot-hat/vosk-model-small-en-us-0.15",
        home / ".local/share/vosk/vosk-model-small-en-us-0.15",
        home / ".cache/vosk/vosk-model-small-en-us-0.15",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # The official example may have been unpacked under a slightly different
    # version name. Restrict the search to likely locations.
    for root in (home / "robot-hat", home / ".local/share", home / ".cache"):
        if root.is_dir():
            for candidate in root.glob("**/vosk-model-small-en*"):
                if candidate.is_dir():
                    return candidate

    raise FileNotFoundError(
        "no Vosk model found; set NOX_VOSK_MODEL to the unpacked model directory"
    )


def grammar_phrases(stop_without_wake: bool) -> list[str]:
    phrases = [
        f"{wake} {phrase}"
        for wake in WAKE_WORDS
        for phrase in PHRASE_TO_COMMAND
    ]
    if stop_without_wake:
        phrases.extend(("stop", "emergency stop"))
    phrases.append("[unk]")
    return phrases


def canonical_command(text: str, stop_without_wake: bool) -> str | None:
    words = text.lower().strip().split()
    if not words:
        return None
    if words[0] in WAKE_WORDS:
        phrase = " ".join(words[1:])
        return PHRASE_TO_COMMAND.get(phrase)
    phrase = " ".join(words)
    if stop_without_wake and phrase in {"stop", "emergency stop"}:
        return "stop"
    return None


def result_confidence(result: dict) -> float | None:
    values = [
        float(word["conf"])
        for word in result.get("result", [])
        if "conf" in word
    ]
    return round(sum(values) / len(values), 3) if values else None


def apply_gain(chunk: bytes, gain: float) -> bytes:
    if gain == 1.0:
        return chunk
    samples = array("h")
    samples.frombytes(chunk)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, value in enumerate(samples):
        samples[index] = max(-32768, min(32767, round(value * gain)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 1.5,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def daemon_json(host: str, port: int, payload: dict) -> dict:
    with socket.create_connection((host, port), timeout=1.5) as connection:
        connection.settimeout(3.0)
        connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        response = connection.recv(8192).decode("utf-8").strip()
    return json.loads(response) if response else {"ok": False, "error": "empty reply"}


def run_local_action(args: argparse.Namespace, command: str) -> tuple[str | None, dict]:
    if command == "stop":
        daemon_command = {"cmd": "halt", "speed": 40}
    elif command in {"lie_down", "stop_and_lie_down"}:
        daemon_command = {"cmd": "lie_down", "speed": 40}
    else:
        return None, {"ok": True, "not_local": True}

    try:
        result = http_json(
            "POST",
            f"{args.bridge.rstrip('/')}/command",
            daemon_command,
        )
        return daemon_command["cmd"], result
    except Exception as bridge_error:
        print(
            f"[voice] Bridge unavailable for {command}: {bridge_error}; "
            "using direct daemon fallback",
            flush=True,
        )
        try:
            result = daemon_json(args.daemon_host, args.daemon_port, daemon_command)
            return daemon_command["cmd"], result
        except Exception as daemon_error:
            return daemon_command["cmd"], {
                "ok": False,
                "error": f"bridge and daemon failed: {daemon_error}",
            }


def relay_command(
    args: argparse.Namespace,
    text: str,
    command: str,
    confidence: float | None,
    local_action: str | None,
    local_result: dict,
) -> bool:
    payload = {
        "text": text,
        "source": "pidog_vosk",
        "command": command,
        "local_action": local_action,
        "local_ok": bool(local_result.get("ok")),
    }
    if confidence is not None:
        payload["confidence"] = confidence
    try:
        response = http_json(
            "POST",
            f"{args.bridge.rstrip('/')}/voice/input",
            payload,
        )
        return bool(response.get("ok"))
    except Exception as exc:
        print(f"[voice] Could not relay {command} to desktop inbox: {exc}", flush=True)
        return False


def echo_suppressed(args: argparse.Namespace) -> bool:
    try:
        status = http_json(
            "GET",
            f"{args.bridge.rstrip('/')}/voice/echo_until",
            timeout=0.5,
        )
        return time.time() < float(status.get("echo_until", 0))
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    if not 0.25 <= args.gain <= 20.0:
        raise ValueError("--gain must be between 0.25 and 20")
    if args.debounce_seconds < 0.5:
        raise ValueError("--debounce-seconds must be at least 0.5")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("--minimum-confidence must be between 0 and 1")
    if not 0.0 <= args.bare_stop_minimum_confidence <= 1.0:
        raise ValueError(
            "--bare-stop-minimum-confidence must be between 0 and 1"
        )

    model_path = discover_model(args.model)
    SetLogLevel(-1)
    print(f"[voice] Loading Vosk model: {model_path}", flush=True)
    model = Model(str(model_path))
    grammar = grammar_phrases(args.stop_without_wake)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(grammar))
    recognizer.SetWords(True)

    arecord_command = [
        "arecord",
        "-q",
        "-D",
        args.device,
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        "1",
    ]
    print(
        f"[voice] Listening on {args.device}; say 'Nox' followed by a command",
        flush=True,
    )
    if args.stop_without_wake:
        print("[voice] Bare 'stop' is enabled as a fail-safe", flush=True)

    process = subprocess.Popen(arecord_command, stdout=subprocess.PIPE)
    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    last_command = ""
    last_command_time = 0.0
    last_echo_check = 0.0
    suppress_echo = False

    try:
        while not stopping:
            if process.stdout is None:
                raise RuntimeError("arecord stdout pipe was not created")
            chunk = process.stdout.read(8000)
            if not chunk:
                return_code = process.poll()
                raise RuntimeError(f"arecord stopped unexpectedly ({return_code})")

            now = time.monotonic()
            if now - last_echo_check >= 0.5:
                suppress_echo = echo_suppressed(args)
                last_echo_check = now
            if suppress_echo:
                recognizer.Reset()
                continue

            chunk = apply_gain(chunk, args.gain)
            if not recognizer.AcceptWaveform(chunk):
                continue

            result = json.loads(recognizer.Result())
            text = " ".join(result.get("text", "").split())
            command = canonical_command(text, args.stop_without_wake)
            if command is None:
                # Vosk often finalises a trailing fragment such as "head"
                # separately from "Nox arm head". These fragments have no
                # command authority and do not need to flood the journal.
                harmless_fragments = {
                    "nox", "knox", "knocks", "head", "arm", "track", "me"
                }
                if text and text != "[unk]" and text not in harmless_fragments:
                    print(f"[voice] Ignored: {text!r}", flush=True)
                continue

            confidence = result_confidence(result)
            words = text.lower().strip().split()
            has_wake_word = bool(words and words[0] in WAKE_WORDS)
            minimum_confidence = (
                args.minimum_confidence
                if has_wake_word
                else args.bare_stop_minimum_confidence
            )
            if confidence is None or confidence < minimum_confidence:
                print(
                    f"[voice] REJECTED low-confidence command: text={text!r} "
                    f"command={command} confidence={confidence} "
                    f"required={minimum_confidence:.2f}",
                    flush=True,
                )
                continue

            now = time.monotonic()
            if command == last_command and now - last_command_time < args.debounce_seconds:
                print(f"[voice] Debounced duplicate: {command}", flush=True)
                continue
            last_command = command
            last_command_time = now

            local_action, local_result = run_local_action(args, command)
            relayed = relay_command(
                args,
                text,
                command,
                confidence,
                local_action,
                local_result,
            )
            print(
                f"[voice] ACCEPTED {command}: text={text!r} "
                f"confidence={confidence} local_ok={local_result.get('ok')} "
                f"desktop_relay={relayed}",
                flush=True,
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[voice] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
