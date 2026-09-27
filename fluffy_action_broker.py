#!/usr/bin/env python3
"""Validated stationary-action broker for Fluffy's desktop LLM.

The language model can propose only a symbolic action name. This broker owns
the allowlist, safety-state checks, daemon payloads, timeouts and audit log.
It intentionally exposes no locomotion, tracking controls, servo angles, URLs
or arbitrary command parameters.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ActionSpec:
    daemon_action: str
    speed: int
    spoken_name: str


ACTION_SPECS: dict[str, ActionSpec] = {
    "bark": ActionSpec("bark", 70, "bark"),
    "wag_tail": ActionSpec("wag_tail", 80, "wag my tail"),
    "nod": ActionSpec("nod", 70, "nod"),
    "shake_head": ActionSpec("shake_head", 70, "shake my head"),
    "think": ActionSpec("think", 70, "make a thinking gesture"),
    "pant": ActionSpec("pant", 70, "pant"),
    "sit": ActionSpec("sit", 60, "sit"),
    "stand": ActionSpec("stand", 60, "stand"),
    "hand_shake": ActionSpec("hand_shake", 70, "offer my paw"),
    "high_five": ActionSpec("high_five", 70, "give a high five"),
    "stretch": ActionSpec("stretch", 70, "stretch"),
    "push_up": ActionSpec("push_up", 70, "do push-ups"),
    "scratch": ActionSpec("scratch", 70, "scratch"),
    "howling": ActionSpec("howling", 70, "howl"),
    "relax_neck": ActionSpec("relax_neck", 70, "relax my neck"),
    "surprise": ActionSpec("surprise", 70, "look surprised"),
}

ALIASES = {
    "wag tail": "wag_tail",
    "shake head": "shake_head",
    "paw": "hand_shake",
    "handshake": "hand_shake",
    "high five": "high_five",
    "push up": "push_up",
    "push ups": "push_up",
    "howl": "howling",
    "relax neck": "relax_neck",
}


def normalise_action_name(value: object) -> str:
    name = re.sub(r"[^a-z0-9_ ]+", "", str(value).lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return ALIASES.get(name, name.replace(" ", "_"))


class FluffyActionBroker:
    """Execute at most two allowlisted actions during one LLM turn."""

    MAX_ACTIONS_PER_TURN = 2

    def __init__(
        self,
        robot_api: str,
        *,
        session: requests.Session | None = None,
        request_timeout: tuple[float, float] = (1.0, 8.0),
        minimum_interval_seconds: float = 0.35,
    ):
        self.robot_api = robot_api.rstrip("/")
        self.session = session or requests.Session()
        self.request_timeout = request_timeout
        self.minimum_interval_seconds = minimum_interval_seconds
        self._lock = threading.Lock()
        self._last_action_at = 0.0

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return tuple(ACTION_SPECS)

    def execute(
        self,
        requested_action: object,
        robot_context: dict[str, object],
    ) -> dict[str, Any]:
        """Validate state and submit one fixed daemon command."""
        action = normalise_action_name(requested_action)
        spec = ACTION_SPECS.get(action)
        if spec is None:
            result = {
                "ok": False,
                "executed": False,
                "action": action,
                "reason": "action is not in the stationary LLM allowlist",
                "allowed_actions": list(self.allowed_actions),
            }
            self._log(result)
            return result

        if bool(robot_context.get("head_tracking_armed")):
            result = {
                "ok": False,
                "executed": False,
                "action": action,
                "reason": "head tracking is armed; disarm it before conversational gestures",
            }
            self._log(result)
            return result
        if bool(robot_context.get("body_following_armed")):
            result = {
                "ok": False,
                "executed": False,
                "action": action,
                "reason": "body following is armed; disarm it before conversational gestures",
            }
            self._log(result)
            return result

        payload = {
            "cmd": "move_if_idle",
            "action": spec.daemon_action,
            "steps": 1,
            "speed": spec.speed,
        }
        with self._lock:
            delay = self.minimum_interval_seconds - (
                time.monotonic() - self._last_action_at
            )
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.session.post(
                    f"{self.robot_api}/command",
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                body = response.json()
            except Exception as exc:
                result = {
                    "ok": False,
                    "executed": False,
                    "action": action,
                    "reason": (
                        f"robot command failed: {type(exc).__name__}: {exc}"
                    ),
                }
                self._log(result)
                return result
            finally:
                self._last_action_at = time.monotonic()

        accepted = bool(body.get("ok")) and bool(body.get("accepted", True))
        result = {
            "ok": accepted,
            "executed": accepted,
            "action": action,
            "description": spec.spoken_name,
            "robot_reply": body,
        }
        if not accepted:
            result["reason"] = (
                body.get("reason")
                or body.get("error")
                or ("robot is busy" if body.get("busy") else "robot rejected action")
            )
        self._log(result)
        return result

    @staticmethod
    def _log(result: dict[str, Any]) -> None:
        status = "EXECUTED" if result.get("executed") else "REJECTED"
        detail = result.get("reason") or result.get("description") or ""
        print(
            f"[LLM ACTION] {status} {result.get('action')}: {detail}",
            flush=True,
        )
