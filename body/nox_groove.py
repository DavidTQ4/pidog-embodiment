#!/usr/bin/env python3
"""Music-responsive head and tail controller for the PiDog voice service.

Tempo-estimation design adapted from koua29/pidog-groove (MIT License):
https://github.com/koua29/pidog-groove

Copyright (c) 2026 koua29
Copyright (c) 2026 David Hallett

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the conditions in the MIT License.

This module does not open ALSA and does not instantiate Pidog.  nox_voice owns
microphone capture and feeds PCM here; nox_daemon remains the sole servo owner.
"""

from __future__ import annotations

import math
import os
import threading
import time
from array import array
from collections import deque
from typing import Callable


SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class GrooveController:
    """Estimate musical pulse from PCM energy and animate head/tail safely."""

    def __init__(
        self,
        send_command: Callable[[dict], dict],
        *,
        analysis_seconds: float | None = None,
        confidence_min: float | None = None,
        max_seconds: float | None = None,
    ):
        self._send = send_command
        self.analysis_seconds = analysis_seconds or float(
            os.environ.get("NOX_GROOVE_ANALYSIS_SECONDS", "8")
        )
        self.confidence_min = confidence_min or float(
            os.environ.get("NOX_GROOVE_CONFIDENCE_MIN", "0.45")
        )
        self.max_seconds = max_seconds or float(
            os.environ.get("NOX_GROOVE_MAX_SECONDS", "120")
        )
        self._energy = deque(
            maxlen=max(100, round(self.analysis_seconds * 1000 / FRAME_MS))
        )
        self._pcm_remainder = b""
        self._lock = threading.RLock()
        self._active = False
        self._started = 0.0
        self._listen_after = 0.0
        self._next_analysis = 0.0
        self._bpm: float | None = None
        self._confidence = 0.0
        self._beat_period = 0.5
        self._dance_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self, acknowledgement_seconds: float = 1.0) -> dict:
        """Arm one bounded dance session; music analysis begins after the bark."""
        self.stop(centre=False)
        now = time.monotonic()
        with self._lock:
            self._active = True
            self._started = now
            self._listen_after = now + max(0.0, acknowledgement_seconds)
            self._next_analysis = self._listen_after + self.analysis_seconds
            self._energy.clear()
            self._pcm_remainder = b""
            self._bpm = None
            self._confidence = 0.0
            self._beat_period = 0.5
            self._stop_event.clear()
            self._dance_thread = threading.Thread(
                target=self._dance_loop,
                name="nox-groove",
                daemon=True,
            )
            self._dance_thread.start()
        print(
            f"[groove] ARMED: listening for {self.analysis_seconds:.1f}s of music "
            f"(confidence >= {self.confidence_min:.2f})",
            flush=True,
        )
        return {
            "ok": True,
            "groove": "armed",
            "analysis_seconds": self.analysis_seconds,
            "max_seconds": self.max_seconds,
        }

    def stop(self, *, centre: bool = True) -> dict:
        with self._lock:
            was_active = self._active
            self._active = False
            self._stop_event.set()
            self._bpm = None
            self._energy.clear()
            self._pcm_remainder = b""
        if was_active and centre:
            try:
                self._send({"cmd": "groove_stop"})
            except Exception as exc:
                print(f"[groove] neutral pose failed: {exc}", flush=True)
        if was_active:
            print("[groove] STOPPED", flush=True)
        return {"ok": True, "groove": "stopped", "was_active": was_active}

    def feed(self, pcm_s16le: bytes) -> None:
        """Accept microphone PCM without blocking the Vosk capture loop."""
        now = time.monotonic()
        with self._lock:
            if not self._active:
                return
            if now - self._started >= self.max_seconds:
                print("[groove] maximum dance time reached", flush=True)
                self.stop()
                return
            if now < self._listen_after:
                return

            data = self._pcm_remainder + pcm_s16le
            frame_bytes = FRAME_SAMPLES * 2
            usable = len(data) - (len(data) % frame_bytes)
            self._pcm_remainder = data[usable:]
            for offset in range(0, usable, frame_bytes):
                samples = array("h")
                samples.frombytes(data[offset : offset + frame_bytes])
                if not samples:
                    continue
                # RMS envelope is intentionally cheap enough for a Pi 4.
                square_mean = sum(float(v) * float(v) for v in samples) / len(samples)
                self._energy.append(math.sqrt(square_mean) / 32768.0)

            if now >= self._next_analysis and len(self._energy) == self._energy.maxlen:
                bpm, confidence = self._estimate_tempo(tuple(self._energy))
                self._confidence = confidence
                if bpm is not None and confidence >= self.confidence_min:
                    self._bpm = bpm
                    self._beat_period = 60.0 / bpm
                    print(
                        f"[groove] music locked: {bpm:.1f} BPM "
                        f"confidence={confidence:.2f}",
                        flush=True,
                    )
                else:
                    self._bpm = None
                    print(
                        f"[groove] no stable musical pulse "
                        f"(confidence={confidence:.2f}); still listening",
                        flush=True,
                    )
                # Re-estimate every four seconds using the rolling window.
                self._next_analysis = now + 4.0

    @staticmethod
    def _estimate_tempo(energy: tuple[float, ...]) -> tuple[float | None, float]:
        if len(energy) < 100:
            return None, 0.0

        # Positive energy differences approximate onset strength. Removing the
        # baseline makes steady fan noise and room ambience contribute little.
        flux = [0.0]
        for previous, current in zip(energy, energy[1:]):
            flux.append(max(0.0, current - previous))
        mean_flux = sum(flux) / len(flux)
        envelope = [max(0.0, value - mean_flux) for value in flux]
        power = sum(value * value for value in envelope)
        peak_count = sum(
            1
            for index in range(1, len(envelope) - 1)
            if envelope[index] > envelope[index - 1]
            and envelope[index] >= envelope[index + 1]
            and envelope[index] > mean_flux * 1.5
        )
        if power < 1e-8 or peak_count < 6:
            return None, 0.0

        frames_per_second = 1000.0 / FRAME_MS
        minimum_lag = max(1, round(frames_per_second * 60.0 / 180.0))
        maximum_lag = round(frames_per_second * 60.0 / 60.0)
        best_lag = 0
        best_score = 0.0
        for lag in range(minimum_lag, maximum_lag + 1):
            numerator = sum(
                envelope[index] * envelope[index - lag]
                for index in range(lag, len(envelope))
            )
            left = sum(value * value for value in envelope[lag:])
            right = sum(value * value for value in envelope[:-lag])
            denominator = math.sqrt(left * right)
            score = numerator / denominator if denominator > 1e-12 else 0.0
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag == 0:
            return None, 0.0
        bpm = 60.0 * frames_per_second / best_lag
        return bpm, round(best_score, 3)

    def _dance_loop(self) -> None:
        phase = 1
        next_pose = time.monotonic()
        while not self._stop_event.wait(0.02):
            with self._lock:
                if not self._active:
                    return
                bpm = self._bpm
                beat_period = self._beat_period
            if bpm is None:
                continue

            now = time.monotonic()
            if now < next_pose:
                continue
            try:
                result = self._send(
                    {
                        "cmd": "groove_pose",
                        "yaw": 30.0 * phase,
                        "roll": -16.0 * phase,
                        "pitch": 7.0 if phase > 0 else -5.0,
                        "tail": 38.0 * phase,
                    }
                )
                if not result.get("ok"):
                    print(f"[groove] pose rejected: {result}", flush=True)
            except Exception as exc:
                print(f"[groove] pose command failed: {exc}", flush=True)
            phase *= -1
            # A broad motion every two beats looks deliberate and limits servo
            # heat while leaving the voice thread responsive.
            next_pose = now + max(0.28, beat_period * 2.0)
