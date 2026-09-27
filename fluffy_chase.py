"""Red-target tracking through Fluffy's existing camera and motion broker."""
import cv2
import numpy as np


def red_target(frame):
    """Return one unambiguous red blob in a normalized 640x480 image."""
    image = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = sorted((c for c in contours if cv2.contourArea(c) >= 150), key=cv2.contourArea, reverse=True)
    if not blobs or (len(blobs) > 1 and cv2.contourArea(blobs[1]) > cv2.contourArea(blobs[0]) * 0.5):
        return None
    x, y, w, h = cv2.boundingRect(blobs[0])
    return x + w / 2, y + h / 2, w


class TargetChase:
    def __init__(self, head, move, halt, keep_awake, interval=0.5):
        self.head, self.move, self.halt, self.keep_awake = head, move, halt, keep_awake
        self.interval = interval
        self.active = False
        self.reason = "disarmed"
        self.yaw = self.pitch = 0.0

    def start(self, now):
        self.yaw = self.pitch = 0.0
        self.last_frame = self.last_move = self.last_lease = now
        self.confirmed = 0
        self.active = True
        self.reason = "waiting for stable red target"

    def stop(self, reason):
        was_active = self.active
        self.active = False
        self.reason = reason
        if was_active:
            self.halt()

    def watchdog(self, now):
        if self.active and now - self.last_frame > 1.0:
            self.stop("camera stalled")

    def update(self, frame, now):
        if not self.active:
            return
        dt = min(max(now - self.last_frame, 0), 0.1)
        self.last_frame = now
        target = red_target(frame)
        if target is None:
            self.stop("red target lost or ambiguous")
            return
        x, y, width = target
        if width >= 300:
            self.stop("target reached")
            return
        self.confirmed += 1
        # Original example: 0.5 degrees every 20ms, scaled to camera rate.
        step = 25 * dt
        self.yaw = max(-80, min(80, self.yaw + (step if x < 305 else -step if x > 335 else 0)))
        self.pitch = max(-40, min(20, self.pitch + (step if y < 215 else -step if y > 265 else 0)))
        if not self.head(self.yaw, self.pitch):
            self.stop("head command failed")
            return
        if now - self.last_lease >= 2:
            if not self.keep_awake():
                self.stop("Pi activity lease failed")
                return
            self.last_lease = now
        if self.confirmed < 3 or now - self.last_move < self.interval:
            return
        action = "turn_right" if self.yaw < -30 else "turn_left" if self.yaw > 30 else None
        if action is None and abs(x - 320) <= 40:
            action = "forward"
        if action is None:
            return
        self.last_move = now
        result, _, reason = self.move(action)
        if result not in {"accepted", "busy"}:
            self.stop(reason or "movement rejected")
        else:
            self.reason = "chasing red target"
