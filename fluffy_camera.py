"""Capture a fresh still from the desktop's existing camera stream."""
import base64
import time

import cv2


def capture_camera_still(camera, timeout=2.0):
    """Wait for a new frame, then return an in-memory JPEG data URL."""
    initial_sequence = camera.latest()[0]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sequence, frame, error = camera.latest()
        if sequence > initial_sequence and frame is not None and not error:
            height, width = frame.shape[:2]
            if max(height, width) > 1280:
                scale = 1280 / max(height, width)
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Camera still JPEG encoding failed")
            return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
        time.sleep(0.02)
    raise RuntimeError("No fresh camera frame arrived within two seconds")
