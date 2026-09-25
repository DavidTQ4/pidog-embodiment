"""Guided, local face enrolment for PiDog's owner-recognition profile.

Only aligned face crops and SFace embeddings are stored. Full camera frames are
not written to disk. The PiDog head follows the single visible face when armed.

Controls:
  C       Centre head and establish its reference position
  M       Arm/disarm face-following head movement
  S       Start/pause enrolment capture
  F       Finish early (requires at least 10 accepted samples)
  Q/Esc   Quit
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from pidog_face_identity import (
    DEFAULT_FACE_THRESHOLD,
    FaceIdentityEngine,
    FaceObservation,
    create_profile,
    save_profile,
)


DEFAULT_STREAM = "http://127.0.0.1:19000/mjpg"
DEFAULT_ROBOT_API = "http://127.0.0.1:18888"
DEFAULT_PROFILE = "face_profiles/Joss.npz"
DEFAULT_IMAGES = "face_profiles/Joss_images"

YAW_LIMITS = (-80.0, 80.0)
PITCH_LIMITS = (-30.0, 30.0)
X_DEADBAND = 0.08
Y_DEADBAND = 0.08
YAW_GAIN = 7.0
PITCH_GAIN = 8.0
MAX_STEP_DEGREES = 3.0
COMMAND_INTERVAL = 0.35
CAPTURE_INTERVAL = 0.35
MIN_FACE_PIXELS = 80
MIN_BLUR_VARIANCE = 55.0
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 220.0
MAX_DUPLICATE_SIMILARITY = 0.9985

POSE_GUIDANCE = (
    "Look straight at PiDog",
    "Slowly turn your face left",
    "Slowly turn your face right",
    "Tilt your face slightly upward",
    "Tilt your face slightly downward",
    "Vary expression and distance slightly",
)


class LatestFrameCamera:
    def __init__(self, url: str):
        self.url = url
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.error: str | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def latest(self) -> tuple[int, np.ndarray | None, str | None]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return self.sequence, frame, self.error

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        while not self.stop_event.is_set():
            capture = self._open()
            if not capture.isOpened():
                with self.lock:
                    self.error = f"Could not open stream: {self.url}"
                time.sleep(1)
                continue
            with self.lock:
                self.error = None
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    with self.lock:
                        self.error = "Stream stopped; reconnecting"
                    break
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.error = None
            capture.release()
            if not self.stop_event.is_set():
                time.sleep(0.5)


def command_head(
    session: requests.Session,
    robot_api: str,
    yaw: float,
    pitch: float,
) -> bool:
    try:
        response = session.post(
            f"{robot_api.rstrip('/')}/head",
            json={"yaw": yaw, "roll": 0, "pitch": pitch},
            timeout=3,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            print(f"Head command rejected: {payload}")
            return False
        return True
    except (requests.RequestException, ValueError) as exc:
        print(f"Head command failed: {exc}")
        return False


def face_quality(face: FaceObservation) -> tuple[bool, str, float, float]:
    x1, y1, x2, y2 = face.box
    if min(x2 - x1, y2 - y1) < MIN_FACE_PIXELS:
        return False, "Move closer", 0.0, 0.0
    grey = cv2.cvtColor(face.aligned_face, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())
    if blur < MIN_BLUR_VARIANCE:
        return False, "Hold still - image is blurred", blur, brightness
    if brightness < MIN_BRIGHTNESS:
        return False, "Face is too dark", blur, brightness
    if brightness > MAX_BRIGHTNESS:
        return False, "Face is overexposed", blur, brightness
    return True, "Face quality good", blur, brightness


def guidance_for(sample_count: int, target_samples: int) -> str:
    phase_size = max(1, target_samples // len(POSE_GUIDANCE))
    phase = min(len(POSE_GUIDANCE) - 1, sample_count // phase_size)
    return POSE_GUIDANCE[phase]


def is_diverse_enough(
    embedding: np.ndarray,
    accepted: list[np.ndarray],
) -> bool:
    if not accepted:
        return True
    similarities = np.asarray(accepted) @ embedding
    return float(similarities.max()) < MAX_DUPLICATE_SIMILARITY


def prepare_output(
    profile_path: Path,
    image_directory: Path,
    replace: bool,
) -> None:
    if profile_path.exists() and not replace:
        raise FileExistsError(
            f"{profile_path} already exists. Use --replace only when you "
            "intend to replace the existing biometric profile."
        )
    if (
        image_directory.exists()
        and any(image_directory.glob("*.jpg"))
        and not replace
    ):
        raise FileExistsError(
            f"{image_directory} contains an earlier partial scan. Use "
            "--replace only when you intend to discard and replace it."
        )
    if replace:
        profile_path.unlink(missing_ok=True)
        if image_directory.exists():
            shutil.rmtree(image_directory)
    image_directory.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Joss's PiDog profile")
    parser.add_argument("--name", default="Joss")
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    parser.add_argument("--robot-api", default=DEFAULT_ROBOT_API)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--model-directory", default="face_models")
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_FACE_THRESHOLD,
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 10:
        raise ValueError("--samples must be at least 10")

    profile_path = Path(args.profile)
    image_directory = Path(args.images)
    prepare_output(profile_path, image_directory, args.replace)

    face_engine = FaceIdentityEngine(args.model_directory)
    session = requests.Session()
    camera = LatestFrameCamera(args.stream)
    camera.start()

    accepted_embeddings: list[np.ndarray] = []
    sample_metadata: list[dict[str, float | int]] = []
    movement_enabled = False
    capture_enabled = False
    head_reference_known = False
    profile_saved = False
    yaw = 0.0
    pitch = 0.0
    last_command_time = 0.0
    last_capture_time = 0.0
    last_sequence = -1
    quality_message = "Press C, then M, then S to begin"

    def finish_profile() -> None:
        nonlocal profile_saved, capture_enabled, movement_enabled
        if len(accepted_embeddings) < 10:
            print("At least 10 accepted samples are required")
            return
        profile = create_profile(
            args.name,
            accepted_embeddings,
            threshold=args.threshold,
            metadata={
                "image_directory": str(image_directory),
                "accepted_samples": len(accepted_embeddings),
                "quality": sample_metadata,
                "detector": "OpenCV YuNet",
                "recognizer": "OpenCV SFace",
            },
        )
        save_profile(profile, profile_path)
        profile_saved = True
        capture_enabled = False
        movement_enabled = False
        print(f"Saved {profile.name} profile to {profile_path}")

    print("C: centre | M: arm tracking | S: capture | F: finish | Q: quit")
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            _, frame, stream_error = camera.latest()
            if frame is not None:
                break
            if stream_error:
                print(stream_error)
            time.sleep(0.5)
        else:
            raise RuntimeError("No MJPEG frames received after 15 seconds")

        while True:
            sequence, frame, stream_error = camera.latest()
            if frame is None or sequence == last_sequence:
                time.sleep(0.003)
                continue
            last_sequence = sequence
            observations = face_engine.observe(frame)
            target = (
                max(
                    observations,
                    key=lambda face: (
                        face.box[2] - face.box[0]
                    ) * (face.box[3] - face.box[1]),
                )
                if len(observations) == 1
                else None
            )

            now = time.monotonic()
            if target is not None:
                valid, quality_message, blur, brightness = face_quality(target)
                frame_height, frame_width = frame.shape[:2]
                error_x = (
                    target.centre[0] - frame_width / 2
                ) / max(frame_width / 2, 1)
                error_y = (
                    target.centre[1] - frame_height / 2
                ) / max(frame_height / 2, 1)

                if (
                    movement_enabled
                    and now - last_command_time >= COMMAND_INTERVAL
                ):
                    yaw_step = (
                        float(
                            np.clip(
                                -error_x * YAW_GAIN,
                                -MAX_STEP_DEGREES,
                                MAX_STEP_DEGREES,
                            )
                        )
                        if abs(error_x) >= X_DEADBAND
                        else 0.0
                    )
                    pitch_step = (
                        float(
                            np.clip(
                                -error_y * PITCH_GAIN,
                                -MAX_STEP_DEGREES,
                                MAX_STEP_DEGREES,
                            )
                        )
                        if abs(error_y) >= Y_DEADBAND
                        else 0.0
                    )
                    if yaw_step != 0.0 or pitch_step != 0.0:
                        new_yaw = float(
                            np.clip(yaw + yaw_step, *YAW_LIMITS)
                        )
                        new_pitch = float(
                            np.clip(pitch + pitch_step, *PITCH_LIMITS)
                        )
                        if command_head(
                            session,
                            args.robot_api,
                            new_yaw,
                            new_pitch,
                        ):
                            yaw, pitch = new_yaw, new_pitch
                        else:
                            movement_enabled = False
                    last_command_time = now

                if (
                    capture_enabled
                    and valid
                    and now - last_capture_time >= CAPTURE_INTERVAL
                ):
                    if is_diverse_enough(
                        target.embedding,
                        accepted_embeddings,
                    ):
                        accepted_embeddings.append(target.embedding.copy())
                        sample_number = len(accepted_embeddings)
                        image_path = image_directory / (
                            f"{args.name}_{sample_number:03d}.jpg"
                        )
                        if not cv2.imwrite(str(image_path), target.aligned_face):
                            accepted_embeddings.pop()
                            raise RuntimeError(f"Could not write {image_path}")
                        sample_metadata.append(
                            {
                                "sample": sample_number,
                                "blur": round(blur, 2),
                                "brightness": round(brightness, 2),
                                "detector_score": round(
                                    target.detector_score,
                                    4,
                                ),
                            }
                        )
                        print(
                            f"Accepted face {sample_number}/{args.samples}: "
                            f"{guidance_for(sample_number, args.samples)}"
                        )
                        if sample_number >= args.samples:
                            finish_profile()
                    else:
                        quality_message = "Change head angle slightly"
                    last_capture_time = now
            elif len(observations) > 1:
                quality_message = "Multiple faces visible - capture paused"
            else:
                quality_message = "No face detected"

            for face in observations:
                x1, y1, x2, y2 = face.box
                colour = (0, 255, 0) if face is target else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                for landmark in face.landmarks:
                    cv2.circle(frame, landmark, 2, (255, 0, 255), -1)
            if target is not None:
                cv2.drawMarker(
                    frame,
                    target.centre,
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

            phase_instruction = guidance_for(
                len(accepted_embeddings),
                args.samples,
            )
            lines = (
                f"{args.name} enrolment: {len(accepted_embeddings)}/"
                f"{args.samples}",
                phase_instruction,
                quality_message,
                (
                    f"Tracking {'ARMED' if movement_enabled else 'DISARMED'} "
                    f"| Capture {'ON' if capture_enabled else 'PAUSED'} "
                    f"| yaw={yaw:.1f} pitch={pitch:.1f}"
                ),
                "C centre | M tracking | S capture | F finish | Q quit",
            )
            for index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (10, 28 + index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58 if index == 0 else 0.52,
                    (255, 255, 255),
                    2 if index < 2 else 1,
                    cv2.LINE_AA,
                )
            if stream_error:
                cv2.putText(
                    frame,
                    stream_error,
                    (10, 163),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("PiDog Joss face enrolment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                movement_enabled = False
                if command_head(session, args.robot_api, 0.0, 0.0):
                    yaw = pitch = 0.0
                    head_reference_known = True
                    print("Head centred; movement remains disarmed")
            if key == ord("m"):
                if movement_enabled:
                    movement_enabled = False
                    print("Face following disarmed")
                elif not head_reference_known:
                    print("Press C to centre the head before arming")
                elif target is None:
                    print("Exactly one face must be visible before arming")
                else:
                    movement_enabled = True
                    print("Face following armed")
            if key == ord("s"):
                if profile_saved:
                    print("Profile is already saved; restart with --replace")
                else:
                    capture_enabled = not capture_enabled
                    print(
                        f"Enrolment capture: "
                        f"{'ON' if capture_enabled else 'PAUSED'}"
                    )
            if key == ord("f"):
                finish_profile()
    finally:
        camera.stop()
        session.close()
        cv2.destroyAllWindows()
        if not profile_saved:
            print(
                "No completed profile was saved. Partial aligned images remain "
                f"in {image_directory}."
            )


if __name__ == "__main__":
    main()
