"""Real-time YOLO tracking plus asynchronous Qwen3-VL scene reasoning.

All visual processing runs on the desktop GPU. The Raspberry Pi provides the
MJPEG stream and accepts deliberately bounded head-position commands. Movement
starts disarmed and selecting a person never arms it.

Default camera transport:
  tcp://127.0.0.1:19001 -> Pi 127.0.0.1:9001 (raw hardware H.264)

Explicit snapshot/MJPEG fallback:
  --stream http://127.0.0.1:19000/mjpg.jpg

Controls:
  V / Space  Ask Qwen about the newest clean frame and current YOLO tracks
  Y          Enable/disable continuous YOLO processing
  1..9       Select a visible person, ordered left-to-right; lock to their
             recognised identity when one is available
  0          Clear the selected person and disarm movement
  C          Centre the head and establish a known position
  M          Arm/disarm bounded head tracking (starts disarmed)
  T          Arm/disarm identity-locked following (turn and walk forward)
  A          Toggle centre/upper-body vertical aiming point
  H          Enable/disable facial-keypoint head aiming
  Q / Esc    Quit
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile

try:
    import av
except ImportError:
    av = None
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    pipeline,
)
from ultralytics import YOLO

from pidog_face_identity import (
    DEFAULT_IDENTITY_MARGIN,
    FaceIdentityEngine,
    FaceObservation,
    FaceProfile,
    face_inside_person,
    identify_embedding,
    load_profiles,
)


DEFAULT_STREAM = "tcp://127.0.0.1:19001"
DEFAULT_ROBOT_API = "http://127.0.0.1:18888"
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_YOLO_MODEL = "yolo11n.pt"
DEFAULT_POSE_MODEL = "yolo11n-pose.pt"
DEFAULT_PROMPT = (
    "You are Fluffy, a robot dog. Describe what you can currently see from "
    "your own first-person viewpoint, using natural spoken English. Use the "
    "supplied YOLO tracks as fallible hints and check them against the image. "
    "Mention the most relevant people, objects, activity, and any immediate "
    "obstacles or hazards. If useful, identify an operator-selected person by "
    "name, but do not speak YOLO track IDs or technical metadata. Do not issue "
    "movement commands. Respond with plain text in two to four concise "
    "sentences suitable for speaking aloud."
)

# Conservative person-tracking controller. The VLM never supplies angles.
PERSON_ARM_CONFIDENCE = 0.50
PERSON_CONFIRM_FRAMES = 5
REACQUIRE_DELAY_SECONDS = 0.75
REACQUIRE_CONFIRM_FRAMES = 5
REACQUIRE_MAX_POSITION_DISTANCE = 0.55
REACQUIRE_MIN_AREA_RATIO = 0.35
REACQUIRE_MAX_AREA_RATIO = 2.85
X_DEADBAND = 0.10
Y_DEADBAND = 0.10
YAW_GAIN = 15.0
PITCH_GAIN = 8.0
MAX_YAW_STEP_DEGREES = 7.0
MAX_PITCH_STEP_DEGREES = 3.0
# SunFounder's face-tracking example uses these working limits.
YAW_LIMITS = (-80.0, 80.0)
PITCH_LIMITS = (-30.0, 30.0)
COMMAND_INTERVAL = 0.10
DEFAULT_TURN_YAW_THRESHOLD = 18.0
DEFAULT_BODY_TURN_SCREEN_THRESHOLD = 0.18
DEFAULT_TURN_COMMAND_INTERVAL = 0.75
DEFAULT_FOLLOW_DISTANCE_CM = 55.0
DEFAULT_TURN_CLEARANCE_CM = 20.0
BODY_KEEP_AWAKE_INTERVAL = 20.0
CENTRE_AIM_FRACTION = 0.50
UPPER_BODY_AIM_FRACTION = 0.20
FACE_KEYPOINT_CONFIDENCE = 0.35
MIN_POSE_BOX_IOU = 0.25


class LatestFrameCamera:
    """Continuously decode HTTP MJPEG while retaining only the newest frame.

    OpenCV/FFmpeg's ``VideoCapture`` can build a substantial internal buffer on
    higher-latency links. Reading the multipart byte stream directly keeps the
    camera thread aligned with arrived frames and avoids processing a stale
    queue.
    """

    JPEG_START = b"\xff\xd8"
    JPEG_END = b"\xff\xd9"
    STREAM_CHUNK_BYTES = 16 * 1024
    MAX_BUFFER_BYTES = 8 * 1024 * 1024

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

    def _run_snapshot(self) -> None:
        """Repeatedly request ViLib's current JPEG without building a backlog."""
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                try:
                    response = session.get(
                        self.url,
                        headers={"Cache-Control": "no-cache"},
                        timeout=(5, 5),
                    )
                    response.raise_for_status()
                    frame = cv2.imdecode(
                        np.frombuffer(response.content, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None:
                        raise ValueError("camera returned an invalid JPEG")
                    with self.lock:
                        self.frame = frame
                        self.sequence += 1
                        self.error = None
                except (requests.RequestException, ValueError) as exc:
                    with self.lock:
                        self.error = f"Latest-frame request failed: {exc}"
                    self.stop_event.wait(0.2)
        finally:
            session.close()

    def _run_h264(self) -> None:
        """Decode a low-latency raw H.264 TCP stream, retaining one frame."""
        if av is None:
            with self.lock:
                self.error = (
                    "H.264 mode requires PyAV; install it with: python -m pip install av"
                )
            return

        while not self.stop_event.is_set():
            container = None
            try:
                container = av.open(
                    self.url,
                    format="h264",
                    mode="r",
                    options={
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                        "probesize": "32",
                        "analyzeduration": "0",
                    },
                    timeout=(5.0, 5.0),
                )
                with self.lock:
                    self.error = None
                for decoded in container.decode(video=0):
                    if self.stop_event.is_set():
                        break
                    frame = decoded.to_ndarray(format="bgr24")
                    with self.lock:
                        self.frame = frame
                        self.sequence += 1
                        self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = f"H.264 stream interrupted: {exc}"
                self.stop_event.wait(0.5)
            finally:
                if container is not None:
                    container.close()

    def _run(self) -> None:
        if self.url.lower().startswith("tcp://"):
            self._run_h264()
            return

        if self.url.lower().split("?", 1)[0].endswith((".jpg", ".jpeg")):
            self._run_snapshot()
            return

        while not self.stop_event.is_set():
            try:
                response = requests.get(
                    self.url,
                    stream=True,
                    timeout=(5, 10),
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                with self.lock:
                    self.error = f"Could not open stream: {exc}"
                time.sleep(1)
                continue

            with self.lock:
                self.error = None

            buffer = bytearray()
            try:
                for chunk in response.iter_content(self.STREAM_CHUNK_BYTES):
                    if self.stop_event.is_set():
                        break
                    if not chunk:
                        continue

                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(self.JPEG_START)
                        if start < 0:
                            # Retain one byte in case a JPEG marker is split
                            # across adjacent HTTP chunks.
                            if len(buffer) > 1:
                                del buffer[:-1]
                            break

                        end = buffer.find(self.JPEG_END, start + 2)
                        if end < 0:
                            if start:
                                del buffer[:start]
                            if len(buffer) > self.MAX_BUFFER_BYTES:
                                buffer.clear()
                                with self.lock:
                                    self.error = (
                                        "Oversized/incomplete MJPEG frame; "
                                        "resynchronising"
                                    )
                            break

                        jpeg = bytes(buffer[start : end + 2])
                        del buffer[: end + 2]
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if frame is None:
                            continue
                        with self.lock:
                            self.frame = frame
                            self.sequence += 1
                            self.error = None
            except requests.RequestException as exc:
                with self.lock:
                    self.error = f"Stream interrupted ({exc}); reconnecting"
            finally:
                response.close()

            if not self.stop_event.is_set() and self.error is None:
                with self.lock:
                    self.error = "Stream stopped; reconnecting"
            if not self.stop_event.is_set():
                time.sleep(0.5)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    track_id: int | None
    box: tuple[int, int, int, int]
    centre: tuple[int, int]
    horizontal: str
    vertical: str
    distance_hint: str

    def prompt_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 2),
            "position": f"{self.vertical}-{self.horizontal}",
            "distance_hint": self.distance_hint,
        }


@dataclass
class AnalysisState:
    running: bool = False
    text: str = "Press V or Space for VLM scene analysis"
    seconds: float | None = None
    error: str | None = None
    detections_used: int = 0


@dataclass
class YoloState:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class HeadPose:
    body_box: tuple[int, int, int, int]
    head_point: tuple[int, int]
    face_points: tuple[tuple[int, int], ...]


@dataclass
class PoseState:
    heads: list[HeadPose] = field(default_factory=list)
    inference_ms: float = 0.0
    error: str | None = None


class FollowStateLogger:
    """Concise state-change logging for eyes-off-screen robot testing."""

    def __init__(self, repeat_seconds: float = 3.0):
        self.repeat_seconds = repeat_seconds
        self.last_key: tuple[str, str] | None = None
        self.last_print_time = 0.0

    def report(
        self,
        mode: str,
        reason: str,
        *,
        identity: str | None,
        track_id: int | None,
        yaw: float,
        screen_error: float | None,
        distance_cm: float | None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        key = (mode, reason)
        if (
            not force
            and key == self.last_key
            and now - self.last_print_time < self.repeat_seconds
        ):
            return
        target = identity or "unrecognised"
        track = "none" if track_id is None else str(track_id)
        error = "unknown" if screen_error is None else f"{screen_error:+.2f}"
        distance = (
            "unknown" if distance_cm is None else f"{distance_cm:.1f}cm"
        )
        print(
            f"[FOLLOW] {mode} | {reason} | target={target} "
            f"track={track} | screen={error} yaw={yaw:+.1f}deg | "
            f"ultrasonic={distance}"
        )
        self.last_key = key
        self.last_print_time = now


class YoloTracker:
    """Run persistent ByteTrack tracking and convert results to simple records."""

    def __init__(
        self,
        model_name: str,
        confidence: float,
        image_size: int,
    ):
        print(f"Loading YOLO detector: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size
        self.names = self.model.names
        print("YOLO ready")

    def process(self, frame: np.ndarray) -> YoloState:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        try:
            result = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                imgsz=self.image_size,
                device=0,
                quantize=16,
                max_det=30,
                verbose=False,
            )[0]

            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                coordinates = boxes.xyxy.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                ids = (
                    boxes.id.detach().cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [None] * len(coordinates)
                )

                for xyxy, class_index, score, track_id in zip(
                    coordinates,
                    classes,
                    confidences,
                    ids,
                ):
                    x1, y1, x2, y2 = (int(value) for value in xyxy)
                    centre_x = (x1 + x2) // 2
                    centre_y = (y1 + y2) // 2
                    horizontal = relative_axis(centre_x / max(width, 1))
                    vertical = relative_vertical(centre_y / max(height, 1))
                    area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / max(
                        width * height,
                        1,
                    )
                    distance_hint = area_to_distance(area_fraction)
                    class_name = (
                        self.names[class_index]
                        if isinstance(self.names, dict)
                        else self.names[class_index]
                    )
                    detections.append(
                        Detection(
                            class_name=str(class_name),
                            confidence=float(score),
                            track_id=(
                                None if track_id is None else int(track_id)
                            ),
                            box=(x1, y1, x2, y2),
                            centre=(centre_x, centre_y),
                            horizontal=horizontal,
                            vertical=vertical,
                            distance_hint=distance_hint,
                        )
                    )

            elapsed_ms = (time.perf_counter() - started) * 1000
            return YoloState(detections=detections, inference_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return YoloState(
                inference_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
            )


class PoseEstimator:
    """Locate facial COCO pose keypoints while object YOLO keeps track IDs."""

    def __init__(
        self,
        model_name: str,
        confidence: float,
        image_size: int,
    ):
        print(f"Loading YOLO pose model: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size
        print("YOLO pose estimator ready")

    def process(self, frame: np.ndarray) -> PoseState:
        started = time.perf_counter()
        try:
            result = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=0,
                quantize=16,
                max_det=20,
                verbose=False,
            )[0]
            boxes = result.boxes
            keypoints = result.keypoints
            heads: list[HeadPose] = []
            if (
                boxes is not None
                and keypoints is not None
                and len(boxes) > 0
            ):
                coordinates = boxes.xyxy.detach().cpu().numpy()
                points_xy = keypoints.xy.detach().cpu().numpy()
                points_conf = (
                    keypoints.conf.detach().cpu().numpy()
                    if keypoints.conf is not None
                    else np.ones(points_xy.shape[:2], dtype=np.float32)
                )

                for box, person_xy, person_conf in zip(
                    coordinates,
                    points_xy,
                    points_conf,
                ):
                    # COCO indices 0..4 are nose, left/right eye and ear.
                    visible_face_points: list[tuple[int, int]] = []
                    visible_weights: list[float] = []
                    for point, score in zip(
                        person_xy[:5],
                        person_conf[:5],
                    ):
                        x, y = float(point[0]), float(point[1])
                        if (
                            float(score) >= FACE_KEYPOINT_CONFIDENCE
                            and x > 0
                            and y > 0
                        ):
                            visible_face_points.append((int(x), int(y)))
                            visible_weights.append(float(score))

                    if not visible_face_points:
                        continue

                    point_array = np.asarray(
                        visible_face_points,
                        dtype=np.float32,
                    )
                    head_xy = np.average(
                        point_array,
                        axis=0,
                        weights=np.asarray(visible_weights),
                    )
                    x1, y1, x2, y2 = (int(value) for value in box)
                    heads.append(
                        HeadPose(
                            body_box=(x1, y1, x2, y2),
                            head_point=(int(head_xy[0]), int(head_xy[1])),
                            face_points=tuple(visible_face_points),
                        )
                    )

            return PoseState(
                heads=heads,
                inference_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return PoseState(
                inference_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )


class VLMObserver:
    """Own Qwen and run no more than one scene analysis concurrently."""

    def __init__(self, model_id: str, max_new_tokens: int):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot see CUDA. Confirm that the CUDA-enabled build "
                "is installed in this virtual environment."
            )

        self.max_new_tokens = max_new_tokens
        self.state = AnalysisState()
        self.lock = threading.Lock()
        self.conversation_history: deque[dict[str, str]] = deque(maxlen=12)
        self.transcriber = None
        self.last_scene_text = ""
        self.last_scene_time = 0.0

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Loading VLM: {model_id}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Qwen VLM ready")

    def snapshot(self) -> AnalysisState:
        with self.lock:
            return AnalysisState(
                running=self.state.running,
                text=self.state.text,
                seconds=self.state.seconds,
                error=self.state.error,
                detections_used=self.state.detections_used,
            )

    def submit(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        prompt: str,
        on_complete=None,
    ) -> bool:
        with self.lock:
            if self.state.running:
                return False
            self.state.running = True
            self.state.error = None
            self.state.text = "Analysing image with YOLO context..."
            self.state.seconds = None
            self.state.detections_used = len(detections)

        threading.Thread(
            target=self._analyse,
            args=(frame.copy(), list(detections), prompt, on_complete),
            daemon=True,
        ).start()
        return True

    def _analyse(
        self,
        bgr_frame: np.ndarray,
        detections: list[Detection],
        prompt: str,
        on_complete=None,
    ) -> None:
        started = time.perf_counter()
        try:
            detection_payload = [item.prompt_record() for item in detections]
            combined_prompt = (
                f"{prompt}\n\n"
                "YOLO detections from this exact frame:\n"
                f"{json.dumps(detection_payload, ensure_ascii=False)}\n"
                "The distance values are box-size estimates, not measured depth."
            )

            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": combined_prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            generated_ids = output_ids[:, input_length:]
            result = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            seconds = time.perf_counter() - started
            with self.lock:
                self.state.text = result or "Model returned an empty response"
                self.state.seconds = seconds
                self.state.running = False
                self.last_scene_text = result
                self.last_scene_time = time.time()
            print(
                f"\nVLM ({seconds:.2f}s, {len(detections)} YOLO tracks):\n"
                f"{result}\n"
            )
            if on_complete is not None and result:
                try:
                    on_complete(result)
                except Exception as callback_error:
                    print(
                        "VLM result callback failed: "
                        f"{type(callback_error).__name__}: {callback_error}"
                    )
        except Exception as exc:
            seconds = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.state.text = "VLM inference failed; see terminal"
                self.state.seconds = seconds
                self.state.error = message
                self.state.running = False
            print(f"\nVLM error: {message}\n")


    def submit_conversation(
        self,
        audio: bytes,
        sample_rate: int,
        fallback_text: str,
        robot_context: dict[str, object],
        on_complete=None,
    ) -> bool:
        """Transcribe and answer one conversational voice turn asynchronously."""
        with self.lock:
            if self.state.running:
                return False
            self.state.running = True
            self.state.error = None
            self.state.text = "Listening and preparing a response..."
            self.state.seconds = None
            self.state.detections_used = 0

        threading.Thread(
            target=self._converse,
            args=(
                bytes(audio),
                int(sample_rate),
                fallback_text,
                dict(robot_context),
                on_complete,
            ),
            daemon=True,
        ).start()
        return True

    def _get_transcriber(self):
        if self.transcriber is None:
            print("Loading desktop Whisper: openai/whisper-base.en")
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base.en",
                device=0,
                dtype=torch.float16,
            )
            print("Desktop Whisper ready")
        return self.transcriber

    def _converse(
        self,
        audio: bytes,
        sample_rate: int,
        fallback_text: str,
        robot_context: dict[str, object],
        on_complete=None,
    ) -> None:
        started = time.perf_counter()
        try:
            transcript = fallback_text
            if audio:
                samples = np.frombuffer(audio, dtype="<i2").astype(np.float32)
                samples /= 32768.0
                transcription = self._get_transcriber()(
                    {"raw": samples, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "language": "english",
                        "task": "transcribe",
                    },
                )
                candidate = str(transcription.get("text", "")).strip()
                if candidate:
                    transcript = candidate

            words = transcript.strip().split()
            if words and words[0].lower().rstrip(",.!?") == "fluffy":
                transcript = " ".join(words[1:]).strip()
            if not transcript:
                transcript = "What did you hear me say?"

            scene_age = (
                round(time.time() - self.last_scene_time, 1)
                if self.last_scene_time
                else None
            )
            context = dict(robot_context)
            context["last_visual_description"] = self.last_scene_text or None
            context["visual_description_age_seconds"] = scene_age

            system_prompt = (
                "You are Fluffy, an embodied robot dog speaking with a person. "
                "Be warm, curious and concise without pretending to be a real "
                "animal. Ground every claim about sight, identity, movement, "
                "distance and completed actions in the supplied robot state. "
                "A visible identity is not proof of who is speaking. Never claim "
                "that a requested action happened unless its confirmed state or "
                "result says so. You have no authority to invent or directly "
                "execute movement. If asked to do something outside the existing "
                "voice commands, explain that briefly. Reply in plain spoken "
                "English, normally one to three sentences, with no markdown."
            )
            context_text = json.dumps(context, ensure_ascii=False)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            ]
            for item in self.conversation_history:
                messages.append({
                    "role": item["role"],
                    "content": [{"type": "text", "text": item["content"]}],
                })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"Current verified robot state:\n{context_text}\n\n"
                        f"Person says: {transcript}"
                    ),
                }],
            })

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_new_tokens, 180),
                    do_sample=False,
                    use_cache=True,
                )
            generated_ids = output_ids[:, input_length:]
            result = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if not result:
                result = "I am not sure how to answer that yet."

            seconds = time.perf_counter() - started
            with self.lock:
                self.conversation_history.append({
                    "role": "user",
                    "content": transcript,
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": result,
                })
                self.state.text = result
                self.state.seconds = seconds
                self.state.running = False
            print(
                f"\nCONVERSATION ({seconds:.2f}s)\n"
                f"HEARD: {transcript}\n"
                f"FLUFFY: {result}\n"
            )
            if on_complete is not None:
                try:
                    on_complete(result)
                except Exception as callback_error:
                    print(
                        "Conversation speech callback failed: "
                        f"{type(callback_error).__name__}: {callback_error}"
                    )
        except Exception as exc:
            seconds = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.state.text = "Conversation failed; see terminal"
                self.state.seconds = seconds
                self.state.error = message
                self.state.running = False
            print(f"\nConversation error: {message}\n")


def synthesize_windows_speech(text: str) -> bytes:
    """Render offline Windows SAPI speech to a PCM WAV on the desktop."""
    if os.name != "nt":
        raise RuntimeError(
            "Desktop TTS currently requires Windows System.Speech"
        )

    with tempfile.TemporaryDirectory(prefix="pidog_tts_") as directory:
        wav_path = Path(directory) / "speech.wav"
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        wav_literal = str(wav_path).replace("'", "''")
        powershell = f"""
Add-Type -AssemblyName System.Speech
$wavPath = '{wav_literal}'
$textBytes = [System.Convert]::FromBase64String('{text_b64}')
$text = [System.Text.Encoding]::UTF8.GetString($textBytes)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $synth.Rate = 1
    $synth.SetOutputToWaveFile($wavPath)
    $synth.Speak($text)
}}
finally {{
    $synth.Dispose()
}}
"""
        encoded_command = base64.b64encode(
            powershell.encode("utf-16le")
        ).decode("ascii")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded_command,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Windows speech synthesis failed: {detail}")
        if not wav_path.exists():
            raise RuntimeError("Windows speech synthesis produced no WAV file")
        audio = wav_path.read_bytes()

    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise RuntimeError("Windows speech synthesis returned an invalid WAV")
    return audio


def speak_robot(robot_api: str, text: str) -> None:
    """Synthesize speech on the desktop and upload only the WAV to PiDog."""
    started = time.perf_counter()
    audio = synthesize_windows_speech(text)
    synthesis_seconds = time.perf_counter() - started

    response = requests.post(
        f"{robot_api.rstrip('/')}/audio/play",
        json={"audio_b64": base64.b64encode(audio).decode("ascii")},
        timeout=(1.0, 15.0),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error") or payload.get("ok") is False:
        raise RuntimeError(str(payload))
    print(
        f"[TTS] desktop synthesis {synthesis_seconds:.2f}s | "
        f"{len(audio) / 1024:.0f} KiB uploaded | "
        f"audio {payload.get('duration_s', '?')}s"
    )


def command_head(
    session: requests.Session,
    robot_api: str,
    yaw: float,
    pitch: float,
) -> bool:
    """Send one smooth manual head-position command through the Pi bridge."""

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


class LatestHeadController:
    """Deliver only the newest autonomous head target on a worker thread."""

    def __init__(self, robot_api: str):
        self.robot_api = robot_api.rstrip("/")
        self.condition = threading.Condition()
        self.pending: tuple[float, float] | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self.error_generation = 0
        self.reported_error_generation = 0
        self.sent_count = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=3)

    def submit(self, yaw: float, pitch: float) -> None:
        # A single pending slot is intentional: stale tracking positions must
        # never execute after a newer camera observation has arrived.
        with self.condition:
            self.pending = (float(yaw), float(pitch))
            self.condition.notify()

    def consume_failure(self) -> tuple[int, str] | None:
        with self.condition:
            if self.error_generation == self.reported_error_generation:
                return None
            self.reported_error_generation = self.error_generation
            return self.consecutive_failures, self.last_error or "unknown error"

    def _run(self) -> None:
        session = requests.Session()
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.pending is not None or self.stopping
                    )
                    if self.stopping:
                        return
                    yaw, pitch = self.pending
                    self.pending = None
                try:
                    response = session.post(
                        f"{self.robot_api}/head",
                        json={
                            "yaw": yaw,
                            "roll": 0,
                            "pitch": pitch,
                            "tracking": True,
                        },
                        timeout=(0.35, 1.0),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("error"):
                        raise RuntimeError(str(payload["error"]))
                    with self.condition:
                        self.consecutive_failures = 0
                        self.last_error = None
                        self.sent_count += 1
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    with self.condition:
                        self.consecutive_failures += 1
                        self.last_error = str(exc)
                        self.error_generation += 1
        finally:
            session.close()


VOICE_COMMANDS = {
    "select me": "select_me",
    "arm head": "arm_head",
    "track me": "arm_head",
    "follow me": "follow_me",
    "what do you see": "describe_scene",
    "what can you see": "describe_scene",
    "tell me what you see": "describe_scene",
    "stop": "stop",
    "halt": "stop",
    "lie down": "lie_down",
    "lay down": "lie_down",
    "stop and lie down": "stop_and_lie_down",
    "stop and lay down": "stop_and_lie_down",
}
VOICE_COMMAND_MAX_AGE_SECONDS = 5.0


def voice_message_command(message: dict[str, object]) -> str | None:
    """Return a canonical bounded command from one bridge inbox message."""
    command = message.get("command")
    if isinstance(command, str) and command in {
        "select_me",
        "arm_head",
        "follow_me",
        "describe_scene",
        "conversation",
        "stop",
        "lie_down",
        "stop_and_lie_down",
    }:
        return command
    text = message.get("text")
    if not isinstance(text, str):
        return None
    words = text.lower().strip().split()
    if words and words[0] in {"fluffy", "nox", "knox", "knocks"}:
        words = words[1:]
    return VOICE_COMMANDS.get(" ".join(words))


def fetch_voice_commands(
    session: requests.Session,
    robot_api: str,
) -> list[tuple[str, dict[str, object]]]:
    """Drain structured Pi voice messages without blocking the vision loop."""
    response = session.get(
        f"{robot_api.rstrip('/')}/voice/inbox",
        timeout=(0.35, 0.75),
    )
    response.raise_for_status()
    payload = response.json()
    commands: list[tuple[str, dict[str, object]]] = []
    received_at = time.time()
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        timestamp = message.get("ts")
        if (
            isinstance(timestamp, (int, float))
            and received_at - float(timestamp) > VOICE_COMMAND_MAX_AGE_SECONDS
        ):
            print(
                "Ignoring stale voice command "
                f"({received_at - float(timestamp):.1f}s old): "
                f"{message.get('text')!r}"
            )
            continue
        command = voice_message_command(message)
        if command is not None:
            commands.append((command, message))
    return commands


class VoiceCommandPoller:
    """Poll the Pi voice inbox without blocking the vision/control loop."""

    def __init__(self, robot_api: str, interval: float):
        self.robot_api = robot_api
        self.interval = interval
        self.commands: deque[tuple[str, dict[str, object]]] = deque()
        self.error: str | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def drain(
        self,
    ) -> tuple[list[tuple[str, dict[str, object]]], str | None]:
        with self.lock:
            commands = list(self.commands)
            self.commands.clear()
            error = self.error
            self.error = None
        return commands, error

    def _run(self) -> None:
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                try:
                    commands = fetch_voice_commands(session, self.robot_api)
                    with self.lock:
                        self.commands.extend(commands)
                        self.error = None
                except requests.RequestException as exc:
                    with self.lock:
                        self.error = str(exc)

                if self.stop_event.wait(self.interval):
                    break
        finally:
            session.close()


def command_body_action(
    session: requests.Session,
    robot_api: str,
    action: str,
    steps: int,
    speed: int,
    min_distance_cm: float,
) -> tuple[str, float | None, str | None]:
    """Request one allow-listed gait with Pi-side idle and clearance gates."""

    if action not in {"turn_left", "turn_right", "forward"}:
        raise ValueError(f"Body-follow controller rejected action: {action}")
    try:
        response = session.post(
            f"{robot_api.rstrip('/')}/action",
            json={
                "actions": [{
                    "cmd": "move",
                    "action": action,
                    "steps": steps,
                    "speed": speed,
                    "min_distance_cm": min_distance_cm,
                }],
                "require_idle": True,
            },
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            print(f"Body action returned unexpected data: {payload!r}")
            return "failed", None, "unexpected response"
        if payload.get("require_idle") is not True:
            print(
                "Body action rejected locally: the Pi bridge did not confirm "
                "idle enforcement. Install the updated nox_daemon.py and "
                "nox_brain_bridge.py before enabling body following."
            )
            return "failed", None, "idle enforcement unavailable"
        distance_cm = payload.get("distance_cm")
        if payload.get("safety_stop"):
            return "blocked", distance_cm, payload.get("reason")
        if payload.get("busy") and payload.get("accepted") is False:
            return "busy", distance_cm, None
        if (
            payload.get("error")
            or payload.get("ok") is False
            or payload.get("status") == "error"
        ):
            print(f"Body action rejected: {payload}")
            return "failed", distance_cm, payload.get("error")
        if payload.get("accepted") is True:
            results = payload.get("results") or []
            result = results[0] if results and isinstance(results[0], dict) else {}
            return "accepted", result.get("distance_cm"), None
        print(f"Body action returned no acceptance decision: {payload}")
        return "failed", distance_cm, "no acceptance decision"
    except (requests.RequestException, ValueError) as exc:
        print(f"Body action failed: {exc}")
        return "failed", None, str(exc)


def prepare_body_for_following(
    session: requests.Session,
    robot_api: str,
    stand_speed: int,
    completion_timeout: float = 12.0,
) -> tuple[bool, str | None]:
    """Slowly stand, then wait for authoritative Pi-side leg completion."""

    try:
        response = session.post(
            f"{robot_api.rstrip('/')}/action",
            json={
                "actions": [{
                    "cmd": "move",
                    "action": "stand",
                    "steps": 1,
                    "speed": stand_speed,
                }],
                "require_idle": True,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return False, "slow stand returned unexpected data"
        if payload.get("require_idle") is not True:
            return False, "Pi bridge did not confirm idle enforcement"
        if payload.get("busy") and payload.get("accepted") is False:
            return False, "legs are already busy"
        if payload.get("error") or payload.get("ok") is False:
            return False, str(payload.get("error") or payload)
        if payload.get("accepted") is not True:
            return False, "Pi did not accept the slow stand"

        deadline = time.monotonic() + completion_timeout
        while time.monotonic() < deadline:
            status_response = session.get(
                f"{robot_api.rstrip('/')}/motion/status",
                timeout=4,
            )
            status_response.raise_for_status()
            status = status_response.json()
            if not isinstance(status, dict) or status.get("ok") is False:
                return False, f"invalid motion status: {status!r}"
            if status.get("legs_done") and not status.get("busy"):
                time.sleep(0.5)
                arm_response = session.post(
                    f"{robot_api.rstrip('/')}/command",
                    json={"cmd": "arm_motion"},
                    timeout=4,
                )
                arm_response.raise_for_status()
                arm_payload = arm_response.json()
                if (
                    not isinstance(arm_payload, dict)
                    or arm_payload.get("ok") is not True
                    or arm_payload.get("motion_armed") is not True
                ):
                    return False, f"Pi motion latch did not arm: {arm_payload!r}"
                return True, None
            time.sleep(0.1)
        return False, "slow stand did not complete before timeout"
    except (requests.RequestException, ValueError) as exc:
        return False, str(exc)


def keep_body_awake(session: requests.Session, robot_api: str) -> bool:
    """Renew the Pi idle lease without generating a servo action."""

    try:
        response = session.post(
            f"{robot_api.rstrip('/')}/keep_awake",
            json={},
            timeout=4,
        )
        response.raise_for_status()
        payload = response.json()
        return isinstance(payload, dict) and payload.get("ok") is True
    except (requests.RequestException, ValueError):
        return False


def selectable_people(detections: list[Detection]) -> list[Detection]:
    """Return trackable people ordered by their current screen position."""

    return sorted(
        (
            item
            for item in detections
            if item.class_name == "person" and item.track_id is not None
        ),
        key=lambda item: item.centre[0],
    )[:9]


def selected_detection(
    detections: list[Detection],
    selected_track_id: int | None,
) -> Detection | None:
    if selected_track_id is None:
        return None
    return next(
        (
            item
            for item in detections
            if item.class_name == "person"
            and item.track_id == selected_track_id
        ),
        None,
    )


def identity_tracks(
    identity_labels: dict[int, tuple[str, float]],
    identity: str | None,
) -> list[int]:
    """Return the YOLO tracks currently owned by one recognised identity."""

    if identity is None:
        return []
    return [
        track_id
        for track_id, (name, _) in identity_labels.items()
        if name == identity
    ]


def target_aim_point(
    detection: Detection,
    vertical_fraction: float,
) -> tuple[int, int]:
    """Return a visible aim point within a detection's bounding box."""

    x1, y1, x2, y2 = detection.box
    return (
        (x1 + x2) // 2,
        int(y1 + (y2 - y1) * vertical_fraction),
    )


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / max(union, 1)


def matched_head_pose(
    detection: Detection | None,
    pose_state: PoseState,
) -> HeadPose | None:
    if detection is None or not pose_state.heads:
        return None
    best = max(
        pose_state.heads,
        key=lambda pose: box_iou(detection.box, pose.body_box),
    )
    if box_iou(detection.box, best.body_box) < MIN_POSE_BOX_IOU:
        return None
    x1, y1, x2, y2 = detection.box
    head_x, head_y = best.head_point
    if not (x1 <= head_x <= x2 and y1 <= head_y <= y2):
        return None
    return best


def target_signature(
    detection: Detection,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    """Coarse position/size signature used only for cautious reacquisition."""

    x1, y1, x2, y2 = detection.box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (
        detection.centre[0] / max(frame_width, 1),
        detection.centre[1] / max(frame_height, 1),
        area / max(frame_width * frame_height, 1),
    )


def reacquisition_candidates(
    detections: list[Detection],
    excluded_track_ids: set[int],
    signature: tuple[float, float, float] | None,
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    """Find new, unambiguous-looking tracks near the lost target signature."""

    if signature is None:
        return []
    old_x, old_y, old_area = signature
    candidates: list[Detection] = []
    for person in selectable_people(detections):
        if (
            person.track_id in excluded_track_ids
            or person.confidence < PERSON_ARM_CONFIDENCE
        ):
            continue
        new_x, new_y, new_area = target_signature(
            person,
            frame_width,
            frame_height,
        )
        position_distance = float(
            np.hypot(new_x - old_x, new_y - old_y)
        )
        area_ratio = new_area / max(old_area, 1e-6)
        if (
            position_distance <= REACQUIRE_MAX_POSITION_DISTANCE
            and REACQUIRE_MIN_AREA_RATIO
            <= area_ratio
            <= REACQUIRE_MAX_AREA_RATIO
        ):
            candidates.append(person)
    return candidates


def relative_axis(value: float) -> str:
    if value < 0.38:
        return "left"
    if value > 0.62:
        return "right"
    return "centre"


def relative_vertical(value: float) -> str:
    if value < 0.38:
        return "upper"
    if value > 0.62:
        return "lower"
    return "middle"


def area_to_distance(area_fraction: float) -> str:
    """Coarse monocular hint only; object size makes this inherently uncertain."""

    if area_fraction >= 0.22:
        return "near"
    if area_fraction >= 0.06:
        return "middle"
    return "far_or_small"


def track_colour(track_id: int | None) -> tuple[int, int, int]:
    seed = 0 if track_id is None else track_id
    return (
        60 + (seed * 67) % 196,
        60 + (seed * 97) % 196,
        60 + (seed * 137) % 196,
    )


def draw_detections(
    frame: np.ndarray,
    state: YoloState,
    selected_track_id: int | None,
    identity_labels: dict[int, tuple[str, float]],
) -> None:
    person_slots = {
        person.track_id: index
        for index, person in enumerate(
            selectable_people(state.detections),
            start=1,
        )
    }
    for detection in state.detections:
        x1, y1, x2, y2 = detection.box
        selected = detection.track_id == selected_track_id
        colour = (0, 255, 0) if selected else track_colour(detection.track_id)
        track_text = (
            "?" if detection.track_id is None else str(detection.track_id)
        )
        slot = person_slots.get(detection.track_id)
        slot_text = f"[{slot}] " if slot is not None else ""
        identity = (
            identity_labels.get(detection.track_id)
            if detection.track_id is not None
            else None
        )
        identity_text = (
            f" {identity[0]}:{identity[1]:.2f}"
            if identity is not None
            else ""
        )
        label = (
            f"{slot_text}{detection.class_name} #{track_text} "
            f"{detection.confidence:.2f}{identity_text}"
        )
        thickness = 4 if selected else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        cv2.circle(frame, detection.centre, 6 if selected else 4, colour, -1)
        label_y = max(20, y1 - 7)
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_vlm_panel(frame: np.ndarray, state: AnalysisState) -> None:
    height, width = frame.shape[:2]
    lines = textwrap.wrap(state.text, width=max(36, width // 12))[:7]
    if state.running:
        heading = f"VLM analysing ({state.detections_used} tracks supplied)"
        colour = (0, 215, 255)
    elif state.error:
        heading = "VLM error - see terminal"
        colour = (0, 0, 255)
    elif state.seconds is not None:
        heading = (
            f"VLM {state.seconds:.2f}s "
            f"({state.detections_used} tracks supplied)"
        )
        colour = (0, 255, 0)
    else:
        heading = "VLM ready"
        colour = (255, 255, 255)

    panel_height = min(height, 48 + 24 * max(1, len(lines)))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, height - panel_height),
        (width, height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = height - panel_height + 27
    cv2.putText(
        frame,
        heading,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        colour,
        2,
        cv2.LINE_AA,
    )
    for line in lines:
        y += 24
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PiDog YOLO tracker with asynchronous Qwen3-VL reasoning."
    )
    parser.add_argument(
        "--stream",
        default=DEFAULT_STREAM,
        help=(
            "Camera source. Defaults to the low-latency H.264 tunnel at "
            "tcp://127.0.0.1:19001. Use "
            "--stream http://127.0.0.1:19000/mjpg.jpg for frame mode."
        ),
    )
    parser.add_argument("--robot-api", default=DEFAULT_ROBOT_API)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--face-profiles-directory",
        default="face_profiles",
        help="Directory containing one .npz file per consenting identity.",
    )
    parser.add_argument("--face-model-directory", default="face_models")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Maximum VLM response tokens; 220 allows concise busy-scene summaries.",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--pose-confidence", type=float, default=0.35)
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=None,
        help="Override the recognition threshold stored in the profile.",
    )
    parser.add_argument("--face-fps", type=float, default=3.0)
    parser.add_argument(
        "--face-margin",
        type=float,
        default=DEFAULT_IDENTITY_MARGIN,
        help="Required cosine-score lead over the second-best identity.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--yolo-fps",
        type=float,
        default=15.0,
        help="Maximum YOLO processing rate; display remains uncapped.",
    )
    parser.add_argument(
        "--turn-yaw-threshold",
        type=float,
        default=DEFAULT_TURN_YAW_THRESHOLD,
        help="Head yaw in degrees that triggers one identity-locked body turn.",
    )
    parser.add_argument(
        "--body-turn-screen-threshold",
        type=float,
        default=DEFAULT_BODY_TURN_SCREEN_THRESHOLD,
        help=(
            "Normalised horizontal image error that triggers a body turn; "
            "forward motion pauses before this threshold is reached."
        ),
    )
    parser.add_argument(
        "--turn-interval",
        type=float,
        default=DEFAULT_TURN_COMMAND_INTERVAL,
        help="Minimum seconds between idle-enforced turn requests.",
    )
    parser.add_argument(
        "--turn-steps",
        type=int,
        default=3,
        help="Step count for each admitted turn; three completes the gait.",
    )
    parser.add_argument(
        "--turn-countersteer-degrees",
        type=float,
        default=12.0,
        help=(
            "Immediate opposite head-yaw correction when a body turn is "
            "accepted; visual tracking continues to refine it."
        ),
    )
    parser.add_argument(
        "--turn-speed",
        type=int,
        default=98,
        help=(
            "PiDog SDK speed for admitted turning corrections "
            "(98 matches the standard ball tracker)."
        ),
    )
    parser.add_argument(
        "--forward-steps",
        type=int,
        default=3,
        help="Step count for each admitted forward gait; three completes it.",
    )
    parser.add_argument(
        "--forward-speed",
        type=int,
        default=98,
        help=(
            "PiDog SDK speed for admitted forward gaits "
            "(98 matches the standard ball tracker)."
        ),
    )
    parser.add_argument(
        "--follow-distance",
        type=float,
        default=DEFAULT_FOLLOW_DISTANCE_CM,
        help="Stop admitting forward gaits at or below this ultrasonic distance (cm).",
    )
    parser.add_argument(
        "--turn-clearance",
        type=float,
        default=DEFAULT_TURN_CLEARANCE_CM,
        help="Block turns at or below this ultrasonic distance (cm).",
    )
    parser.add_argument(
        "--stand-speed",
        type=int,
        default=30,
        help="Low-speed stand used before body following is armed.",
    )
    parser.add_argument(
        "--camera-warmup-seconds",
        type=float,
        default=2.0,
        help="Fresh-frame warm-up period before automatic head centring.",
    )
    parser.add_argument(
        "--voice-poll-interval",
        type=float,
        default=0.25,
        help="Seconds between checks of the Pi voice-command inbox.",
    )
    parser.add_argument(
        "--disable-voice-commands",
        action="store_true",
        help="Do not consume commands from the Pi voice service.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.yolo_fps <= 0:
        raise ValueError("--yolo-fps must be greater than zero")
    if args.face_fps <= 0:
        raise ValueError("--face-fps must be greater than zero")
    if args.face_margin < 0:
        raise ValueError("--face-margin cannot be negative")
    if not 10.0 <= args.turn_yaw_threshold <= 70.0:
        raise ValueError("--turn-yaw-threshold must be between 10 and 70")
    if not 0.12 <= args.body_turn_screen_threshold <= 0.60:
        raise ValueError(
            "--body-turn-screen-threshold must be between 0.12 and 0.60"
        )
    if args.turn_interval < 0.25:
        raise ValueError("--turn-interval must be at least 0.25 seconds")
    if not 1 <= args.turn_steps <= 3:
        raise ValueError("--turn-steps must be between 1 and 3")
    if not 0.0 <= args.turn_countersteer_degrees <= 30.0:
        raise ValueError(
            "--turn-countersteer-degrees must be between 0 and 30"
        )
    if not 20 <= args.turn_speed <= 100:
        raise ValueError("--turn-speed must be between 20 and 100")
    if not 1 <= args.forward_steps <= 3:
        raise ValueError("--forward-steps must be between 1 and 3")
    if not 20 <= args.forward_speed <= 100:
        raise ValueError("--forward-speed must be between 20 and 100")
    if not 20.0 <= args.follow_distance <= 200.0:
        raise ValueError("--follow-distance must be between 20 and 200 cm")
    if not 10.0 <= args.turn_clearance <= 100.0:
        raise ValueError("--turn-clearance must be between 10 and 100 cm")
    if not 20 <= args.stand_speed <= 60:
        raise ValueError("--stand-speed must be between 20 and 60")
    if not 0.5 <= args.camera_warmup_seconds <= 10.0:
        raise ValueError(
            "--camera-warmup-seconds must be between 0.5 and 10 seconds"
        )
    if not 0.1 <= args.voice_poll_interval <= 5.0:
        raise ValueError("--voice-poll-interval must be between 0.1 and 5 seconds")

    yolo = YoloTracker(args.yolo_model, args.confidence, args.imgsz)
    pose = PoseEstimator(
        args.pose_model,
        args.pose_confidence,
        args.imgsz,
    )
    face_engine: FaceIdentityEngine | None = None
    face_profiles: list[FaceProfile] = []
    face_profiles_directory = Path(args.face_profiles_directory)
    if face_profiles_directory.exists():
        face_profiles = load_profiles(face_profiles_directory)
    if face_profiles:
        face_engine = FaceIdentityEngine(args.face_model_directory)
        profile_summary = ", ".join(
            f"{profile.name} ({len(profile.embeddings)} samples)"
            for profile in face_profiles
        )
        print(f"Face recognition gallery ready: {profile_summary}")
    else:
        print(
            "Face recognition disabled: no .npz identity profiles found in "
            f"{face_profiles_directory}"
        )
    vlm = VLMObserver(args.vlm_model, args.max_new_tokens)
    robot_session = requests.Session()
    head_controller = LatestHeadController(args.robot_api)
    head_controller.start()
    camera = LatestFrameCamera(args.stream)
    camera.start()

    yolo_enabled = True
    yolo_state = YoloState()
    pose_state = PoseState()
    yolo_frame: np.ndarray | None = None
    selected_track_id: int | None = None
    selected_identity: str | None = None
    selected_seen_count = 0
    target_missing_since: float | None = None
    known_other_track_ids: set[int] = set()
    lost_other_track_ids: set[int] = set()
    last_selected_signature: tuple[float, float, float] | None = None
    reacquire_candidate_id: int | None = None
    reacquire_seen_count = 0
    movement_enabled = False
    turning_enabled = False
    head_reference_known = False
    head_tracking_enabled = True
    aim_y_fraction = CENTRE_AIM_FRACTION
    yaw = 0.0
    pitch = 0.0
    last_command_time = 0.0
    last_turn_command_time = 0.0
    body_safety_stopped = False
    last_body_distance: float | None = None
    last_safety_reason: str | None = None
    last_body_keep_awake_time = 0.0
    last_yolo_time = 0.0
    last_sequence = -1
    display_fps = 0.0
    last_display_time = time.perf_counter()
    last_face_time = 0.0
    face_observations: list[FaceObservation] = []
    recognition_history: dict[int, deque[tuple[str | None, float]]] = {}
    identity_labels: dict[int, tuple[str, float]] = {}
    follow_log = FollowStateLogger()
    voice_command_queue: deque[tuple[str, dict[str, object]]] = deque()
    voice_poller: VoiceCommandPoller | None = None
    last_voice_error_log_time = 0.0
    voice_event_history: deque[dict[str, object]] = deque(maxlen=12)

    print(
        "1-9: select person | 0: clear | C: centre | M: head arm/disarm | "
        "T: identity body-follow arm/disarm | "
        "A: fallback aim | H: head aim | V/Space: VLM | Y: YOLO on/off | "
        "Q/Esc: quit"
    )
    if args.disable_voice_commands:
        print("Pi voice-command polling disabled")
    else:
        voice_poller = VoiceCommandPoller(
            args.robot_api,
            args.voice_poll_interval,
        )
        voice_poller.start()
        print(
            "Voice commands enabled asynchronously: select me | arm head | "
            "follow me | what do you see | conversation | stop | lie down"
        )
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
            raise RuntimeError(
                "No frames after 15 seconds. Check the camera stream and the "
                "SSH port-19000 forward."
            )

        warmup_deadline = time.monotonic() + args.camera_warmup_seconds
        warmup_frames = 0
        warmup_sequence = -1
        print(
            f"Warming camera stream for {args.camera_warmup_seconds:.1f} "
            "seconds..."
        )
        while time.monotonic() < warmup_deadline:
            sequence, warmup_frame, stream_error = camera.latest()
            if warmup_frame is not None and sequence != warmup_sequence:
                warmup_sequence = sequence
                warmup_frames += 1
            if stream_error:
                print(f"Camera warm-up warning: {stream_error}")
            time.sleep(0.01)
        if warmup_frames == 0:
            raise RuntimeError(
                "Camera stopped producing fresh frames during warm-up"
            )
        print(f"Camera stream ready after {warmup_frames} fresh frames")

        if command_head(robot_session, args.robot_api, 0.0, 0.0):
            yaw = 0.0
            pitch = 0.0
            head_reference_known = True
            print(
                "Head centred automatically; tracking and body following "
                "remain disarmed"
            )
        else:
            print(
                "Automatic head centring failed; press C after checking the "
                "Pi connection"
            )

        while True:
            sequence, clean_frame, stream_error = camera.latest()
            if clean_frame is None or sequence == last_sequence:
                time.sleep(0.003)
                continue
            last_sequence = sequence

            now = time.perf_counter()
            head_failure = head_controller.consume_failure()
            if head_failure is not None:
                failure_count, failure_reason = head_failure
                print(
                    f"[HEAD] tracking command failed ({failure_count}/3): "
                    f"{failure_reason}"
                )
                if failure_count >= 3 and movement_enabled:
                    movement_enabled = False
                    turning_enabled = False
                    print(
                        "[HEAD] head and body following disarmed after three "
                        "consecutive tracking-command failures"
                    )
            if (
                turning_enabled
                and now - last_body_keep_awake_time
                >= BODY_KEEP_AWAKE_INTERVAL
            ):
                if keep_body_awake(robot_session, args.robot_api):
                    last_body_keep_awake_time = now
                else:
                    turning_enabled = False
                    body_safety_stopped = False
                    print(
                        "Body following disarmed because the Pi activity "
                        "lease could not be renewed"
                    )
            if (
                yolo_enabled
                and now - last_yolo_time >= 1.0 / args.yolo_fps
            ):
                yolo_state = yolo.process(clean_frame)
                pose_state = pose.process(clean_frame)
                # Preserve the exact unannotated frame that produced this set
                # of detections so Qwen receives matching visual/text context.
                yolo_frame = clean_frame.copy()
                last_yolo_time = time.perf_counter()
                if yolo_state.error:
                    print(f"YOLO error: {yolo_state.error}")
                if pose_state.error:
                    print(f"Pose error: {pose_state.error}")

                # Once a selected person has a recognised name, that name owns
                # the target.  ByteTrack IDs can remain alive yet swap bodies
                # when people cross, so do not wait for the old track to vanish.
                named_tracks = identity_tracks(
                    identity_labels,
                    selected_identity,
                )
                if (
                    selected_identity is not None
                    and len(named_tracks) == 1
                    and named_tracks[0] != selected_track_id
                ):
                    old_track_id = selected_track_id
                    selected_track_id = named_tracks[0]
                    selected_seen_count = 0
                    target_missing_since = None
                    reacquire_candidate_id = None
                    reacquire_seen_count = 0
                    print(
                        f"Identity lock moved {selected_identity}: YOLO "
                        f"track #{old_track_id} -> #{selected_track_id}"
                    )

                target = selected_detection(
                    yolo_state.detections,
                    selected_track_id,
                )
                selected_track_label = identity_labels.get(
                    selected_track_id
                )
                identity_ownership_valid = (
                    selected_identity is None
                    or (
                        selected_track_label is not None
                        and selected_track_label[0] == selected_identity
                    )
                )
                target_is_valid = (
                    target is not None
                    and target.confidence >= PERSON_ARM_CONFIDENCE
                    and identity_ownership_valid
                )
                tracking_now = time.monotonic()
                frame_height, frame_width = clean_frame.shape[:2]
                if selected_track_id is not None and target_is_valid:
                    selected_seen_count += 1
                    target_missing_since = None
                    known_other_track_ids = {
                        person.track_id
                        for person in selectable_people(
                            yolo_state.detections
                        )
                        if person.track_id is not None
                        and person.track_id != selected_track_id
                    }
                    lost_other_track_ids.clear()
                    reacquire_candidate_id = None
                    reacquire_seen_count = 0
                    last_selected_signature = target_signature(
                        target,
                        frame_width,
                        frame_height,
                    )
                elif selected_track_id is not None:
                    selected_seen_count = 0
                    if turning_enabled:
                        follow_log.report(
                            "PAUSED_TARGET",
                            "selected identity is not currently valid",
                            identity=selected_identity,
                            track_id=selected_track_id,
                            yaw=yaw,
                            screen_error=None,
                            distance_cm=last_body_distance,
                        )
                    if target_missing_since is None:
                        target_missing_since = tracking_now
                        lost_other_track_ids = known_other_track_ids.copy()
                        reacquire_candidate_id = None
                        reacquire_seen_count = 0
                        if movement_enabled:
                            print(
                                "Selected person lost; tracking is paused "
                                "while reacquisition is attempted"
                            )

                    if (
                        tracking_now - target_missing_since
                        >= REACQUIRE_DELAY_SECONDS
                    ):
                        identity_candidates = [
                            person
                            for person in selectable_people(
                                yolo_state.detections
                            )
                            if person.track_id is not None
                            and selected_identity is not None
                            and identity_labels.get(person.track_id, (None, 0))[0]
                            == selected_identity
                        ]
                        if selected_identity is not None:
                            # A named target must only be reacquired by name.
                            # Falling back to position would select the other
                            # person after they exchange sides in the image.
                            candidates = (
                                identity_candidates
                                if len(identity_candidates) == 1
                                else []
                            )
                            reacquire_method = "face identity"
                        else:
                            candidates = reacquisition_candidates(
                                yolo_state.detections,
                                lost_other_track_ids,
                                last_selected_signature,
                                frame_width,
                                frame_height,
                            )
                            reacquire_method = "position and size"
                        if len(candidates) == 1:
                            candidate = candidates[0]
                            if candidate.track_id == reacquire_candidate_id:
                                reacquire_seen_count += 1
                            else:
                                reacquire_candidate_id = candidate.track_id
                                reacquire_seen_count = 1

                            if (
                                reacquire_seen_count
                                >= REACQUIRE_CONFIRM_FRAMES
                            ):
                                old_track_id = selected_track_id
                                selected_track_id = candidate.track_id
                                target = candidate
                                target_is_valid = True
                                selected_seen_count = PERSON_CONFIRM_FRAMES
                                target_missing_since = None
                                known_other_track_ids = {
                                    person.track_id
                                    for person in selectable_people(
                                        yolo_state.detections
                                    )
                                    if person.track_id is not None
                                    and person.track_id != selected_track_id
                                }
                                lost_other_track_ids.clear()
                                reacquire_candidate_id = None
                                reacquire_seen_count = 0
                                last_selected_signature = target_signature(
                                    candidate,
                                    frame_width,
                                    frame_height,
                                )
                                print(
                                    "Re-established selected person using "
                                    f"{reacquire_method}: YOLO "
                                    f"track #{old_track_id} -> "
                                    f"#{selected_track_id}"
                                )
                        else:
                            reacquire_candidate_id = None
                            reacquire_seen_count = 0

                if (
                    movement_enabled
                    and target_is_valid
                    and tracking_now - last_command_time >= COMMAND_INTERVAL
                ):
                    # Once armed, head recovery is intentionally immediate.
                    # Body gait admission below still requires five stable
                    # frames, but one blurred turn frame must not freeze the
                    # camera for the whole reacquisition confirmation window.
                    current_head_pose = matched_head_pose(
                        target,
                        pose_state,
                    )
                    if head_tracking_enabled and current_head_pose is not None:
                        aim_x, aim_y = current_head_pose.head_point
                    else:
                        fallback_fraction = (
                            UPPER_BODY_AIM_FRACTION
                            if head_tracking_enabled
                            else aim_y_fraction
                        )
                        aim_x, aim_y = target_aim_point(
                            target,
                            fallback_fraction,
                        )
                    error_x = (
                        aim_x - frame_width / 2
                    ) / max(frame_width / 2, 1)
                    error_y = (
                        aim_y - frame_height / 2
                    ) / max(frame_height / 2, 1)

                    yaw_step = 0.0
                    pitch_step = 0.0
                    if abs(error_x) >= X_DEADBAND:
                        yaw_step = float(
                            np.clip(
                                -error_x * YAW_GAIN,
                                -MAX_YAW_STEP_DEGREES,
                                MAX_YAW_STEP_DEGREES,
                            )
                        )
                    if abs(error_y) >= Y_DEADBAND:
                        pitch_step = float(
                            np.clip(
                                -error_y * PITCH_GAIN,
                                -MAX_PITCH_STEP_DEGREES,
                                MAX_PITCH_STEP_DEGREES,
                            )
                        )

                    if yaw_step != 0.0 or pitch_step != 0.0:
                        new_yaw = float(
                            np.clip(yaw + yaw_step, *YAW_LIMITS)
                        )
                        new_pitch = float(
                            np.clip(pitch + pitch_step, *PITCH_LIMITS)
                        )
                        head_controller.submit(new_yaw, new_pitch)
                        yaw = new_yaw
                        pitch = new_pitch
                    last_command_time = tracking_now

                # Identity-follow experiment: first centre the selected person
                # with a complete turn gait, then advance with a complete
                # forward gait. Every request is independently gated by the
                # Pi's live ultrasonic reading; backward is never allow-listed.
                if (
                    turning_enabled
                    and movement_enabled
                    and selected_identity is not None
                    and target_is_valid
                    and selected_seen_count >= PERSON_CONFIRM_FRAMES
                    and tracking_now - last_turn_command_time
                    >= args.turn_interval
                ):
                    body_head_pose = matched_head_pose(target, pose_state)
                    if head_tracking_enabled and body_head_pose is not None:
                        body_aim_x = body_head_pose.head_point[0]
                    else:
                        body_aim_x = target_aim_point(
                            target,
                            UPPER_BODY_AIM_FRACTION
                            if head_tracking_enabled else aim_y_fraction,
                        )[0]
                    body_error_x = (
                        body_aim_x - frame_width / 2
                    ) / max(frame_width / 2, 1)

                    body_action: str | None = None
                    decision_reason = "target is between steering regions"
                    body_steps: int
                    body_speed: int
                    min_distance_cm: float
                    if body_error_x >= args.body_turn_screen_threshold:
                        body_action = "turn_right"
                        decision_reason = "target is right of image centre"
                        body_steps = args.turn_steps
                        body_speed = args.turn_speed
                        min_distance_cm = args.turn_clearance
                    elif yaw <= -args.turn_yaw_threshold:
                        body_action = "turn_right"
                        decision_reason = "head yaw requires body realignment"
                        body_steps = args.turn_steps
                        body_speed = args.turn_speed
                        min_distance_cm = args.turn_clearance
                    elif body_error_x <= -args.body_turn_screen_threshold:
                        body_action = "turn_left"
                        decision_reason = "target is left of image centre"
                        body_steps = args.turn_steps
                        body_speed = args.turn_speed
                        min_distance_cm = args.turn_clearance
                    elif yaw >= args.turn_yaw_threshold:
                        body_action = "turn_left"
                        decision_reason = "head yaw requires body realignment"
                        body_steps = args.turn_steps
                        body_speed = args.turn_speed
                        min_distance_cm = args.turn_clearance
                    elif abs(body_error_x) <= X_DEADBAND:
                        body_action = "forward"
                        decision_reason = "target is centred and clearance is requested"
                        body_steps = args.forward_steps
                        body_speed = args.forward_speed
                        min_distance_cm = args.follow_distance

                    # In the band between head deadband and turn threshold,
                    # hold position. This lets the head catch up instead of
                    # walking forward while the target drifts out of frame.
                    if body_action is not None:
                        last_turn_command_time = tracking_now
                        action_result, distance_cm, safety_reason = command_body_action(
                            robot_session,
                            args.robot_api,
                            body_action,
                            body_steps,
                            body_speed,
                            min_distance_cm,
                        )
                        if distance_cm is not None:
                            last_body_distance = distance_cm
                        if action_result == "accepted":
                            if body_action in {"turn_left", "turn_right"}:
                                countersteer_sign = (
                                    1.0 if body_action == "turn_right" else -1.0
                                )
                                yaw = float(
                                    np.clip(
                                        yaw
                                        + countersteer_sign
                                        * args.turn_countersteer_degrees,
                                        *YAW_LIMITS,
                                    )
                                )
                                head_controller.submit(yaw, pitch)
                                print(
                                    "[HEAD] turn countersteer "
                                    f"{body_action}: yaw={yaw:+.1f}deg; "
                                    "visual corrections remain active"
                                )
                            if body_safety_stopped:
                                print(
                                    "Ultrasonic clearance restored; following resumed"
                                )
                            body_safety_stopped = False
                            last_safety_reason = None
                            follow_log.report(
                                body_action.upper(),
                                decision_reason,
                                identity=selected_identity,
                                track_id=selected_track_id,
                                yaw=yaw,
                                screen_error=body_error_x,
                                distance_cm=last_body_distance,
                                force=True,
                            )
                        elif action_result == "busy":
                            follow_log.report(
                                "LEGS_BUSY",
                                f"waiting before {body_action}",
                                identity=selected_identity,
                                track_id=selected_track_id,
                                yaw=yaw,
                                screen_error=body_error_x,
                                distance_cm=last_body_distance,
                            )
                        elif action_result == "blocked":
                            follow_log.report(
                                "SAFETY_STOP",
                                safety_reason or "ultrasonic admission rejected",
                                identity=selected_identity,
                                track_id=selected_track_id,
                                yaw=yaw,
                                screen_error=body_error_x,
                                distance_cm=distance_cm,
                            )
                            body_safety_stopped = True
                            last_safety_reason = safety_reason
                        elif action_result == "failed":
                            turning_enabled = False
                            body_safety_stopped = False
                            print(
                                "Body following disarmed because the action "
                                "command failed"
                            )
                    else:
                        follow_log.report(
                            "HOLD_CENTERING",
                            "target is off-centre; allowing head to catch up",
                            identity=selected_identity,
                            track_id=selected_track_id,
                            yaw=yaw,
                            screen_error=body_error_x,
                            distance_cm=last_body_distance,
                        )

            if (
                face_engine is not None
                and face_profiles
                and now - last_face_time >= 1.0 / args.face_fps
            ):
                face_observations = face_engine.observe(clean_frame)
                last_face_time = time.perf_counter()
                visible_person_ids = {
                    person.track_id
                    for person in selectable_people(yolo_state.detections)
                    if person.track_id is not None
                }
                for track_id in list(recognition_history):
                    if track_id not in visible_person_ids:
                        recognition_history.pop(track_id, None)
                        identity_labels.pop(track_id, None)

                for person in selectable_people(yolo_state.detections):
                    if person.track_id is None:
                        continue
                    matching_faces = [
                        face
                        for face in face_observations
                        if face_inside_person(face, person.box)
                    ]
                    if not matching_faces:
                        continue
                    face = max(
                        matching_faces,
                        key=lambda item: item.detector_score,
                    )
                    identity, score, _ = identify_embedding(
                        face.embedding,
                        face_profiles,
                        threshold_override=args.face_threshold,
                        minimum_margin=args.face_margin,
                    )
                    history = recognition_history.setdefault(
                        person.track_id,
                        deque(maxlen=5),
                    )
                    history.append((identity, score))

                    # A confident observation of a different known person is
                    # enough to revoke the old track's label immediately.  A
                    # new label still needs temporal consensus below before it
                    # can own the robot's target.
                    previous_label = identity_labels.get(person.track_id)
                    conflicting_observation = (
                        identity is not None
                        and previous_label is not None
                        and identity != previous_label[0]
                    )
                    if conflicting_observation:
                        identity_labels.pop(person.track_id, None)

                    if len(history) >= 5:
                        name_counts: dict[str, int] = {}
                        for observed_name, _ in history:
                            if observed_name is not None:
                                name_counts[observed_name] = (
                                    name_counts.get(observed_name, 0) + 1
                                )
                        if name_counts:
                            winning_name, winning_count = max(
                                name_counts.items(),
                                key=lambda item: item[1],
                            )
                        else:
                            winning_name, winning_count = None, 0

                        if (
                            not conflicting_observation
                            and winning_name is not None
                            and winning_count >= 4
                        ):
                            # A consenting identity can own only one current
                            # track.  Retire any stale pre-crossing assignment.
                            for other_track_id, other_label in list(
                                identity_labels.items()
                            ):
                                if (
                                    other_track_id != person.track_id
                                    and other_label[0] == winning_name
                                ):
                                    identity_labels.pop(other_track_id, None)
                            winning_scores = [
                                observed_score
                                for observed_name, observed_score in history
                                if observed_name == winning_name
                            ]
                            identity_labels[person.track_id] = (
                                winning_name,
                                float(np.mean(winning_scores)),
                            )
                        else:
                            previous_label = identity_labels.get(
                                person.track_id
                            )
                            if previous_label is not None:
                                retained_count = sum(
                                    observed_name == previous_label[0]
                                    for observed_name, _ in history
                                )
                                if retained_count <= 1:
                                    identity_labels.pop(
                                        person.track_id,
                                        None,
                                    )

                if selected_track_id is not None:
                    selected_label = identity_labels.get(selected_track_id)
                    if (
                        selected_identity is None
                        and selected_label is not None
                    ):
                        # Learn an identity for an anonymously selected track,
                        # but never overwrite an existing operator target with
                        # the different person who inherited its YOLO ID.
                        selected_identity = selected_label[0]

            display_frame = clean_frame.copy()
            draw_detections(
                display_frame,
                yolo_state,
                selected_track_id,
                identity_labels,
            )
            for face in face_observations:
                x1, y1, x2, y2 = face.box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    1,
                )
                for landmark in face.landmarks:
                    cv2.circle(
                        display_frame,
                        landmark,
                        2,
                        (255, 0, 255),
                        -1,
                    )
            display_target = selected_detection(
                yolo_state.detections,
                selected_track_id,
            )
            frame_height, frame_width = display_frame.shape[:2]
            frame_centre = (frame_width // 2, frame_height // 2)
            cv2.drawMarker(
                display_frame,
                frame_centre,
                (255, 255, 0),
                cv2.MARKER_CROSS,
                28,
                2,
            )
            if display_target is not None:
                display_head_pose = matched_head_pose(
                    display_target,
                    pose_state,
                )
                if head_tracking_enabled and display_head_pose is not None:
                    display_aim_point = display_head_pose.head_point
                    for face_point in display_head_pose.face_points:
                        cv2.circle(
                            display_frame,
                            face_point,
                            3,
                            (255, 0, 255),
                            -1,
                        )
                    aim_marker_colour = (255, 0, 255)
                else:
                    display_fallback_fraction = (
                        UPPER_BODY_AIM_FRACTION
                        if head_tracking_enabled
                        else aim_y_fraction
                    )
                    display_aim_point = target_aim_point(
                        display_target,
                        display_fallback_fraction,
                    )
                    aim_marker_colour = (0, 255, 0)
                cv2.line(
                    display_frame,
                    frame_centre,
                    display_aim_point,
                    aim_marker_colour,
                    2,
                )
                cv2.drawMarker(
                    display_frame,
                    display_aim_point,
                    aim_marker_colour,
                    cv2.MARKER_CROSS,
                    18,
                    2,
                )
            vlm_state = vlm.snapshot()
            draw_vlm_panel(display_frame, vlm_state)

            display_now = time.perf_counter()
            period = display_now - last_display_time
            if period > 0:
                instant_fps = 1.0 / period
                display_fps = (
                    instant_fps
                    if display_fps == 0
                    else 0.85 * display_fps + 0.15 * instant_fps
                )
            last_display_time = display_now

            status = (
                f"YOLO {'ON' if yolo_enabled else 'OFF'} | "
                f"{len(yolo_state.detections)} tracks | "
                f"detect {yolo_state.inference_ms:.0f} ms | "
                f"pose {pose_state.inference_ms:.0f} ms | "
                f"{display_fps:.1f} FPS"
            )
            cv2.putText(
                display_frame,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            target_text = (
                "none"
                if selected_track_id is None
                else (
                    f"{selected_identity or 'person'} "
                    f"#{selected_track_id}"
                )
            )
            target_ready = (
                display_target is not None
                and display_target.confidence >= PERSON_ARM_CONFIDENCE
                and selected_seen_count >= PERSON_CONFIRM_FRAMES
            )
            if movement_enabled and target_ready:
                movement_text = "TRACKING"
            elif movement_enabled:
                movement_text = "PAUSED/REACQUIRING"
            else:
                movement_text = "DISARMED"
            movement_colour = (
                (0, 255, 0)
                if movement_enabled and target_ready
                else (
                    (0, 215, 255)
                    if movement_enabled
                    else (0, 180, 255)
                )
            )
            head_pose_available = (
                matched_head_pose(display_target, pose_state) is not None
            )
            if head_tracking_enabled:
                aim_mode_text = (
                    "head-keypoints"
                    if head_pose_available
                    else "head-fallback"
                )
            else:
                aim_mode_text = (
                    "centre"
                    if aim_y_fraction == CENTRE_AIM_FRACTION
                    else "upper-body"
                )
            selected_display_label = identity_labels.get(selected_track_id)
            turn_target_ready = (
                movement_enabled
                and target_ready
                and selected_identity is not None
                and selected_display_label is not None
                and selected_display_label[0] == selected_identity
            )
            if turning_enabled and turn_target_ready:
                if body_safety_stopped:
                    turning_text = "SAFETY STOP"
                    turning_colour = (0, 0, 255)
                else:
                    turning_text = "ARMED"
                    turning_colour = (0, 255, 0)
            elif turning_enabled:
                turning_text = "PAUSED"
                turning_colour = (0, 215, 255)
            else:
                turning_text = "DISARMED"
                turning_colour = (0, 180, 255)
            cv2.putText(
                display_frame,
                f"Target {target_text} | {movement_text} | "
                f"stable {min(selected_seen_count, PERSON_CONFIRM_FRAMES)}/"
                f"{PERSON_CONFIRM_FRAMES} | yaw={yaw:.1f} pitch={pitch:.1f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                movement_colour,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Body follow {turning_text} | T toggle | "
                f"stop {args.follow_distance:.0f}cm | "
                f"aim:{aim_mode_text}",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                turning_colour,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "1-9 select | C centre | M head arm | T body follow | "
                "V analyse | Q quit",
                (10, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if stream_error:
                cv2.putText(
                    display_frame,
                    stream_error,
                    (10, 132),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("PiDog YOLO + Qwen VLM", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # The worker performs all voice-inbox network I/O. Safety
            # commands have already been executed locally by the Pi; consuming
            # them here disarms the desktop controller so it cannot admit
            # another gait.
            voice_now = time.perf_counter()
            speak_analysis_result = False
            if voice_poller is not None:
                new_voice_commands, voice_error = voice_poller.drain()
                voice_command_queue.extend(new_voice_commands)
                if (
                    voice_error is not None
                    and voice_now - last_voice_error_log_time >= 10.0
                ):
                    print(f"Voice inbox unavailable: {voice_error}")
                    last_voice_error_log_time = voice_now

            if voice_command_queue:
                voice_command, voice_message = voice_command_queue.popleft()
                confidence = voice_message.get("confidence")
                print(
                    f"VOICE COMMAND: {voice_command} "
                    f"(heard={voice_message.get('text')!r}, "
                    f"confidence={confidence})"
                )
                voice_event_history.append({
                    "time": round(time.time(), 3),
                    "command": voice_command,
                    "heard": voice_message.get("text"),
                    "local_action": voice_message.get("local_action"),
                    "local_ok": voice_message.get("local_ok"),
                })
                if voice_command in {
                    "stop",
                    "lie_down",
                    "stop_and_lie_down",
                }:
                    movement_enabled = False
                    turning_enabled = False
                    body_safety_stopped = False
                    key = 255
                    local_status = (
                        "confirmed by Pi"
                        if voice_message.get("local_ok")
                        else "Pi acknowledgement missing"
                    )
                    print(
                        f"Voice {voice_command}: head/body following DISARMED; "
                        f"local posture action {local_status}"
                    )
                elif key == 255 and voice_command == "conversation":
                    encoded_audio = voice_message.get("audio_b64")
                    try:
                        if not isinstance(encoded_audio, str):
                            raise ValueError("conversation message has no audio")
                        conversation_audio = base64.b64decode(
                            encoded_audio,
                            validate=True,
                        )
                        sample_rate = int(
                            voice_message.get("sample_rate", 16000)
                        )
                        if not 8000 <= sample_rate <= 48000:
                            raise ValueError(
                                f"invalid sample rate: {sample_rate}"
                            )
                        robot_context = {
                            "robot_name": "Fluffy",
                            "speaker_identity": "unknown",
                            "selected_visible_identity": selected_identity,
                            "selected_track_id": selected_track_id,
                            "head_tracking_armed": movement_enabled,
                            "body_following_armed": turning_enabled,
                            "head_yaw_degrees": round(yaw, 1),
                            "head_pitch_degrees": round(pitch, 1),
                            "ultrasonic_distance_cm": last_body_distance,
                            "body_safety_stopped": body_safety_stopped,
                            "last_safety_reason": last_safety_reason,
                            "visible_detections": [
                                item.prompt_record()
                                for item in yolo_state.detections
                            ],
                            "recent_voice_events": list(
                                voice_event_history
                            ),
                        }
                        accepted = vlm.submit_conversation(
                            conversation_audio,
                            sample_rate,
                            str(voice_message.get("text", "")),
                            robot_context,
                            on_complete=lambda result: speak_robot(
                                args.robot_api,
                                result,
                            ),
                        )
                        if accepted:
                            print(
                                "Conversational turn accepted; desktop "
                                "Whisper and Fluffy LLM response started"
                            )
                        else:
                            print(
                                "Conversation ignored because Qwen is "
                                "already processing another request"
                            )
                            speak_robot(
                                args.robot_api,
                                "I am still thinking about the previous request.",
                            )
                    except Exception as exc:
                        print(f"Conversation request rejected: {exc}")
                    key = 255
                elif key == 255 and voice_command == "describe_scene":
                    speak_analysis_result = True
                    key = ord("v")
                    print(
                        "Voice scene request accepted: analysing the newest "
                        "camera frame and preparing a spoken response"
                    )
                elif key == 255 and voice_command == "select_me":
                    people = selectable_people(yolo_state.detections)
                    recognised_slots = [
                        slot
                        for slot, person in enumerate(people[:9])
                        if person.track_id in identity_labels
                    ]
                    if len(recognised_slots) == 1:
                        slot = recognised_slots[0]
                        identity = identity_labels[people[slot].track_id][0]
                        print(
                            f"Voice select me resolved uniquely to {identity} "
                            f"in person slot {slot + 1}"
                        )
                        key = ord("1") + slot
                    elif not recognised_slots:
                        print(
                            "Voice select me rejected: no uniquely recognised "
                            "visible identity"
                        )
                    else:
                        identities = [
                            identity_labels[people[slot].track_id][0]
                            for slot in recognised_slots
                        ]
                        print(
                            "Voice select me rejected: multiple recognised "
                            f"people are visible ({', '.join(identities)}); "
                            "speaker verification is not enabled yet"
                        )
                elif key == 255 and voice_command == "arm_head":
                    if movement_enabled:
                        print("Voice arm head: head tracking is already armed")
                    else:
                        key = ord("m")
                elif key == 255 and voice_command == "follow_me":
                    if turning_enabled:
                        print("Voice follow me: body following is already armed")
                    else:
                        key = ord("t")

            if key in (ord("q"), 27):
                break
            if key == ord("y"):
                yolo_enabled = not yolo_enabled
                if not yolo_enabled:
                    movement_enabled = False
                    turning_enabled = False
                print(f"YOLO enabled: {yolo_enabled}")
            if ord("1") <= key <= ord("9"):
                slot = key - ord("1")
                people = selectable_people(yolo_state.detections)
                if slot < len(people):
                    chosen_person = people[slot]
                    selected_track_id = chosen_person.track_id
                    selected_label = identity_labels.get(selected_track_id)
                    selected_identity = (
                        selected_label[0]
                        if selected_label is not None
                        else None
                    )
                    selected_seen_count = 0
                    target_missing_since = None
                    known_other_track_ids = {
                        person.track_id
                        for person in people
                        if person.track_id is not None
                        and person.track_id != selected_track_id
                    }
                    lost_other_track_ids.clear()
                    reacquire_candidate_id = None
                    reacquire_seen_count = 0
                    selection_height, selection_width = clean_frame.shape[:2]
                    last_selected_signature = target_signature(
                        chosen_person,
                        selection_width,
                        selection_height,
                    )
                    movement_enabled = False
                    turning_enabled = False
                    identity_status = (
                        f"identity {selected_identity} locked"
                        if selected_identity is not None
                        else "identity pending"
                    )
                    print(
                        f"Selected person {slot + 1}: YOLO track "
                        f"#{selected_track_id}, {identity_status}. "
                        "Movement remains disarmed."
                    )
                else:
                    print(
                        f"Cannot select person {slot + 1}: only "
                        f"{len(people)} trackable people are visible"
                    )
            if key == ord("0"):
                selected_track_id = None
                selected_identity = None
                selected_seen_count = 0
                target_missing_since = None
                known_other_track_ids.clear()
                lost_other_track_ids.clear()
                last_selected_signature = None
                reacquire_candidate_id = None
                reacquire_seen_count = 0
                movement_enabled = False
                turning_enabled = False
                print("Person selection cleared; head and body disarmed")
            if key == ord("a"):
                movement_enabled = False
                turning_enabled = False
                aim_y_fraction = (
                    UPPER_BODY_AIM_FRACTION
                    if aim_y_fraction == CENTRE_AIM_FRACTION
                    else CENTRE_AIM_FRACTION
                )
                aim_mode = (
                    "centre"
                    if aim_y_fraction == CENTRE_AIM_FRACTION
                    else "upper-body"
                )
                print(
                    f"Fallback vertical aim point: {aim_mode}; "
                    "movement disarmed"
                )
            if key == ord("h"):
                movement_enabled = False
                turning_enabled = False
                head_tracking_enabled = not head_tracking_enabled
                print(
                    "Facial-keypoint head aiming "
                    f"{'enabled' if head_tracking_enabled else 'disabled'}; "
                    "movement disarmed"
                )
            if key == ord("c"):
                movement_enabled = False
                turning_enabled = False
                if command_head(
                    robot_session,
                    args.robot_api,
                    0.0,
                    0.0,
                ):
                    yaw = 0.0
                    pitch = 0.0
                    head_reference_known = True
                    print("Head centred; movement remains disarmed")
            if key == ord("m"):
                if movement_enabled:
                    movement_enabled = False
                    turning_enabled = False
                    print("Head tracking and body following disarmed")
                else:
                    current_target = selected_detection(
                        yolo_state.detections,
                        selected_track_id,
                    )
                    if selected_track_id is None:
                        print("Cannot arm: select a person with keys 1-9")
                    elif not head_reference_known:
                        print("Cannot arm: press C to centre the head first")
                    elif current_target is None:
                        print("Cannot arm: selected person is not visible")
                    elif current_target.confidence < PERSON_ARM_CONFIDENCE:
                        print(
                            "Cannot arm: selected person's confidence is "
                            f"{current_target.confidence:.2f}; need "
                            f"{PERSON_ARM_CONFIDENCE:.2f}"
                        )
                    elif selected_seen_count < PERSON_CONFIRM_FRAMES:
                        print(
                            "Cannot arm: waiting for stable tracking "
                            f"({selected_seen_count}/{PERSON_CONFIRM_FRAMES})"
                        )
                    else:
                        movement_enabled = True
                        last_command_time = 0.0
                        print(
                            "Person tracking ARMED for "
                            f"{selected_identity or 'unrecognised person'} "
                            f"on YOLO track #{selected_track_id}"
                        )
            if key == ord("t"):
                if turning_enabled:
                    turning_enabled = False
                    body_safety_stopped = False
                    print("Identity-locked body following disarmed")
                else:
                    current_target = selected_detection(
                        yolo_state.detections,
                        selected_track_id,
                    )
                    current_label = identity_labels.get(selected_track_id)
                    if not movement_enabled:
                        print("Cannot arm body following: arm head tracking with M")
                    elif selected_identity is None:
                        print(
                            "Cannot arm body following: wait for a recognised "
                            "identity label"
                        )
                    elif (
                        current_label is None
                        or current_label[0] != selected_identity
                    ):
                        print(
                            "Cannot arm body following: selected identity is "
                            "not confirmed on the current track"
                        )
                    elif current_target is None:
                        print(
                            "Cannot arm body following: selected person is not "
                            "visible"
                        )
                    elif current_target.confidence < PERSON_ARM_CONFIDENCE:
                        print(
                            "Cannot arm body following: selected person's "
                            f"confidence is {current_target.confidence:.2f}; "
                            f"need {PERSON_ARM_CONFIDENCE:.2f}"
                        )
                    elif selected_seen_count < PERSON_CONFIRM_FRAMES:
                        print(
                            "Cannot arm body following: waiting for stable "
                            f"tracking ({selected_seen_count}/"
                            f"{PERSON_CONFIRM_FRAMES})"
                        )
                    else:
                        print(
                            f"Preparing body: slow stand at speed "
                            f"{args.stand_speed}..."
                        )
                        prepared, preparation_error = prepare_body_for_following(
                            robot_session,
                            args.robot_api,
                            args.stand_speed,
                        )
                        if not prepared:
                            turning_enabled = False
                            body_safety_stopped = False
                            print(
                                "Body following remains disarmed: slow stand "
                                f"failed ({preparation_error})"
                            )
                        else:
                            turning_enabled = True
                            body_safety_stopped = False
                            last_body_keep_awake_time = time.perf_counter()
                            last_turn_command_time = last_body_keep_awake_time
                            print(
                                "IDENTITY-LOCKED BODY FOLLOWING ARMED for "
                                f"{selected_identity} after slow stand completed. "
                                "The Pi will reject commands while its legs are "
                                "busy or ultrasonic clearance is unsafe. "
                                "Backward remains disabled."
                            )
            if key in (ord("v"), ord(" ")):
                if yolo_enabled and yolo_frame is not None:
                    analysis_frame = yolo_frame
                    analysis_detections = yolo_state.detections
                else:
                    _, analysis_frame, _ = camera.latest()
                    analysis_detections = []
                if analysis_frame is not None:
                    selected_context = (
                        "No person is currently operator-selected."
                        if selected_track_id is None
                        else (
                            "The operator-selected person has YOLO track_id "
                            f"{selected_track_id} and locally recognised "
                            f"identity {selected_identity or 'unknown'}."
                        )
                    )
                    completion_callback = (
                        (lambda result: speak_robot(args.robot_api, result))
                        if speak_analysis_result
                        else None
                    )
                    if vlm.submit(
                        analysis_frame,
                        analysis_detections,
                        f"{args.prompt}\n{selected_context}",
                        on_complete=completion_callback,
                    ):
                        print(
                            "VLM analysis started with "
                            f"{len(analysis_detections)} YOLO tracks"
                        )
                    else:
                        print("VLM inference already running; request ignored")
                        if speak_analysis_result:
                            try:
                                speak_robot(
                                    args.robot_api,
                                    "I am still looking at the previous scene.",
                                )
                            except Exception as exc:
                                print(f"Could not speak busy response: {exc}")
    finally:
        if voice_poller is not None:
            voice_poller.stop()
        head_controller.stop()
        camera.stop()
        robot_session.close()
        cv2.destroyAllWindows()
        print("PiDog YOLO + VLM person tracker stopped")


if __name__ == "__main__":
    main()
