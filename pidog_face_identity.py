"""Local face detection, embedding, enrolment and matching for PiDog.

Uses OpenCV Zoo YuNet (face detection) and SFace (face recognition). Profiles
contain normalized embeddings and metadata; they do not contain executable
model code. Full camera frames are never stored by this module.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests


YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{YUNET_FILENAME}"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_recognition_sface/{SFACE_FILENAME}"
)
DEFAULT_FACE_THRESHOLD = 0.45
DEFAULT_IDENTITY_MARGIN = 0.04


@dataclass(frozen=True)
class FaceObservation:
    box: tuple[int, int, int, int]
    landmarks: tuple[tuple[int, int], ...]
    detector_score: float
    embedding: np.ndarray
    aligned_face: np.ndarray

    @property
    def centre(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True)
class FaceProfile:
    name: str
    embeddings: np.ndarray
    centroid: np.ndarray
    threshold: float
    created_utc: str
    metadata: dict[str, object]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} from the official OpenCV Zoo...")
    response = requests.get(url, stream=True, timeout=(10, 120))
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=destination.parent,
        suffix=".download",
    ) as temporary:
        temporary_path = Path(temporary.name)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                temporary.write(chunk)
    if temporary_path.stat().st_size < 100_000:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {destination.name} is unexpectedly small; "
            "the network may have returned an error page."
        )
    os.replace(temporary_path, destination)


def ensure_face_models(model_directory: str | Path) -> tuple[Path, Path]:
    model_directory = Path(model_directory)
    yunet_path = model_directory / YUNET_FILENAME
    sface_path = model_directory / SFACE_FILENAME
    if not yunet_path.exists():
        _download(YUNET_URL, yunet_path)
    if not sface_path.exists():
        _download(SFACE_URL, sface_path)
    return yunet_path, sface_path


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    flattened = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if norm <= 1e-12:
        raise ValueError("Face model produced a zero-length embedding")
    return flattened / norm


class FaceIdentityEngine:
    """OpenCV YuNet detector plus SFace embedding model."""

    def __init__(
        self,
        model_directory: str | Path = "face_models",
        detection_threshold: float = 0.80,
    ):
        yunet_path, sface_path = ensure_face_models(model_directory)
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=detection_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    def observe(self, frame: np.ndarray) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        observations: list[FaceObservation] = []
        for face in faces:
            x, y, w, h = (int(value) for value in face[:4])
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                embedding = normalize_embedding(
                    self.recognizer.feature(aligned)
                )
            except cv2.error:
                continue
            landmark_values = face[4:14].reshape(5, 2)
            landmarks = tuple(
                (int(point[0]), int(point[1]))
                for point in landmark_values
            )
            observations.append(
                FaceObservation(
                    box=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    detector_score=float(face[-1]),
                    embedding=embedding,
                    aligned_face=aligned,
                )
            )
        return observations


def create_profile(
    name: str,
    embeddings: list[np.ndarray] | np.ndarray,
    threshold: float = DEFAULT_FACE_THRESHOLD,
    metadata: dict[str, object] | None = None,
) -> FaceProfile:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 5:
        raise ValueError("At least five face embeddings are required")
    matrix = np.stack([normalize_embedding(row) for row in matrix])
    centroid = normalize_embedding(matrix.mean(axis=0))
    return FaceProfile(
        name=name,
        embeddings=matrix,
        centroid=centroid,
        threshold=float(threshold),
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=dict(metadata or {}),
    )


def save_profile(profile: FaceProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(profile.name),
        embeddings=profile.embeddings.astype(np.float32),
        centroid=profile.centroid.astype(np.float32),
        threshold=np.asarray(profile.threshold, dtype=np.float32),
        created_utc=np.asarray(profile.created_utc),
        metadata_json=np.asarray(json.dumps(profile.metadata)),
    )


def load_profile(path: str | Path) -> FaceProfile:
    with np.load(Path(path), allow_pickle=False) as stored:
        embeddings = np.asarray(stored["embeddings"], dtype=np.float32)
        centroid = normalize_embedding(stored["centroid"])
        metadata = json.loads(str(stored["metadata_json"].item()))
        return FaceProfile(
            name=str(stored["name"].item()),
            embeddings=np.stack(
                [normalize_embedding(row) for row in embeddings]
            ),
            centroid=centroid,
            threshold=float(stored["threshold"].item()),
            created_utc=str(stored["created_utc"].item()),
            metadata=metadata,
        )


def load_profiles(directory: str | Path) -> list[FaceProfile]:
    """Load every identity profile in a directory in stable filename order."""

    directory = Path(directory)
    profiles = [load_profile(path) for path in sorted(directory.glob("*.npz"))]
    names = [profile.name.casefold() for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate identity names found in {directory}; each profile's "
            "embedded name must be unique."
        )
    return profiles


def profile_similarity(
    embedding: np.ndarray,
    profile: FaceProfile,
) -> float:
    query = normalize_embedding(embedding)
    sample_scores = profile.embeddings @ query
    top_count = min(3, len(sample_scores))
    top_scores = np.partition(sample_scores, -top_count)[-top_count:]
    gallery_score = float(np.mean(top_scores))
    centroid_score = float(profile.centroid @ query)
    return max(gallery_score, centroid_score)


def identify_embedding(
    embedding: np.ndarray,
    profiles: list[FaceProfile],
    threshold_override: float | None = None,
    minimum_margin: float = DEFAULT_IDENTITY_MARGIN,
) -> tuple[str | None, float, float]:
    """Return an identity only when it clears threshold and runner-up margin."""

    if not profiles:
        return None, 0.0, 0.0
    ranked = sorted(
        (
            (profile_similarity(embedding, profile), profile)
            for profile in profiles
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_profile = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    threshold = (
        best_profile.threshold
        if threshold_override is None
        else threshold_override
    )
    if best_score < threshold:
        return None, float(best_score), float(second_score)
    if len(ranked) > 1 and best_score - second_score < minimum_margin:
        return None, float(best_score), float(second_score)
    return best_profile.name, float(best_score), float(second_score)


def recognise(
    observation: FaceObservation,
    profile: FaceProfile,
    threshold: float | None = None,
) -> tuple[str | None, float]:
    score = profile_similarity(observation.embedding, profile)
    effective_threshold = profile.threshold if threshold is None else threshold
    return (profile.name if score >= effective_threshold else None, score)


def face_inside_person(
    face: FaceObservation,
    person_box: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = person_box
    face_x, face_y = face.centre
    return x1 <= face_x <= x2 and y1 <= face_y <= y2
