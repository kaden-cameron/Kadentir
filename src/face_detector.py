"""
Face detection and recognition module.
Uses MediaPipe for face landmarks and face_recognition for locations/embeddings.
"""
import os
import urllib.request
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Tuple, Optional, Any, Sequence

# MediaPipe: use tasks API (0.10+)
try:
    from mediapipe.tasks.python.vision import face_landmarker as fl
    from mediapipe.tasks.python.vision.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision.core.image import Image as MPImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    _HAS_TASKS = True
except ImportError:
    _HAS_TASKS = False

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _get_face_landmarker_model_path() -> Path:
    """Download face_landmarker.task to data/ if not present."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / "face_landmarker.task"
    if path.is_file():
        return path
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception:
        pass
    return path


class FaceDetector:
    """Handles face detection, landmark detection, and face encoding."""

    def __init__(self):
        self._face_landmarker = None
        if _HAS_TASKS:
            model_path = _get_face_landmarker_model_path()
            if model_path.is_file():
                base_options = base_options_lib.BaseOptions(model_asset_path=str(model_path))
                options = fl.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=10,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    running_mode=fl.RunningMode.IMAGE,
                )
                self._face_landmarker = fl.FaceLandmarker.create_from_options(options)

    def detect_faces(self, image: np.ndarray) -> Tuple[List, Any]:
        """
        Detect faces and extract facial landmarks.
        Returns Tuple of (face_locations, face_landmark_results).
        face_locations: list of (top, right, bottom, left).
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        results = None
        if self._face_landmarker is not None:
            try:
                mp_image = MPImage(ImageFormat.SRGB, rgb_image)
                results = self._face_landmarker.detect(mp_image)
            except Exception:
                pass
        return face_locations, results

    def detect_face_landmarks(self, image: np.ndarray) -> Any:
        """
        Run only MediaPipe face landmarker on the image.
        Use this when you already know a face is present (e.g. from detect_faces on a smaller frame)
        and want landmarks on the full-resolution image for drawing the mesh.
        Returns face_landmark_results or None.
        """
        if self._face_landmarker is None:
            return None
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(ImageFormat.SRGB, rgb_image)
            return self._face_landmarker.detect(mp_image)
        except Exception:
            return None

    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """Generate 128-dimensional face encoding. Returns None if failed."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image, [face_location])
        if encodings:
            return encodings[0]
        return None

    def draw_face_boxes(self, image: np.ndarray, face_locations: List) -> np.ndarray:
        """Draw bounding boxes around detected faces (in-place)."""
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        return image

    def _draw_connections_cv2(
        self,
        image: np.ndarray,
        face_landmark_results: Any,
        connections: Sequence[Any],
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw landmark connections (tessellation or contours) with cv2. Coords are normalized [0,1] for image."""
        if not _HAS_TASKS or face_landmark_results is None or not connections:
            return
        landmarks_per_face = getattr(face_landmark_results, "face_landmarks", None)
        if not landmarks_per_face:
            return
        h, w = image.shape[:2]
        for landmark_list in landmarks_per_face:
            if not landmark_list:
                continue
            idx_to_px = {}
            for idx, lm in enumerate(landmark_list):
                x = getattr(lm, "x", None)
                y = getattr(lm, "y", None)
                if x is None or y is None:
                    continue
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                px = int(x * (w - 1)) if w else 0
                py = int(y * (h - 1)) if h else 0
                idx_to_px[idx] = (px, py)
            for conn in connections:
                i, j = conn.start, conn.end
                if i in idx_to_px and j in idx_to_px:
                    cv2.line(
                        image,
                        idx_to_px[i],
                        idx_to_px[j],
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )

    def draw_face_landmarks(self, image: np.ndarray, face_landmark_results: Any) -> np.ndarray:
        """Draw MediaPipe face mesh (tessellation + contours) in blue, in-place."""
        if not _HAS_TASKS or face_landmark_results is None:
            return image
        if not getattr(face_landmark_results, "face_landmarks", None):
            return image
        blue_bgr = (255, 0, 0)  # Blue
        self._draw_connections_cv2(
            image,
            face_landmark_results,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            blue_bgr,
            thickness=1,
        )
        self._draw_connections_cv2(
            image,
            face_landmark_results,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            blue_bgr,
            thickness=2,
        )
        return image

    def draw_face_mesh_on_frame(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        padding: int = 30,
    ) -> None:
        """
        Crop face from frame, run MediaPipe on the crop, draw blue mesh + contours on crop, paste back.
        Guarantees mesh is drawn whenever the crop contains a face (MediaPipe sees a clear face image).
        """
        if self._face_landmarker is None:
            return
        top, right, bottom, left = face_location
        height, width = frame.shape[:2]
        t = max(0, top - padding)
        b = min(height, bottom + padding)
        l = max(0, left - padding)
        r = min(width, right + padding)
        crop = frame[t:b, l:r].copy()
        if crop.size == 0:
            return
        results = self.detect_face_landmarks(crop)
        if results is None or not getattr(results, "face_landmarks", None):
            return
        self.draw_face_landmarks(crop, results)
        frame[t:b, l:r] = crop

    def crop_face(
        self, image: np.ndarray, face_location: Tuple, padding: int = 20
    ) -> np.ndarray:
        """Crop face region with padding. face_location: (top, right, bottom, left)."""
        top, right, bottom, left = face_location
        height, width = image.shape[:2]
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)
        return image[top:bottom, left:right]

    def cleanup(self):
        """Release MediaPipe resources."""
        self._face_landmarker = None
