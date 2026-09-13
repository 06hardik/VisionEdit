"""
Face Detector — MediaPipe-based face cropping
=============================================
Detects faces in a BGR frame and returns cropped regions suitable for
passing to the FER CNN (fer_model.FERModel.predict).

Uses MediaPipe Face Detection as the primary backend (more accurate than
OpenCV Haar Cascades for varied poses/lighting). Falls back gracefully
if MediaPipe is not installed — returns empty list with a logged warning.

Usage
-----
    from visionedit.streams.face_detector import detect_faces

    crops = detect_faces(frame_bgr, padding_ratio=0.15, min_confidence=0.5)
    for crop in crops:
        probs = fer_model.predict(crop)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

# Detection backends
_MEDIAPIPE_AVAILABLE: bool = False
_HAAR_AVAILABLE: bool = False
_mp_face_detection = None          # lazy-loaded module
_mp_detector_instance = None       # shared across calls (avoid re-init overhead)

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning(
        "[FaceDetector] mediapipe not installed. "
        "Falling back to OpenCV Haar Cascade. "
        "Install with: pip install mediapipe"
    )

# Check Haar availability (may be missing in opencv-headless builds)
try:
    import cv2 as _cv2_check
    _cv2_check.CascadeClassifier  # raises AttributeError if headless build lacks it
    _HAAR_AVAILABLE = True
except AttributeError:
    logger.warning(
        "[FaceDetector] cv2.CascadeClassifier not available (opencv-headless build). "
        "Install mediapipe for face detection: pip install mediapipe"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def detect_faces(
    frame_bgr: np.ndarray,
    padding_ratio: float = 0.15,
    min_confidence: float = 0.5,
) -> List[np.ndarray]:
    """
    Detect all faces in a BGR frame and return cropped face regions.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input frame in BGR colour order, shape (H, W, 3).
    padding_ratio : float
        Fraction of bounding box size added as padding on each side.
        0.15 = 15% padding → captures chin and forehead.
    min_confidence : float
        Minimum MediaPipe detection confidence to accept a face.

    Returns
    -------
    list of np.ndarray
        List of BGR face crops (variable sizes).
        Empty list if no faces detected or detector unavailable.
    """
    if frame_bgr is None or frame_bgr.ndim < 2:
        return []

    if _MEDIAPIPE_AVAILABLE:
        return _detect_mediapipe(frame_bgr, padding_ratio, min_confidence)
    elif _HAAR_AVAILABLE:
        return _detect_haar(frame_bgr, padding_ratio)
    else:
        # Neither backend available — return empty list gracefully
        return []


def detect_faces_batch(
    frames: np.ndarray,
    padding_ratio: float = 0.15,
    min_confidence: float = 0.5,
) -> List[List[np.ndarray]]:
    """
    Detect faces in each frame of an (N, H, W, 3) array.

    Returns a list of length N, where each element is the list of face
    crops detected in that frame (may be empty).
    """
    results = []
    for frame in frames:
        results.append(detect_faces(frame, padding_ratio, min_confidence))
    return results


# ── MediaPipe backend ─────────────────────────────────────────────────────────

def _get_mp_detector(min_confidence: float):
    """Lazy-initialise and cache the MediaPipe FaceDetection instance."""
    global _mp_detector_instance, _mp_face_detection

    # Re-create if confidence threshold has changed (rare edge case)
    if _mp_detector_instance is None:
        import mediapipe as mp
        _mp_face_detection = mp.solutions.face_detection
        _mp_detector_instance = _mp_face_detection.FaceDetection(
            model_selection=0,              # 0 = short-range (<2m), 1 = full-range
            min_detection_confidence=min_confidence,
        )
        logger.debug("[FaceDetector] MediaPipe FaceDetection initialised.")

    return _mp_detector_instance


def _detect_mediapipe(
    frame_bgr: np.ndarray,
    padding_ratio: float,
    min_confidence: float,
) -> List[np.ndarray]:
    """Run MediaPipe Face Detection and return cropped face arrays."""
    detector = _get_mp_detector(min_confidence)

    h, w = frame_bgr.shape[:2]

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(frame_rgb)

    if not results.detections:
        return []

    crops = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        # Convert relative → absolute pixel coords
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        crop = _extract_padded_crop(frame_bgr, x1, y1, bw, bh, padding_ratio)
        if crop is not None:
            crops.append(crop)

    return crops


# ── OpenCV Haar Cascade fallback ──────────────────────────────────────────────

_haar_cascade = None


def _get_haar_cascade():
    """Lazy-load the Haar cascade classifier."""
    global _haar_cascade
    if _haar_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
        logger.debug("[FaceDetector] Haar cascade loaded as fallback.")
    return _haar_cascade
def _detect_haar(frame_bgr: np.ndarray, padding_ratio: float) -> List[np.ndarray]:
    """Fallback Haar Cascade face detection (OpenCV, no extra install needed)."""
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return []

    crops = []
    for (x, y, bw, bh) in faces:
        crop = _extract_padded_crop(frame_bgr, x, y, bw, bh, padding_ratio)
        if crop is not None:
            crops.append(crop)
    return crops


# ── Shared crop utility ───────────────────────────────────────────────────────

def _extract_padded_crop(
    frame: np.ndarray,
    x: int, y: int, bw: int, bh: int,
    padding_ratio: float,
) -> np.ndarray | None:
    """
    Extract a padded bounding-box crop from the frame.

    Returns None if the resulting crop would be degenerate (0-area).
    """
    h, w = frame.shape[:2]
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]
