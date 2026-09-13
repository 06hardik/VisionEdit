"""
Stream B (OD) -- Kalman Object Tracker with Adaptive Persistence
================================================================
Implements per-object tracking with temporal persistence scoring for the
Object/Event Detection stream.

Motivation
----------
Shah et al. (2026) [B3] "A Two-Stage Spatiotemporal CNN-YOLOv9 Framework"
uses a Kalman filter + fixed threshold tau=75 frames to declare objects
"abandoned" (i.e., persistently present).  This module:

  1. Reproduces the Kalman + persistence pattern.
  2. Fixes the key limitation self-reported by B3: tau is a hard-coded
     constant not adaptive to frame rate, camera distance, or scene context.
     Here tau is computed as:
         tau_eff = tau_base_sec * fps
     At minimum this normalises for frame rate (B3 assumes 15 FPS, tau=75=5s).

  3. Produces a decomposed track-stability score (track_stability in [0,1])
     alongside persistence, consistent with Pattern 1 from the literature
     synthesis: decomposed quality beats a flat scalar.

References
----------
- Shah et al. (2026) [B3] -- Kalman + fixed-tau cascade template
- Miri Rekavandi et al. (2025) [B2] -- video-based temporal persistence gap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger


TAU_BASE_SECONDS: float = 5.0
MAX_MISSING_FRAMES: int = 10
IOU_THRESHOLD: float = 0.30


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    missed: int = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def centroid(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return cx, cy

    @property
    def track_stability(self) -> float:
        if self.age == 0:
            return 0.0
        return min(1.0, self.hits / self.age)

    @property
    def mean_confidence(self) -> float:
        if not self.confidence_history:
            return self.confidence
        return float(np.mean(list(self.confidence_history)))

    @property
    def motion_magnitude(self) -> float:
        hist = list(self.centroid_history)
        if len(hist) < 2:
            return 0.0
        displacements = [
            np.sqrt((hist[i][0]-hist[i-1][0])**2 + (hist[i][1]-hist[i-1][1])**2)
            for i in range(1, len(hist))
        ]
        return float(np.mean(displacements))


class _KalmanBox:
    """
    Constant-velocity Kalman filter for a single bbox [cx,cy,w,h].
    Adapted from SORT (Bewley et al., 2016) -- same foundation as B3.
    """

    def __init__(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        self.F = np.eye(8)
        for i in range(4): self.F[i, i+4] = 1.0
        self.H = np.zeros((4, 8)); self.H[:4, :4] = np.eye(4)
        self.Q = np.eye(8) * 1e-2
        self.R = np.eye(4) * 1e-1
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.])
        self.P = np.eye(8); self.P[4:, 4:] *= 1e3

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        cx, cy, w, h = self.x[:4]
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    def update(self, bbox: np.ndarray):
        cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
        w=bbox[2]-bbox[0]; h=bbox[3]-bbox[1]
        z = np.array([cx, cy, w, h])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _greedy_match(tracks, detections, iou_threshold):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    iou_matrix = np.zeros((len(tracks), len(detections)))
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            iou_matrix[ti, di] = _iou(t.bbox, np.array(d["bbox"]))
    matched = []
    used_t, used_d = set(), set()
    pairs = sorted(
        [(iou_matrix[t,d], t, d)
         for t in range(len(tracks)) for d in range(len(detections))],
        reverse=True)
    for iou_val, ti, di in pairs:
        if iou_val < iou_threshold: break
        if ti in used_t or di in used_d: continue
        matched.append((ti, di)); used_t.add(ti); used_d.add(di)
    return matched, [t for t in range(len(tracks)) if t not in used_t],                     [d for d in range(len(detections)) if d not in used_d]


class ObjectTracker:
    """
    IoU-based Kalman tracker with adaptive persistence threshold.

    Key improvement over Shah et al. (2026) [B3]:
      tau_eff_frames = tau_base_sec * fps
    This normalises tau for frame rate, fixing B3's hard-coded tau=75 @ 15fps.
    """

    def __init__(self, fps=25.0, tau_base_sec=TAU_BASE_SECONDS,
                 max_missing=MAX_MISSING_FRAMES, iou_threshold=IOU_THRESHOLD):
        self.fps = fps
        self.tau_base_sec = tau_base_sec
        self.tau_eff_frames = max(1, int(tau_base_sec * fps))
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self._tracks: List[Track] = []
        self._kalman: Dict[int, _KalmanBox] = {}
        self._next_id = 0
        self._frame_count = 0
        logger.debug(
            f"[Tracker] fps={fps}, tau={tau_base_sec}s = {self.tau_eff_frames} frames "
            f"(adaptive tau -- fixes B3 fixed tau=75 @ 15fps)")

    def update(self, detections: List[dict]) -> List[Track]:
        self._frame_count += 1
        for t in self._tracks:
            if t.track_id in self._kalman:
                t.bbox = self._kalman[t.track_id].predict()
        matched, unmatched_t, unmatched_d = _greedy_match(
            self._tracks, detections, self.iou_threshold)
        for ti, di in matched:
            t = self._tracks[ti]; d = detections[di]
            bbox = np.array(d["bbox"], dtype=float)
            if t.track_id in self._kalman:
                self._kalman[t.track_id].update(bbox)
                t.bbox = self._kalman[t.track_id].get_state()
            else:
                t.bbox = bbox
            t.confidence = d["confidence"]; t.hits += 1; t.missed = 0; t.age += 1
            t.confidence_history.append(d["confidence"])
            t.centroid_history.append(t.centroid)
        for ti in unmatched_t:
            self._tracks[ti].missed += 1; self._tracks[ti].age += 1
        for di in unmatched_d:
            d = detections[di]; bbox = np.array(d["bbox"], dtype=float)
            nt = Track(track_id=self._next_id, bbox=bbox,
                       class_id=d.get("class_id",-1),
                       class_name=d.get("class_name","unknown"),
                       confidence=d["confidence"], age=1, hits=1)
            nt.confidence_history.append(d["confidence"])
            nt.centroid_history.append(nt.centroid)
            self._kalman[self._next_id] = _KalmanBox(bbox)
            self._tracks.append(nt); self._next_id += 1
        alive = [t for t in self._tracks if t.missed <= self.max_missing]
        for t in self._tracks:
            if t.missed > self.max_missing:
                self._kalman.pop(t.track_id, None)
        self._tracks = alive
        return list(self._tracks)

    def compute_salience_scores(self) -> List[dict]:
        """
        Decomposed salience score per track.
        S = 0.5*mean_conf + 0.3*track_stability + 0.2*persistence_ratio
        Consistent with Pattern 1 (decomposed quality beats flat scalar).
        """
        out = []
        for t in self._tracks:
            pr = min(1.0, t.age / max(1, self.tau_eff_frames))
            salience = 0.50*t.mean_confidence + 0.30*t.track_stability + 0.20*pr
            out.append({
                "track_id": t.track_id,
                "class_id": t.class_id,
                "class_name": t.class_name,
                "mean_confidence": t.mean_confidence,
                "track_stability": t.track_stability,
                "persistence_ratio": pr,
                "object_salience": float(np.clip(salience, 0., 1.)),
                "age_frames": t.age,
                "motion_magnitude": t.motion_magnitude,
                "bbox": t.bbox.tolist(),
            })
        return out

    def top_salience(self) -> float:
        scores = self.compute_salience_scores()
        return max((s["object_salience"] for s in scores), default=0.0)

    def reset(self):
        self._tracks.clear(); self._kalman.clear(); self._frame_count = 0


def compute_cascade_savings(alpha: float, cost_cheap: float, cost_expensive: float) -> float:
    """
    Reduction ratio from Shah et al. (2026) [B3]:
        RR = (1 - alpha) * C_expensive / (C_cheap + C_expensive)
    alpha = fraction of frames triggering expensive stage.
    Example: alpha=0.20, C_cheap=28.6 GFLOPs (YOLOv8-S), C_expensive=189.0 (YOLOv9-E)
      -> RR = 0.80 * 189 / (28.6 + 189) = 0.694  (~69% savings)
    """
    return (1.0 - alpha) * cost_expensive / (cost_cheap + cost_expensive)
