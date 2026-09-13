"""
Stream A (OD) -- Decomposed Object Salience + SaLA Utilities
============================================================
Provides:
  1. SaIS (Shape-aware IoU Score) -- training-side, inference-cost neutral.
     Adapted from Yang et al. (2024) [B4] A2Net.
  2. Decomposed inference-time salience scoring.
     Pattern 1: decomposed quality beats flat scalar (confirmed in both
     FER [A3 LRP intensity, A5 AU] and OD [B4 SaIS, B2 NWD/CIoU] literature).
"""

from __future__ import annotations

from typing import List
import numpy as np


def _aspect(b: np.ndarray) -> float:
    w = max(abs(b[2] - b[0]), 1e-6)
    h = max(abs(b[3] - b[1]), 1e-6)
    return w / h


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Standard IoU between two [x1,y1,x2,y2] boxes."""
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def shape_score(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """
    Shape similarity score in [0,1] based on aspect-ratio match.
    From Yang et al. (2024) [B4] SaLA.
    A score of 1.0 = identical aspect ratios.  0.0 = completely different.
    """
    ar_p = _aspect(pred_bbox)
    ar_g = _aspect(gt_bbox)
    return min(ar_p, ar_g) / max(ar_p, ar_g)


def sais(pred_bbox: np.ndarray, gt_bbox: np.ndarray, shape_weight: float = 0.5) -> float:
    """
    Shape-aware IoU Score (SaIS) = IoU + shape_weight * shape_score.

    From Yang et al. (2024) [B4]: using SaIS in the label assignment cost
    function improves mAP by +2.19 points at ZERO inference cost (training only).
    This is the key reason to use SaLA: free accuracy gain.
    """
    return iou(pred_bbox, gt_bbox) + shape_weight * shape_score(pred_bbox, gt_bbox)


def object_salience(
    detection_conf: float,
    track_stability: float,
    persistence_ratio: float,
    w_conf: float = 0.50,
    w_stability: float = 0.30,
    w_persistence: float = 0.20,
) -> float:
    """
    Decomposed object salience (Pattern 1 -- cross-domain finding).

    Replaces raw YOLO confidence with a multi-component quality score:
        S = w_conf * conf + w_stability * stability + w_persistence * persistence
    """
    s = w_conf*detection_conf + w_stability*track_stability + w_persistence*persistence_ratio
    return float(np.clip(s, 0.0, 1.0))


def clip_salience(track_scores: List[dict]) -> float:
    """Max object salience across all tracks in a clip."""
    if not track_scores:
        return 0.0
    return max(s["object_salience"] for s in track_scores)
