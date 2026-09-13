"""
Phase 4 — Stream C: Technical Quality Scoring
==============================================
Computes two signals per clip:

  Q_i  — Blur quality: mean Laplacian variance across sampled frames.
          Higher = sharper / better focused.
          The quality gate: passes_gate = (Q_i >= θ_blur).

  M_i  — Motion energy: mean absolute frame-difference pixel magnitude,
          normalised to [0, 1] by clamping at a max expected diff of 50.

Formula references (PRD §9):
  - Quality gate: 𝟙(Q_i ≥ θ_blur)
  - M_i: frame differencing (absolute mean pixel delta between consecutive frames)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from loguru import logger


def score(frames: np.ndarray, cfg: dict) -> Tuple[float, float, bool]:
    """
    Compute technical quality metrics for one clip.

    Parameters
    ----------
    frames : np.ndarray
        Sampled frames of shape (N, H, W, 3) in BGR order.
    cfg : dict
        Full pipeline config. Uses ``cfg["streams"]["quality"]``:
        - ``blur_threshold`` : float — Laplacian variance θ_blur

    Returns
    -------
    Tuple[float, float, bool]
        ``(Q_i, M_i, passes_gate)`` where:
        - Q_i           — mean Laplacian variance (higher = sharper)
        - M_i           — normalised motion energy in [0, 1]
        - passes_gate   — True if Q_i >= blur_threshold
    """
    blur_threshold: float = float(
        cfg["streams"]["quality"].get("blur_threshold", 100.0)
    )

    if frames is None or len(frames) == 0:
        return 0.0, 0.0, False

    try:
        Q_i = _compute_blur(frames)
        M_i = _compute_motion(frames)
        passes_gate = Q_i >= blur_threshold

        logger.debug(
            f"[StreamC] Q_i={Q_i:.1f} (gate={'PASS' if passes_gate else 'FAIL'}), "
            f"M_i={M_i:.3f}"
        )
        return Q_i, M_i, passes_gate

    except Exception as exc:
        logger.warning(f"[StreamC] Quality scoring failed: {exc}")
        return 0.0, 0.0, False


# ── Laplacian variance blur detection ─────────────────────────────────────────

def _compute_blur(frames: np.ndarray) -> float:
    """
    Compute the mean Laplacian variance across all frames.

    A higher value indicates a sharper (better focused) clip.
    """
    variances = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        variances.append(lap_var)
    return float(np.mean(variances))


# ── Frame differencing motion energy ──────────────────────────────────────────

# Maximum expected mean absolute pixel difference for normalisation.
# Pure noise / static ~0–5; fast motion scene ~30–60.
_MAX_EXPECTED_DIFF: float = 50.0


def _compute_motion(frames: np.ndarray) -> float:
    """
    Compute normalised motion energy via frame differencing.

    Returns 0.0 if there is only one frame.
    """
    if len(frames) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i - 1], frames[i])
        mean_diff = float(diff.mean())
        diffs.append(mean_diff)

    raw_motion = float(np.mean(diffs))
    # Clamp and normalise to [0, 1]
    normalised = min(raw_motion / _MAX_EXPECTED_DIFF, 1.0)
    return normalised
