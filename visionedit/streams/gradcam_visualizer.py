"""
Stream B -- Grad-CAM Heatmap Visualizer
========================================
Saves per-face, per-clip Grad-CAM overlay images to disk.

Motivation
----------
Punuri et al. (2024) [A3] "Decoding Human Facial Emotions: A Ranking
Approach Using Explainable AI" uses Layer-wise Relevance Propagation (LRP)
heatmaps to show *which facial pixels drove the prediction*.  Our
implementation replaces LRP with Grad-CAM (faster, works on all predictions
including misclassified/ambiguous frames -- the key LRP limitation).

Output format per heatmap
-------------------------
  <output_dir>/<clip_id>/frame<F>_face<N>_<emotion>_<intensity>.png

Each image is the face crop with a semi-transparent JET colormap overlay
showing activated regions, plus an annotation bar:
  - Emotion label + confidence (top-1)
  - Intensity rank: MINIMAL / AVERAGE / STRONG  (Punuri et al. A3 system)
  - Expert flag: "[EX]" if binary expert corrected this prediction

Design principles
-----------------
- Saving is optional and gated by config: gradcam.save_enabled
- Saving only runs on frames selected by the temporal aggregator as peak
  frames (highest cam_score in the window) to keep disk usage bounded
- Gracefully no-ops if cv2/PIL unavailable or output_dir not set
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger


# Colour palette for intensity badge
_INTENSITY_COLOURS = {
    "MINIMAL": (180, 180, 180),   # grey
    "AVERAGE": (255, 170,  30),   # amber
    "STRONG":  (255,  55,  55),   # red
}

_COLOURMAP_JET = None   # lazy-loaded cv2 constant


def _jet_colormap() -> int:
    import cv2
    return cv2.COLORMAP_JET


def create_overlay(
    face_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over a BGR face crop.

    Parameters
    ----------
    face_bgr : np.ndarray
        BGR face crop (H, W, 3), any resolution.
    heatmap : np.ndarray
        Float32 heatmap in [0, 1], shape (H_h, W_h).  Will be resized to
        match face_bgr resolution.
    alpha : float
        Heatmap opacity.  0.0 = invisible, 1.0 = opaque.

    Returns
    -------
    np.ndarray
        Blended BGR image, same shape as face_bgr.
    """
    import cv2

    h, w = face_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_uint8 = (hm_resized * 255).clip(0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(hm_uint8, _jet_colormap())
    return cv2.addWeighted(face_bgr, 1.0 - alpha, coloured, alpha, 0)


def annotate_overlay(
    overlay: np.ndarray,
    label: str,
    confidence: float,
    intensity: str,
    expert_used: bool = False,
    cam_score: float = 0.0,
) -> np.ndarray:
    """
    Add a text annotation bar at the bottom of an overlay image.

    Shows: emotion label, confidence %, intensity badge, expert flag.
    """
    import cv2

    out = overlay.copy()
    h, w = out.shape[:2]
    bar_h = max(28, int(h * 0.15))

    # Dark semi-transparent bar at bottom
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    out[-bar_h:] = cv2.addWeighted(out[-bar_h:], 0.3, bar, 0.7, 0)

    # Intensity badge colour
    badge_colour = _INTENSITY_COLOURS.get(intensity, (200, 200, 200))

    # Main text: "happy  82%  [AVERAGE]  [EX]"
    expert_flag = "  [EX]" if expert_used else ""
    text = f"{label}  {confidence*100:.0f}%  [{intensity}]{expert_flag}"

    font_scale = max(0.3, bar_h / 60.0)
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(4, (w - tw) // 2)
    ty = h - bar_h + th + 4

    # Shadow
    cv2.putText(out, text, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text in badge colour
    cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, badge_colour, thickness, cv2.LINE_AA)

    # Small cam-score dot in top-right corner
    dot_radius = max(4, int(cam_score * 12))
    cv2.circle(out, (w - 8, 8), dot_radius, badge_colour, -1)

    return out


def save_heatmap_overlay(
    face_bgr: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    label: str,
    confidence: float,
    intensity: str,
    expert_used: bool = False,
    cam_score: float = 0.0,
    target_size: tuple = (224, 224),
) -> str | None:
    """
    Create and save a single Grad-CAM overlay image.

    Parameters
    ----------
    face_bgr : np.ndarray
        Original face crop (BGR).
    heatmap : np.ndarray
        Grad-CAM heatmap float32 [0,1].
    output_path : str or Path
        Full destination file path (.png).
    label : str
        Predicted emotion label.
    confidence : float
        Top-1 prediction confidence.
    intensity : str
        "MINIMAL" | "AVERAGE" | "STRONG"
    expert_used : bool
        Whether a binary expert corrected this prediction.
    cam_score : float
        Raw Grad-CAM mean activation (used for dot indicator).
    target_size : tuple
        (width, height) of saved image. Default 224x224.

    Returns
    -------
    str | None
        Saved file path, or None if saving failed.
    """
    try:
        import cv2

        # Upscale face crop for readability
        face_up = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        hm_up = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

        overlay = create_overlay(face_up, hm_up, alpha=0.45)
        annotated = annotate_overlay(
            overlay, label, confidence, intensity, expert_used, cam_score
        )

        # Also draw a side-by-side: original | heatmap | overlay
        hm_coloured = cv2.applyColorMap(
            (hm_up * 255).clip(0, 255).astype(np.uint8), _jet_colormap()
        )
        composite = np.concatenate([face_up, hm_coloured, annotated], axis=1)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), composite)
        return str(out_path)

    except Exception as exc:
        logger.warning(f"[GradCAM] Failed to save overlay to {output_path}: {exc}")
        return None


def save_clip_heatmaps(
    clip_id: str,
    frame_heatmap_data: list,
    output_dir: str | Path,
    max_saves: int = 3,
    target_size: tuple = (224, 224),
) -> list:
    """
    Save the top-N Grad-CAM overlays for a clip (by cam_score).

    Only saves the frames with highest cam_score to keep disk usage bounded.
    Frames where heatmap is None (mini_xception backend) are skipped silently.

    Parameters
    ----------
    clip_id : str
        Identifier for the clip, used in filenames (e.g. "scene_007").
    frame_heatmap_data : list of dict
        Each dict must have keys:
          face_bgr, heatmap, label, confidence, intensity,
          expert_used, cam_score, frame_idx, face_idx
    output_dir : str or Path
        Root directory; images saved under output_dir/clip_id/
    max_saves : int
        Maximum overlays to save per clip (to limit disk use).
    target_size : tuple
        Pixel dimensions of each saved image.

    Returns
    -------
    list of str
        Paths of saved files.
    """
    if not frame_heatmap_data:
        return []

    # Filter out entries with no heatmap (mini_xception backend)
    valid = [d for d in frame_heatmap_data if d.get("heatmap") is not None]
    if not valid:
        return []

    # Sort by cam_score descending, take top max_saves
    top_frames = sorted(valid, key=lambda d: d["cam_score"], reverse=True)[:max_saves]

    saved = []
    for entry in top_frames:
        fname = (
            f"frame{entry['frame_idx']:04d}"
            f"_face{entry['face_idx']}"
            f"_{entry['label']}"
            f"_{entry['intensity']}.png"
        )
        out_path = Path(output_dir) / clip_id / fname
        result = save_heatmap_overlay(
            face_bgr=entry["face_bgr"],
            heatmap=entry["heatmap"],
            output_path=out_path,
            label=entry["label"],
            confidence=entry["confidence"],
            intensity=entry["intensity"],
            expert_used=entry.get("expert_used", False),
            cam_score=entry["cam_score"],
            target_size=target_size,
        )
        if result:
            saved.append(result)
            logger.debug(f"[GradCAM] Saved: {result}")

    if saved:
        logger.info(f"[GradCAM] Clip {clip_id}: saved {len(saved)} overlay(s) in {Path(output_dir)/clip_id}")
    return saved
