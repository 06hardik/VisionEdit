"""
Phase 3 -- Stream B: Affective (Emotion) Scoring
=================================================
Computes per-clip Emotion Score E_i using a custom lightweight
Facial Emotion Recognition (FER) CNN (CLCM or Mini-Xception).

Pipeline per frame
------------------
  frame_bgr
    -> face_detector.detect_faces()         # MediaPipe (+ Haar fallback)
    -> for each face crop:
        -> fer_model.FERModel.predict_with_cam()  # CLCM 48x48 -> 7 emotions + GradCAM
        -> binary expert correction (if confidence < threshold)
        -> temporal_aggregator.update()
        -> pos_score - neg_score
    -> mean across faces in frame -> frame_score

  E_i = mean(frame_scores)                  # mean over all sampled frames

Outputs
-------
  score()      : float  E_i in [-1.0, 1.0]  (backward-compat, unchanged)
  score_rich() : dict   {e_score, label, confidence, intensity,
                          expert_used, num_faces}

Intensity ranking (from Punuri et al., 2024 -- A3)
---------------------------------------------------
  Grad-CAM mean activation -> MINIMAL / AVERAGE / STRONG
  Uses EmotionTemporalAggregator.intensity_rank() with configurable thresholds.

Backends (controlled by cfg["streams"]["affective"]["backend"])
---------------------------------------------------------------
  "fer_cnn"  -- Full FER CNN (production default; supports clcm or mini_xception)
  "stub"     -- Always returns 0.0 / neutral (used in unit tests / CI)

Swapping models
---------------
Set cfg["streams"]["affective"]["backbone"] to "clcm" (recommended) or
"mini_xception" (original).  CLCM enables Grad-CAM and binary experts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


def score(frames: np.ndarray, cfg: dict) -> float:
    """
    Compute the Emotion Score E_i for one clip.

    Parameters
    ----------
    frames : np.ndarray
        Sampled frames of shape (N, H, W, 3) in BGR order.
    cfg : dict
        Full pipeline config.  Uses ``cfg["streams"]["affective"]``::

          backend           : "fer_cnn" | "stub"
          model_weights     : str   — path to .pt weights file
          face_detector     : "mediapipe" | "haar"
          min_face_confidence : float (0–1)
          face_padding_ratio  : float
          positive_emotions : list[str]
          negative_emotions : list[str]

    Returns
    -------
    float
        E_i in [-1.0, 1.0].  Returns 0.0 when backend is "stub" or no
        faces are detected.
    """
    aff_cfg = cfg["streams"]["affective"]
    backend: str = aff_cfg.get("backend", "fer_cnn").lower()

    if frames is None or len(frames) == 0:
        return 0.0

    if backend == "stub":
        return 0.0

    if backend == "fer_cnn":
        try:
            return _score_fer_cnn(frames, aff_cfg)
        except Exception as exc:
            logger.warning(f"[StreamB] Affective scoring failed: {exc}")
            return 0.0

    logger.warning(
        f"[StreamB] Unknown backend '{backend}'. "
        "Supported values: 'fer_cnn', 'stub'. Returning 0.0."
    )
    return 0.0


def score_rich(
    frames: np.ndarray,
    cfg: dict,
    clip_id: str = "clip",
) -> Dict[str, Any]:
    """
    Compute full affective scoring result for one clip.

    Returns a rich dictionary instead of a plain float -- for use by the
    Fusion Layer which needs label, intensity and confidence alongside E_i.

    Parameters
    ----------
    frames : np.ndarray
        Sampled frames (N, H, W, 3) in BGR order.
    cfg : dict
        Full pipeline config.  See score() docstring for keys used.
    clip_id : str
        Unique clip identifier used in Grad-CAM output filenames.
        E.g. "scene_007".  Ignored when gradcam.save_enabled is false.

    Returns
    -------
    dict with keys:
        e_score          : float  [-1.0, 1.0]  (backward-compat E_i)
        label            : str    dominant emotion label
        confidence       : float  top-1 CNN confidence
        intensity        : str    "MINIMAL" | "AVERAGE" | "STRONG"  (per A3)
        expert_used      : bool   True if a binary expert fired this clip
        num_faces        : int    total face detections across the clip
        gradcam_saved    : list   paths of saved Grad-CAM overlay images
    """
    aff_cfg = cfg["streams"]["affective"]
    backend: str = aff_cfg.get("backend", "fer_cnn").lower()

    _default = {
        "e_score": 0.0,
        "label": "neutral",
        "confidence": 0.0,
        "intensity": "MINIMAL",
        "expert_used": False,
        "num_faces": 0,
        "gradcam_saved": [],
    }

    if frames is None or len(frames) == 0 or backend == "stub":
        return _default

    if backend == "fer_cnn":
        try:
            return _score_fer_cnn_rich(frames, aff_cfg, clip_id=clip_id)
        except Exception as exc:
            logger.warning(f"[StreamB] score_rich() failed: {exc}")
            return _default

    logger.warning(f"[StreamB] Unknown backend '{backend}'; returning defaults.")
    return _default


# ── FER CNN backend ───────────────────────────────────────────────────────────

def _score_fer_cnn(frames: np.ndarray, aff_cfg: dict) -> float:
    """
    Emotion scoring via the FER CNN (float output -- backward compat).
    Internally calls _score_fer_cnn_rich and extracts e_score.
    """
    return _score_fer_cnn_rich(frames, aff_cfg, clip_id="clip")["e_score"]


def _score_fer_cnn_rich(
    frames: np.ndarray,
    aff_cfg: dict,
    clip_id: str = "clip",
) -> Dict[str, Any]:
    """
    Full rich emotion scoring via the FER CNN.

    For each frame:
      1. Detect faces (MediaPipe / Haar fallback).
      2. For each face: predict probabilities + Grad-CAM (CLCM backend).
      3. Optionally apply binary expert correction (disgust / fear).
      4. Update temporal aggregator.
      5. Optionally save Grad-CAM overlay to disk (gradcam.save_enabled).
      6. Compute pos_score - neg_score for E_i contribution.

    Returns the rich dict defined in score_rich().
    """
    from visionedit.streams.fer_model import get_model
    from visionedit.streams.face_detector import detect_faces
    from visionedit.streams.expert_model import get_expert_ensemble
    from visionedit.streams.temporal_aggregator import get_aggregator

    backbone: str = aff_cfg.get("backbone", "mini_xception")
    positive: List[str] = [e.lower() for e in aff_cfg.get("positive_emotions", ["happy", "surprise"])]
    negative: List[str] = [e.lower() for e in aff_cfg.get("negative_emotions", [])]
    weights_path: Optional[str] = aff_cfg.get(
        "clcm_weights" if backbone == "clcm" else "model_weights"
    )
    min_face_conf: float = float(aff_cfg.get("min_face_confidence", 0.5))
    padding: float = float(aff_cfg.get("face_padding_ratio", 0.15))
    trigger_thr: float = float(aff_cfg.get("expert_trigger_threshold", 0.60))
    temporal_window: int = int(aff_cfg.get("temporal_window", 10))
    lstm_weights: Optional[str] = aff_cfg.get("lstm_weights")
    intensity_cfg = aff_cfg.get("intensity_thresholds", {})
    intensity_min: float = float(intensity_cfg.get("minimal", 0.40))
    intensity_str: float = float(intensity_cfg.get("strong", 0.70))

    expert_weights = aff_cfg.get("expert_weights", {})

    # ── Grad-CAM visualizer config ─────────────────────────────────────────────
    gradcam_cfg = aff_cfg.get("gradcam", {})
    gradcam_enabled: bool = bool(gradcam_cfg.get("save_enabled", False))
    gradcam_output_dir: str = gradcam_cfg.get("output_dir", "outputs/gradcam")
    gradcam_max_saves: int = int(gradcam_cfg.get("max_saves_per_clip", 3))
    gradcam_target_size: tuple = tuple(
        gradcam_cfg.get("target_size", [224, 224])
    )

    fer = get_model(weights_path=weights_path, backbone=backbone)
    experts = get_expert_ensemble(
        disgust_weights=expert_weights.get("disgust"),
        fear_weights=expert_weights.get("fear"),
    )
    aggregator = get_aggregator(
        window=temporal_window,
        lstm_weights=lstm_weights,
        intensity_minimal=intensity_min,
        intensity_strong=intensity_str,
    )

    frame_scores: List[float] = []
    total_faces: int = 0
    any_expert_used: bool = False
    # Heatmap data collector: list of dicts for gradcam_visualizer
    heatmap_data: List[Dict] = []

    for frame_idx, frame in enumerate(frames):
        face_crops = detect_faces(
            frame_bgr=frame,
            padding_ratio=padding,
            min_confidence=min_face_conf,
        )
        if not face_crops:
            frame_scores.append(0.0)
            continue

        total_faces += len(face_crops)
        face_scores: List[float] = []

        for crop_idx, face_bgr in enumerate(face_crops):
            face_id = f"face_{crop_idx}"

            # Primary prediction (with Grad-CAM if CLCM backbone)
            probs, heatmap, cam_score = fer.predict_with_cam(face_bgr)

            # Binary expert correction for disgust / fear
            probs, expert_fired = _apply_binary_experts(
                experts=experts,
                probs=probs,
                face_bgr=face_bgr,
                trigger_threshold=trigger_thr,
            )
            if expert_fired:
                any_expert_used = True

            # Temporal aggregator update
            aggregator.update(face_id=face_id, probs=probs, cam_score=cam_score)

            # Collect heatmap data for deferred saving (avoids blocking the loop)
            if gradcam_enabled and heatmap is not None:
                clip_result_now = aggregator.get_clip_score(face_id)
                heatmap_data.append({
                    "face_bgr": face_bgr.copy(),
                    "heatmap": heatmap.copy() if heatmap is not None else None,
                    "label": max(probs, key=probs.get),
                    "confidence": float(max(probs.values())),
                    "intensity": clip_result_now["intensity"],
                    "expert_used": expert_fired,
                    "cam_score": float(cam_score),
                    "frame_idx": frame_idx,
                    "face_idx": crop_idx,
                })

            # Frame-level E_i contribution
            pos = sum(probs.get(e, 0.0) for e in positive)
            neg = sum(probs.get(e, 0.0) for e in negative)
            face_scores.append(pos - neg)

        frame_scores.append(float(np.mean(face_scores)))

    e_i = float(np.mean(frame_scores)) if frame_scores else 0.0

    # Get clip-level label from temporal aggregator (face_0 = first/primary face)
    clip_result = aggregator.get_clip_score("face_0")

    # ── Grad-CAM saving (deferred, top-N by cam_score) ────────────────────────
    gradcam_saved: List[str] = []
    if gradcam_enabled and heatmap_data:
        try:
            from visionedit.streams.gradcam_visualizer import save_clip_heatmaps
            gradcam_saved = save_clip_heatmaps(
                clip_id=clip_id,
                frame_heatmap_data=heatmap_data,
                output_dir=gradcam_output_dir,
                max_saves=gradcam_max_saves,
                target_size=gradcam_target_size,
            )
        except Exception as exc:
            logger.warning(f"[GradCAM] Save failed for clip {clip_id!r}: {exc}")

    logger.debug(
        f"[StreamB] E_i={e_i:.4f} label={clip_result['label']} "
        f"intensity={clip_result['intensity']} expert={any_expert_used} "
        f"faces={total_faces} frames={len(frame_scores)} "
        f"gradcam_saved={len(gradcam_saved)}"
    )

    return {
        "e_score": e_i,
        "label": clip_result["label"],
        "confidence": clip_result["confidence"],
        "intensity": clip_result["intensity"],
        "expert_used": any_expert_used,
        "num_faces": total_faces,
        "gradcam_saved": gradcam_saved,
    }


def _apply_binary_experts(
    experts,
    probs: Dict[str, float],
    face_bgr,
    trigger_threshold: float,
) -> tuple:
    """
    Wrapper: apply ExpertEnsemble correction, returning (probs, expert_used).
    Gracefully returns (probs, False) if experts are not available.
    """
    try:
        return experts.apply(
            primary_probs=probs,
            face_bgr=face_bgr,
            trigger_threshold=trigger_threshold,
        )
    except Exception as exc:
        logger.debug(f"[StreamB] Expert correction skipped: {exc}")
        return probs, False


def _score_frame(
    frame: np.ndarray,
    fer,
    detect_fn,
    positive: List[str],
    negative: List[str],
    min_face_conf: float,
    padding: float,
) -> float:
    """
    Score a single BGR frame.  Returns 0.0 if no faces are detected.
    Kept for backward compatibility -- used internally by the old _score_fer_cnn.
    """
    face_crops = detect_fn(
        frame_bgr=frame,
        padding_ratio=padding,
        min_confidence=min_face_conf,
    )

    if not face_crops:
        return 0.0

    # Batch inference across all detected faces in this frame
    probs_list = fer.predict_batch(face_crops)

    face_scores: List[float] = []
    for probs in probs_list:
        pos_score = sum(probs.get(e, 0.0) for e in positive)
        neg_score = sum(probs.get(e, 0.0) for e in negative)
        face_scores.append(pos_score - neg_score)

    return float(np.mean(face_scores)) if face_scores else 0.0
