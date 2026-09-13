"""
Phase 2 -- Stream A: Semantic Relevance Scoring (Two-Stage Cascade OD)
=======================================================================
Two-stage object detection pipeline inspired by the literature review:

Stage 1 (always-on, every frame):
  YOLOv8-S / YOLOv10-S -- lightweight anchor-free detector (~9ms/frame).
  Justified by: Hua et al. (2025) [B1] YOLO benchmark -- YOLOv8-S/YOLOv10-S
  are the best speed-accuracy S-class models. Ramos & Sappa (2025) [B5]
  confirm anchor-free + decoupled head is the 2024 state-of-the-art.

Stage 2 (gated, candidates only):
  YOLOv9-E or YOLOv10-X -- runs only when Stage 1 top confidence
  exceeds stage2_trigger_threshold.  Provides AlignConv-style refinement
  (Yang et al., 2024 [B4]) at reduced average compute.
  Cost savings: RR = (1-alpha)*C_exp / (C_cheap+C_exp)  (Shah et al. [B3]).

Temporal persistence:
  ObjectTracker (Kalman + adaptive tau) from object_tracker.py.
  Fixes B3 self-reported limitation: tau is frame-rate-normalised, not fixed.

Decomposed salience score (Pattern 1 -- cross-domain):
  O_i = f(detection_conf, track_stability, persistence_ratio)
  Not raw YOLO confidence.  Consistent with SaIS in B4 and intensity ranking in A3.

Outputs
-------
  score()      : float  O_i in [0,1]  (backward-compat, unchanged)
  score_rich() : dict   {o_score, top_class, top_confidence, track_count,
                          stage2_used, cascade_savings_pct}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


# Module-level model cache: load once per process run, reuse across scenes.
_stage1_model = None
_stage2_model = None
_yolo_world_model = None
_last_prompt: Optional[List[str]] = None
_tracker = None


# ── Public: backward-compat float score ───────────────────────────────────────

def score(frames: np.ndarray, cfg: dict) -> float:
    """
    Compute the Object Relevance Score O_i for one clip.

    Parameters
    ----------
    frames : np.ndarray
        Sampled frames (N, H, W, 3) BGR order.
    cfg : dict
        Full pipeline config; uses cfg["streams"]["semantic"].

    Returns
    -------
    float
        O_i in [0, 1].  Returns 0.0 on any failure.
    """
    if frames is None or len(frames) == 0:
        return 0.0
    try:
        return score_rich(frames, cfg)["o_score"]
    except Exception as exc:
        logger.warning(f"[StreamA] score() failed: {exc}")
        return 0.0


# ── Public: rich dict output ───────────────────────────────────────────────────

def score_rich(frames: np.ndarray, cfg: dict) -> Dict[str, Any]:
    """
    Full two-stage cascade OD scoring for one clip.

    Returns
    -------
    dict with keys:
        o_score              : float  [0,1]  backward-compat O_i
        top_class            : str    highest-salience detected class name
        top_confidence       : float  raw YOLO confidence of top detection
        track_count          : int    number of active Kalman tracks
        stage2_used          : bool   True if expensive stage fired this clip
        cascade_savings_pct  : float  estimated % FLOP reduction vs. always-on heavy model
    """
    sem_cfg = cfg["streams"]["semantic"]
    prompt: List[str] = sem_cfg.get("prompt", [])

    _default = {
        "o_score": 0.0,
        "top_class": "none",
        "top_confidence": 0.0,
        "track_count": 0,
        "stage2_used": False,
        "cascade_savings_pct": 0.0,
    }

    if frames is None or len(frames) == 0:
        return _default

    try:
        if prompt:
            return _score_world_rich(frames, prompt, sem_cfg)
        else:
            return _score_cascade_rich(frames, sem_cfg)
    except Exception as exc:
        logger.warning(f"[StreamA] score_rich() failed: {exc}")
        return _default


# ── Two-stage cascade path ─────────────────────────────────────────────────────

def _score_cascade_rich(frames: np.ndarray, sem_cfg: dict) -> Dict[str, Any]:
    """
    Two-stage cascade: YOLOv8-S always-on -> YOLOv9-E gated.

    Stage 1 runs on every frame.  Stage 2 is triggered when Stage 1
    top confidence >= stage2_trigger_threshold.

    Compute savings (Shah et al. [B3] formula) are reported in the output.
    """
    global _stage1_model, _stage2_model, _tracker
    from ultralytics import YOLO
    from visionedit.streams.object_tracker import ObjectTracker, compute_cascade_savings
    from visionedit.streams.saliency_score import clip_salience

    # ── Config ────────────────────────────────────────────────────────────────
    stage1_model_name: str = sem_cfg.get("yolo_model", "yolov8s.pt")
    stage2_model_name: str = sem_cfg.get("yolo_model_heavy", "yolov9e.pt")
    min_conf: float = float(sem_cfg.get("min_confidence", 0.25))
    stage2_thr: float = float(sem_cfg.get("stage2_trigger_threshold", 0.60))
    fps: float = float(sem_cfg.get("video_fps", 25.0))
    tau_base: float = float(sem_cfg.get("tau_base_sec", 5.0))

    # FLOP costs for savings calculation (GFLOPs per frame, from Hua et al. B1)
    # YOLOv8-S: 28.6 GFLOPs, YOLOv9-E: 189.0 GFLOPs
    cost_cheap: float = float(sem_cfg.get("stage1_gflops", 28.6))
    cost_heavy: float = float(sem_cfg.get("stage2_gflops", 189.0))

    # ── Lazy load Stage 1 ─────────────────────────────────────────────────────
    if _stage1_model is None:
        logger.info(f"[StreamA] Loading Stage 1 model: {stage1_model_name}")
        _stage1_model = YOLO(stage1_model_name)

    # ── Lazy init tracker (frame-rate-normalised tau) ─────────────────────────
    if _tracker is None:
        _tracker = ObjectTracker(fps=fps, tau_base_sec=tau_base)

    # ── Per-frame processing ───────────────────────────────────────────────────
    frame_max_confs: List[float] = []
    stage2_fired_frames: int = 0
    top_class: str = "none"
    top_conf_overall: float = 0.0

    for frame in frames:
        detections = _run_stage1(frame, min_conf)

        # Determine top-1 confidence from Stage 1
        top1_conf = max((d["confidence"] for d in detections), default=0.0)

        # Stage 2 gating: run expensive model only on high-confidence candidates
        if top1_conf >= stage2_thr:
            stage2_det = _run_stage2(frame, min_conf, stage2_model_name)
            stage2_fired_frames += 1
            # Merge: keep highest-confidence detection per class
            det_map = {d["class_name"]: d for d in detections}
            for d in stage2_det:
                existing = det_map.get(d["class_name"])
                if existing is None or d["confidence"] > existing["confidence"]:
                    det_map[d["class_name"]] = d
            detections = list(det_map.values())

        # Update Kalman tracker with this frame's detections
        _tracker.update(detections)

        # Frame-level score: top decomposed salience from tracker
        track_scores = _tracker.compute_salience_scores()
        frame_salience = clip_salience(track_scores)
        frame_max_confs.append(frame_salience)

        # Track best class across clip
        if track_scores:
            best = max(track_scores, key=lambda s: s["object_salience"])
            if best["object_salience"] > top_conf_overall:
                top_conf_overall = best["object_salience"]
                top_class = best["class_name"]
                top_conf_overall = best["mean_confidence"]

    # ── Clip-level aggregation ─────────────────────────────────────────────────
    o_score = float(np.mean(frame_max_confs)) if frame_max_confs else 0.0

    alpha = stage2_fired_frames / max(1, len(frames))
    rr = compute_cascade_savings(alpha, cost_cheap, cost_heavy)
    savings_pct = round(rr * 100, 1)

    active_tracks = _tracker.compute_salience_scores()

    logger.debug(
        f"[StreamA] O_i={o_score:.4f} top={top_class} stage2={stage2_fired_frames}/{len(frames)} "
        f"tracks={len(active_tracks)} savings={savings_pct}%"
    )

    return {
        "o_score": o_score,
        "top_class": top_class,
        "top_confidence": float(top_conf_overall),
        "track_count": len(active_tracks),
        "stage2_used": stage2_fired_frames > 0,
        "cascade_savings_pct": savings_pct,
    }


def _run_stage1(frame: np.ndarray, min_conf: float) -> List[dict]:
    """Run Stage 1 (YOLOv8-S/YOLOv10-S) and return normalised detections."""
    global _stage1_model
    results = _stage1_model.predict(
        source=frame, conf=min_conf, verbose=False, stream=False
    )
    return _parse_yolo_results(results)


def _run_stage2(frame: np.ndarray, min_conf: float, model_name: str) -> List[dict]:
    """Run Stage 2 (YOLOv9-E class) -- lazy load on first trigger."""
    global _stage2_model
    if _stage2_model is None:
        logger.info(f"[StreamA] Loading Stage 2 model: {model_name} (first trigger)")
        from ultralytics import YOLO
        _stage2_model = YOLO(model_name)
    results = _stage2_model.predict(
        source=frame, conf=min_conf, verbose=False, stream=False
    )
    return _parse_yolo_results(results)


def _parse_yolo_results(results) -> List[dict]:
    """Convert Ultralytics YOLO result to normalised detection dicts."""
    detections = []
    if not results or results[0].boxes is None:
        return detections
    boxes = results[0].boxes
    names = results[0].names or {}
    for i in range(len(boxes)):
        bbox = boxes.xyxy[i].cpu().numpy().astype(float)
        conf = float(boxes.conf[i].item())
        cls_id = int(boxes.cls[i].item())
        cls_name = names.get(cls_id, str(cls_id))
        detections.append({
            "bbox": bbox.tolist(),
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name,
        })
    return detections


# ── YOLO-World path (zero-shot prompts) ───────────────────────────────────────

def _score_world_rich(
    frames: np.ndarray,
    prompt: List[str],
    sem_cfg: dict,
) -> Dict[str, Any]:
    """Zero-shot, open-vocabulary detection via YOLO-World."""
    global _yolo_world_model, _last_prompt

    model_name: str = sem_cfg.get("yolo_world_model", "yolov8s-world.pt")
    min_conf: float = float(sem_cfg.get("min_confidence", 0.25))

    if _yolo_world_model is None:
        logger.info(f"[StreamA] Loading YOLO-World model: {model_name}")
        from ultralytics import YOLOWorld
        _yolo_world_model = YOLOWorld(model_name)

    if prompt != _last_prompt:
        logger.debug(f"[StreamA] Setting YOLO-World classes: {prompt}")
        _yolo_world_model.set_classes(prompt)
        _last_prompt = prompt

    confidences = []
    top_class = "none"
    top_conf = 0.0

    for frame in frames:
        results = _yolo_world_model.predict(
            source=frame, conf=min_conf, verbose=False, stream=False
        )
        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes
            best_i = int(boxes.conf.argmax().item())
            c = float(boxes.conf[best_i].item())
            confidences.append(c)
            if c > top_conf:
                top_conf = c
                names = results[0].names or {}
                top_class = names.get(int(boxes.cls[best_i].item()), "object")
        else:
            confidences.append(0.0)

    o_score = float(np.mean(confidences))

    return {
        "o_score": o_score,
        "top_class": top_class,
        "top_confidence": top_conf,
        "track_count": 0,
        "stage2_used": False,
        "cascade_savings_pct": 0.0,
    }
