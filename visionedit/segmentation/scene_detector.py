"""
Phase 1 — Temporal Segmentation: Scene Detection
=================================================
Wraps PySceneDetect to split a raw video into micro-scenes.

Returns a list of SceneInfo objects (without frames — those are populated
by frame_extractor.py in the next step).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

from visionedit.utils.data_types import SceneInfo


def detect_scenes(video_path: str, cfg: dict) -> List[SceneInfo]:
    """
    Split *video_path* into micro-scenes using PySceneDetect.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the input video file.
    cfg : dict
        Full pipeline config dict. Uses ``cfg["segmentation"]``:
        - ``detector``  : "content" | "adaptive" | "threshold"
        - ``threshold`` : float — scene-change sensitivity
        - ``frames_per_scene`` : int — unused here, used by frame_extractor

    Returns
    -------
    List[SceneInfo]
        Ordered list of detected scenes (no frames yet).

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If an unsupported detector name is specified.
    """
    seg_cfg = cfg["segmentation"]
    detector_name: str = seg_cfg.get("detector", "content").lower()
    threshold: float = float(seg_cfg.get("threshold", 27.0))

    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    logger.info(f"[Segmentation] Detecting scenes in: {video_path}")
    logger.debug(f"[Segmentation] Detector={detector_name}, threshold={threshold}")

    # ── Build detector ────────────────────────────────────────────────────────
    detector = _build_detector(detector_name, threshold)

    # ── Run detection ─────────────────────────────────────────────────────────
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video=video, show_progress=False)

    raw_scene_list = scene_manager.get_scene_list()

    if not raw_scene_list:
        # Treat the entire video as one scene if no cuts were found
        logger.warning(
            "[Segmentation] No scene cuts detected — treating entire video as one scene."
        )
        duration = video.duration
        raw_scene_list = [(video.base_timecode, duration)]

    scenes: List[SceneInfo] = []
    for idx, (start_tc, end_tc) in enumerate(raw_scene_list):
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        scenes.append(
            SceneInfo(index=idx, start_sec=start_sec, end_sec=end_sec)
        )

    logger.info(f"[Segmentation] Found {len(scenes)} scene(s).")
    return scenes


def _build_detector(name: str, threshold: float):
    """Instantiate the requested PySceneDetect detector."""
    if name == "content":
        return ContentDetector(threshold=threshold)
    elif name == "adaptive":
        return AdaptiveDetector(adaptive_threshold=threshold)
    elif name == "threshold":
        return ThresholdDetector(threshold=threshold)
    else:
        raise ValueError(
            f"Unsupported detector '{name}'. "
            "Choose from: 'content', 'adaptive', 'threshold'."
        )
