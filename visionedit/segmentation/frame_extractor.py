"""
Phase 1 — Temporal Segmentation: Frame Extraction
==================================================
Extracts a fixed number of uniformly-sampled frames from each micro-scene
using Decord (fast GPU/CPU video reader), with an OpenCV fallback.
"""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger

from visionedit.utils.data_types import SceneInfo


def extract_frames(
    video_path: str,
    scenes: List[SceneInfo],
    cfg: dict,
) -> List[SceneInfo]:
    """
    Populate the ``frames`` field of each :class:`SceneInfo` with
    uniformly-sampled frames from the video.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    scenes : List[SceneInfo]
        Scene list produced by :func:`scene_detector.detect_scenes`.
        Modified **in-place** (frames field is filled).
    cfg : dict
        Full pipeline config. Uses ``cfg["segmentation"]["frames_per_scene"]``.

    Returns
    -------
    List[SceneInfo]
        Same list with ``frames`` populated.
    """
    n_frames: int = int(cfg["segmentation"].get("frames_per_scene", 16))
    logger.info(f"[FrameExtractor] Extracting {n_frames} frames/scene "
                f"from {len(scenes)} scenes.")

    try:
        import decord  # noqa: F401
        _extract_decord(video_path, scenes, n_frames)
    except ImportError:
        logger.warning(
            "[FrameExtractor] Decord not available — falling back to OpenCV."
        )
        _extract_opencv(video_path, scenes, n_frames)

    return scenes


# ── Decord implementation ──────────────────────────────────────────────────────

def _extract_decord(video_path: str, scenes: List[SceneInfo], n_frames: int) -> None:
    """Use decord.VideoReader for fast, seekable frame extraction."""
    import decord
    decord.bridge.set_bridge("numpy")

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    logger.debug(f"[FrameExtractor/Decord] fps={fps:.2f}, total_frames={total_frames}")

    for scene in scenes:
        start_frame = int(scene.start_sec * fps)
        end_frame = int(scene.end_sec * fps)
        end_frame = min(end_frame, total_frames - 1)
        start_frame = max(start_frame, 0)

        if end_frame <= start_frame:
            # Degenerate scene — duplicate the single frame
            indices = [start_frame] * n_frames
        else:
            indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int).tolist()

        # decord returns RGB; we convert to BGR for consistency with OpenCV pipeline
        frames_rgb = vr.get_batch(indices).asnumpy()   # (N, H, W, 3) uint8 RGB
        frames_bgr = frames_rgb[:, :, :, ::-1].copy()  # RGB → BGR
        scene.frames = frames_bgr

    logger.debug("[FrameExtractor/Decord] Done.")


# ── OpenCV fallback ────────────────────────────────────────────────────────────

def _extract_opencv(video_path: str, scenes: List[SceneInfo], n_frames: int) -> None:
    """OpenCV-based fallback frame extractor (slower, no GPU support)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[FrameExtractor/OpenCV] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.debug(f"[FrameExtractor/OpenCV] fps={fps:.2f}, total_frames={total_frames}")

    for scene in scenes:
        start_frame = int(scene.start_sec * fps)
        end_frame = int(scene.end_sec * fps)
        end_frame = min(end_frame, total_frames - 1)
        start_frame = max(start_frame, 0)

        if end_frame <= start_frame:
            indices = [start_frame] * n_frames
        else:
            indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int).tolist()

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            elif frames:
                frames.append(frames[-1].copy())  # duplicate last good frame
            else:
                # No frames at all — create a black frame
                frames.append(np.zeros((360, 640, 3), dtype=np.uint8))

        scene.frames = np.stack(frames, axis=0)

    cap.release()
    logger.debug("[FrameExtractor/OpenCV] Done.")
