"""
Phase 6 — Rendering: Video Assembler
=====================================
Loads selected clips via MoviePy, applies dynamic slow-motion where
flagged, aligns cut boundaries to the nearest beat, concatenates all
clips, and mixes in the background audio track.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from visionedit.utils.data_types import SelectedClip
from visionedit.rendering.beat_detector import snap_to_beat


def assemble(
    video_path: str,
    selected_clips: List[SelectedClip],
    beat_times: np.ndarray,
    cfg: dict,
) -> "moviepy.video.VideoClip.VideoClip":
    """
    Assemble selected clips into one composite VideoClip.

    Parameters
    ----------
    video_path : str
        Path to the original input video (source for all sub-clips).
    selected_clips : List[SelectedClip]
        Ordered (chronological) list of selected clips from the selector.
    beat_times : np.ndarray
        Beat timestamps from beat_detector.detect_beats(). May be empty.
    cfg : dict
        Full pipeline config. Uses ``cfg["rendering"]``.

    Returns
    -------
    moviepy.video.VideoClip.VideoClip
        The assembled (but not yet written) final clip.
    """
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
        from moviepy.video.fx import all as vfx
    except ImportError as exc:
        raise ImportError(
            "MoviePy is required for rendering. "
            "Install with: pip install moviepy"
        ) from exc

    render_cfg = cfg["rendering"]
    slowmo_factor: float = float(render_cfg.get("slowmo_factor", 0.5))
    audio_path: Optional[str] = render_cfg.get("audio_path") or None
    source_fps = render_cfg.get("fps") or None

    if not selected_clips:
        raise ValueError("[Assembler] No clips selected — cannot assemble.")

    logger.info(f"[Assembler] Assembling {len(selected_clips)} clips from: {video_path}")

    source_video = VideoFileClip(video_path)
    processed_clips = []

    for sc in selected_clips:
        start = sc.trim_start_sec
        end = sc.trim_end_sec

        # ── Snap to nearest beat (if beat data available) ─────────────────────
        if len(beat_times) > 0:
            start = snap_to_beat(start, beat_times)
            end = snap_to_beat(end, beat_times)

        # Guard: ensure start < end and within video bounds
        video_duration = source_video.duration
        start = max(0.0, min(start, video_duration - 0.1))
        end = max(start + 0.1, min(end, video_duration))

        clip = source_video.subclip(start, end)

        # ── Apply slow-motion ─────────────────────────────────────────────────
        if sc.apply_slowmo:
            logger.debug(
                f"  Scene {sc.scene_index:03d}: "
                f"applying {slowmo_factor}x slow-mo "
                f"(S={sc.S_i:.4f})"
            )
            clip = clip.fx(vfx.speedx, slowmo_factor)

        if source_fps:
            clip = clip.set_fps(source_fps)

        processed_clips.append(clip)
        logger.debug(
            f"  Scene {sc.scene_index:03d}: "
            f"{start:.2f}s–{end:.2f}s "
            f"({'slow-mo' if sc.apply_slowmo else 'normal speed'})"
        )

    # ── Concatenate ───────────────────────────────────────────────────────────
    final_clip = concatenate_videoclips(processed_clips, method="compose")

    # ── Mix background audio ──────────────────────────────────────────────────
    if audio_path:
        try:
            from moviepy.editor import CompositeAudioClip
            audio = AudioFileClip(audio_path)

            # Loop audio if shorter than video, trim if longer
            if audio.duration < final_clip.duration:
                from moviepy.audio.fx.all import audio_loop
                audio = audio_loop(audio, duration=final_clip.duration)
            else:
                audio = audio.subclip(0, final_clip.duration)

            # Mix: blend background music with original audio (if any)
            if final_clip.audio is not None:
                audio = audio.volumex(0.4)   # background music at 40%
                mixed = CompositeAudioClip([final_clip.audio, audio])
                final_clip = final_clip.set_audio(mixed)
            else:
                final_clip = final_clip.set_audio(audio)

            logger.info(f"[Assembler] Background audio mixed in from: {audio_path}")
        except Exception as exc:
            logger.warning(f"[Assembler] Audio mixing failed: {exc}. Continuing without audio.")

    logger.info(f"[Assembler] Final clip duration: {final_clip.duration:.2f}s")
    return final_clip
