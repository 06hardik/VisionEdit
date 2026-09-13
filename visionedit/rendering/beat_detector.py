"""
Phase 6 — Rendering: Beat Detection
=====================================
Uses librosa to detect beats in a background audio track.
Returns an array of beat timestamps (seconds) for use in cut alignment.

Returns an empty array gracefully if no audio path is provided.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def detect_beats(audio_path: str | None) -> np.ndarray:
    """
    Detect beat timestamps in an audio file.

    Parameters
    ----------
    audio_path : str or None
        Path to the background audio file (.mp3, .wav, .ogg, etc.).
        If None, returns an empty array (no beat matching will occur).

    Returns
    -------
    np.ndarray
        Sorted 1-D array of beat times in seconds. Shape: (N,).
        Empty array if audio_path is None or beat detection fails.
    """
    if not audio_path:
        logger.info("[BeatDetector] No audio path provided — skipping beat detection.")
        return np.array([], dtype=float)

    try:
        import librosa
    except ImportError:
        logger.warning(
            "[BeatDetector] librosa not installed — skipping beat detection. "
            "Install with: pip install librosa"
        )
        return np.array([], dtype=float)

    logger.info(f"[BeatDetector] Analysing audio: {audio_path}")

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = np.sort(beat_times)

        logger.info(
            f"[BeatDetector] Detected {len(beat_times)} beats "
            f"at {float(tempo):.1f} BPM."
        )
        return beat_times

    except Exception as exc:
        logger.warning(f"[BeatDetector] Beat detection failed: {exc}. Skipping.")
        return np.array([], dtype=float)


def snap_to_beat(time_sec: float, beat_times: np.ndarray, tolerance_sec: float = 0.5) -> float:
    """
    Snap a timestamp to the nearest beat within *tolerance_sec*.

    Parameters
    ----------
    time_sec : float
        The cut time to snap.
    beat_times : np.ndarray
        Array of beat timestamps.
    tolerance_sec : float
        Maximum allowed snap distance. If the nearest beat is farther than
        this, the original time is returned unchanged.

    Returns
    -------
    float
        Snapped (or original) timestamp.
    """
    if len(beat_times) == 0:
        return time_sec

    nearest_idx = int(np.argmin(np.abs(beat_times - time_sec)))
    nearest_beat = float(beat_times[nearest_idx])

    if abs(nearest_beat - time_sec) <= tolerance_sec:
        return nearest_beat

    return time_sec
