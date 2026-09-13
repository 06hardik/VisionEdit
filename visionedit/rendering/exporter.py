"""
Phase 6 — Rendering: Exporter
===============================
Writes the final assembled clip to disk as an .mp4 file and saves
a run_metadata.json alongside it for reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from loguru import logger

from visionedit.utils.data_types import SelectedClip


def export(
    final_clip,   # moviepy VideoClip
    output_path: str,
    cfg: dict,
    selected_clips: List[SelectedClip],
    input_path: str,
    num_scenes: int,
) -> str:
    """
    Render the final clip to an .mp4 file and write run_metadata.json.

    Parameters
    ----------
    final_clip : moviepy.video.VideoClip.VideoClip
        The assembled clip ready for export.
    output_path : str
        Destination file path (e.g. ``output/highlight.mp4``).
    cfg : dict
        Full pipeline config (snapshotted into metadata).
    selected_clips : List[SelectedClip]
        The selected clips (for metadata).
    input_path : str
        Original input video path (for metadata).
    num_scenes : int
        Total number of scenes detected (for metadata).

    Returns
    -------
    str
        Absolute path to the written output file.
    """
    render_cfg = cfg["rendering"]
    codec: str = render_cfg.get("codec", "libx264")
    audio_codec: str = render_cfg.get("audio_codec", "aac")
    fps = render_cfg.get("fps") or None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Exporter] Writing output to: {output_path}")

    write_kwargs: dict[str, Any] = {
        "codec": codec,
        "audio_codec": audio_codec,
        "logger": None,   # suppress MoviePy's built-in progress output (we use loguru)
    }
    if fps:
        write_kwargs["fps"] = fps

    final_clip.write_videofile(str(output_path), **write_kwargs)

    # ── Write metadata ────────────────────────────────────────────────────────
    metadata = _build_metadata(
        input_path=input_path,
        output_path=str(output_path.resolve()),
        cfg=cfg,
        selected_clips=selected_clips,
        num_scenes=num_scenes,
        output_duration_sec=final_clip.duration,
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    logger.info(f"[Exporter] Metadata written to: {metadata_path}")
    logger.success(f"[Exporter] ✅ Export complete: {output_path}")

    return str(output_path.resolve())


def _build_metadata(
    input_path: str,
    output_path: str,
    cfg: dict,
    selected_clips: List[SelectedClip],
    num_scenes: int,
    output_duration_sec: float,
) -> dict:
    """Build a metadata dict for reproducibility."""
    return {
        "visionedit_version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": input_path,
        "output_path": output_path,
        "num_scenes_detected": num_scenes,
        "num_clips_selected": len(selected_clips),
        "output_duration_sec": round(output_duration_sec, 3),
        "selected_clip_indices": [sc.scene_index for sc in selected_clips],
        "selected_clip_saliencies": [round(sc.S_i, 4) for sc in selected_clips],
        "slowmo_applied_to": [
            sc.scene_index for sc in selected_clips if sc.apply_slowmo
        ],
        "config_snapshot": cfg,
    }
