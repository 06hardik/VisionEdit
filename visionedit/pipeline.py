"""
VisionEdit — Top-Level Pipeline Orchestrator
=============================================
Coordinates all four pipeline stages:

  Stage 1: Temporal Segmentation   (segmentation/)
  Stage 2: Tri-Stream Scoring      (streams/)  ← parallel
  Stage 3: Fusion & Selection      (fusion/)
  Stage 4: Assembly & Rendering    (rendering/)

The three intelligence streams (A/B/C) are executed concurrently using
a ThreadPoolExecutor — each stream is I/O + C-extension bound, so Python
threads are sufficient (GIL is released during numpy/OpenCV/YOLO ops).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

from visionedit.segmentation.scene_detector import detect_scenes
from visionedit.segmentation.frame_extractor import extract_frames
from visionedit.streams import semantic, affective, quality
from visionedit.fusion.scorer import fuse
from visionedit.fusion.selector import select
from visionedit.rendering.beat_detector import detect_beats
from visionedit.rendering.assembler import assemble
from visionedit.rendering.exporter import export
from visionedit.utils.data_types import SceneInfo, StreamScores, SelectedClip


def run(video_path: str, cfg: dict) -> dict:
    """
    Execute the full VisionEdit pipeline end-to-end.

    Parameters
    ----------
    video_path : str
        Path to the raw input video.
    cfg : dict
        Fully resolved pipeline configuration (from config_loader.load_config).

    Returns
    -------
    dict
        Summary result:
        {
          "num_scenes":         int,
          "num_selected":       int,
          "output_duration_sec": float,
          "output_path":        str,
        }
    """
    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Temporal Segmentation
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Stage 1 — Temporal Segmentation")
    logger.info("=" * 60)

    scenes: List[SceneInfo] = detect_scenes(video_path, cfg)
    scenes = extract_frames(video_path, scenes, cfg)

    logger.info(f"Segmentation complete: {len(scenes)} scene(s) extracted.")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — Tri-Stream Parallel Scoring
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Stage 2 — Tri-Stream Scoring (A/B/C in parallel)")
    logger.info("=" * 60)

    stream_scores_list: List[StreamScores] = _score_all_scenes(scenes, cfg)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Fusion & Knapsack Selection
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Stage 3 — Fusion & Knapsack Selection")
    logger.info("=" * 60)

    clip_scores = fuse(scenes, stream_scores_list, cfg)
    selected: List[SelectedClip] = select(clip_scores, cfg)

    if not selected:
        logger.warning(
            "No clips selected after quality gate. "
            "Try lowering blur_threshold or adjusting fusion weights."
        )
        return {
            "num_scenes": len(scenes),
            "num_selected": 0,
            "output_duration_sec": 0.0,
            "output_path": None,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4 — Assembly & Rendering
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Stage 4 — Assembly & Rendering")
    logger.info("=" * 60)

    audio_path: Optional[str] = cfg["rendering"].get("audio_path") or None
    beat_times = detect_beats(audio_path)

    final_clip = assemble(
        video_path=video_path,
        selected_clips=selected,
        beat_times=beat_times,
        cfg=cfg,
    )

    output_path = cfg["rendering"]["output_path"]
    abs_output = export(
        final_clip=final_clip,
        output_path=output_path,
        cfg=cfg,
        selected_clips=selected,
        input_path=video_path,
        num_scenes=len(scenes),
    )

    elapsed = time.time() - t_total
    logger.info(f"Pipeline finished in {elapsed:.1f}s")

    return {
        "num_scenes": len(scenes),
        "num_selected": len(selected),
        "output_duration_sec": final_clip.duration,
        "output_path": abs_output,
    }


# ── Per-scene stream scoring ──────────────────────────────────────────────────

def _score_scene(scene: SceneInfo, cfg: dict) -> StreamScores:
    """
    Score a single scene across all three streams concurrently.

    Streams A, B, C are launched as sub-futures inside a nested executor.
    """
    frames = scene.frames

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_O = ex.submit(semantic.score, frames, cfg)
        fut_E = ex.submit(affective.score, frames, cfg)
        fut_QM = ex.submit(quality.score, frames, cfg)

        O_i = fut_O.result()
        E_i = fut_E.result()
        Q_i, M_i, passes_gate = fut_QM.result()

    return StreamScores(
        scene_index=scene.index,
        O_i=O_i,
        E_i=E_i,
        Q_i=Q_i,
        M_i=M_i,
        passes_gate=passes_gate,
    )


def _score_all_scenes(scenes: List[SceneInfo], cfg: dict) -> List[StreamScores]:
    """
    Score all scenes. Scenes are processed in parallel (outer parallelism),
    with each scene's three streams also running concurrently (inner).
    """
    results: dict[int, StreamScores] = {}

    with ThreadPoolExecutor(max_workers=min(4, len(scenes))) as ex:
        future_map = {ex.submit(_score_scene, scene, cfg): scene.index for scene in scenes}

        with tqdm(total=len(scenes), desc="Scoring scenes", unit="scene") as pbar:
            for future in as_completed(future_map):
                scene_idx = future_map[future]
                try:
                    ss = future.result()
                    results[scene_idx] = ss
                except Exception as exc:
                    logger.warning(f"Scene {scene_idx} scoring failed: {exc} — using zeros.")
                    results[scene_idx] = StreamScores(scene_index=scene_idx)
                pbar.update(1)

    # Return in scene order
    return [results[i] for i in range(len(scenes))]
