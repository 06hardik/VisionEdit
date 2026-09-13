"""
Phase 5 -- Fusion: Weighted Saliency Scorer
===========================================
Combines per-clip scores from Streams A, B, and C into a single
master saliency score:

    S_i = (w1*E_i + w2*O_i + w3*M_i) * gate(Q_i >= theta_blur)

Where:
  - E_i  -- Emotion score (Stream B), range [-1, 1]
  - O_i  -- Object/semantic relevance score (Stream A), range [0, 1]
            Now decomposed salience (conf + stability + persistence)
            rather than raw YOLO confidence (Pattern 1 improvement).
  - M_i  -- Motion energy score (Stream C), range [0, 1]
  - gate -- Quality gate: clips below blur threshold -> S_i = 0

Rich metadata
-------------
When StreamScores contains rich OD/FER fields (populated by
update_from_od_rich() / update_from_fer_rich()), the fusion layer
logs structured per-clip rows and returns them via fuse_rich()
for downstream reporting and the results table (Step 5).

Interfaces
----------
fuse()      -> List[ClipScore]               (backward-compatible)
fuse_rich() -> (List[ClipScore], List[dict]) (adds metadata records)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from loguru import logger

from visionedit.utils.data_types import ClipScore, SceneInfo, StreamScores


_TABLE_HEADER = (
    f"{'Sc':>3} {'Start':>6} {'End':>6} {'Dur':>5} "
    f"{'E_i':>6} {'Emotion':<10} {'Inten':<7} "
    f"{'O_i':>6} {'TopClass':<12} {'Stg2':>4} {'Sav%':>5} "
    f"{'M_i':>5} {'Gate':>4} {'S_i':>7}"
)
_TABLE_SEP = "-" * len(_TABLE_HEADER)


def fuse(
    scenes: List[SceneInfo],
    stream_scores_list: List[StreamScores],
    cfg: dict,
) -> List[ClipScore]:
    """
    Fuse stream scores into per-clip saliency scores.

    Backward-compatible interface.  For rich metadata records use fuse_rich().

    Parameters
    ----------
    scenes : List[SceneInfo]
        Scenes from the segmentation stage.
    stream_scores_list : List[StreamScores]
        One StreamScores per scene, in the same order.
    cfg : dict
        Full pipeline config.  Uses cfg["fusion"]["weights"].

    Returns
    -------
    List[ClipScore]
    """
    clip_scores, _ = fuse_rich(scenes, stream_scores_list, cfg)
    return clip_scores


def fuse_rich(
    scenes: List[SceneInfo],
    stream_scores_list: List[StreamScores],
    cfg: dict,
) -> Tuple[List[ClipScore], List[Dict[str, Any]]]:
    """
    Fuse stream scores into per-clip saliency scores with rich metadata.

    Parameters
    ----------
    scenes : List[SceneInfo]
        Scenes from the segmentation stage.
    stream_scores_list : List[StreamScores]
        One StreamScores per scene (with optional rich OD/FER fields).
    cfg : dict
        Full pipeline config.  Uses cfg["fusion"]["weights"].

    Returns
    -------
    clip_scores : List[ClipScore]
    metadata_records : List[dict]
        One dict per clip with all raw and derived fields, ready for
        results-table generation.  Keys:
            scene_index, start_sec, end_sec, duration_sec,
            E_i, fer_label, fer_intensity, fer_expert_used, fer_num_faces,
            O_i, od_top_class, od_track_count, od_stage2_used,
            od_cascade_savings_pct, M_i, Q_i, passes_gate, S_i
    """
    weights = cfg["fusion"]["weights"]
    w1 = float(weights.get("w1", 0.40))   # emotion
    w2 = float(weights.get("w2", 0.35))   # semantic
    w3 = float(weights.get("w3", 0.25))   # motion

    logger.info(
        f"[Fusion] Weights: w1(emotion)={w1}  w2(semantic)={w2}  w3(motion)={w3}"
    )
    logger.info(_TABLE_HEADER)
    logger.info(_TABLE_SEP)

    clip_scores: List[ClipScore] = []
    metadata_records: List[Dict[str, Any]] = []

    for scene, ss in zip(scenes, stream_scores_list):
        if ss.passes_gate:
            # E_i clamped to [0,1]: can be negative when negative emotions subtracted
            e_clamped = max(ss.E_i, 0.0)
            S_i = w1 * e_clamped + w2 * ss.O_i + w3 * ss.M_i
        else:
            S_i = 0.0

        cs = ClipScore(scene=scene, stream_scores=ss, S_i=round(S_i, 6))
        clip_scores.append(cs)

        rec: Dict[str, Any] = {
            "scene_index":            scene.index,
            "start_sec":              round(scene.start_sec, 3),
            "end_sec":                round(scene.end_sec, 3),
            "duration_sec":           round(scene.duration_sec, 3),
            "E_i":                    round(ss.E_i, 4),
            "fer_label":              ss.fer_label,
            "fer_intensity":          ss.fer_intensity,
            "fer_expert_used":        ss.fer_expert_used,
            "fer_num_faces":          ss.fer_num_faces,
            "O_i":                    round(ss.O_i, 4),
            "od_top_class":           ss.od_top_class,
            "od_track_count":         ss.od_track_count,
            "od_stage2_used":         ss.od_stage2_used,
            "od_cascade_savings_pct": round(ss.od_cascade_savings_pct, 1),
            "M_i":                    round(ss.M_i, 4),
            "Q_i":                    round(ss.Q_i, 1),
            "passes_gate":            ss.passes_gate,
            "S_i":                    round(S_i, 6),
        }
        metadata_records.append(rec)

        gate_str = "PASS" if ss.passes_gate else "FAIL"
        stg2_str = "Y" if ss.od_stage2_used else "N"
        ex_str   = "*" if ss.fer_expert_used else " "
        logger.info(
            f"{scene.index:>3d} "
            f"{scene.start_sec:>6.2f} {scene.end_sec:>6.2f} {scene.duration_sec:>5.2f} "
            f"{ss.E_i:>+6.3f} {(ss.fer_label+ex_str):<10} {ss.fer_intensity:<7} "
            f"{ss.O_i:>6.3f} {ss.od_top_class:<12} {stg2_str:>4} "
            f"{ss.od_cascade_savings_pct:>5.1f} "
            f"{ss.M_i:>5.3f} {gate_str:>4} {S_i:>7.4f}"
        )

    logger.info(_TABLE_SEP)
    passed = sum(1 for cs in clip_scores if cs.stream_scores.passes_gate)
    total = len(clip_scores)
    avg_s = sum(cs.S_i for cs in clip_scores) / max(1, total)
    top_cs = max(clip_scores, key=lambda c: c.S_i, default=None)
    expert_clips = sum(1 for r in metadata_records if r["fer_expert_used"])
    stage2_clips = sum(1 for r in metadata_records if r["od_stage2_used"])
    avg_savings  = sum(r["od_cascade_savings_pct"] for r in metadata_records) / max(1, total)

    logger.info(
        f"[Fusion] {total} clips scored | "
        f"{passed} passed gate | {total-passed} dropped | avg S_i={avg_s:.4f}"
    )
    if top_cs:
        logger.info(
            f"[Fusion] Top clip: scene {top_cs.scene.index:03d} "
            f"S_i={top_cs.S_i:.4f} "
            f"({top_cs.stream_scores.fer_label} / {top_cs.stream_scores.od_top_class})"
        )
    logger.info(
        f"[Fusion] Expert corrections: {expert_clips}/{total} clips | "
        f"Stage-2 fired: {stage2_clips}/{total} clips | "
        f"Avg cascade savings: {avg_savings:.1f}%"
    )

    return clip_scores, metadata_records
