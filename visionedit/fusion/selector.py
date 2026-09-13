"""
Phase 5 — Fusion: Knapsack Clip Selector
=========================================
Selects the best subset of clips whose total duration fits within
the target output duration, maximising total saliency.

Algorithm
---------
We use a 0/1 knapsack solved with a 1-D dynamic programming table.

  - Capacity  = target_duration_sec / bin_size_sec  (integer bins)
  - Weight_i  = ceil(scene.duration_sec / bin_size_sec)  (integer bins)
  - Value_i   = S_i  (float — scaled to int for DP stability)

Clips with S_i = 0 (quality gate failed) are pre-filtered.

Post-selection, the output list is sorted by scene index (chronological
order) to preserve narrative coherence, as noted in PRD Risk 2.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from loguru import logger

from visionedit.utils.data_types import ClipScore, SelectedClip


# Score scaling factor: multiply float S_i by this before int conversion
# to preserve 4 decimal places of precision in the DP table.
_SCALE = 10_000


def select(
    clip_scores: List[ClipScore],
    cfg: dict,
    slowmo_threshold: float | None = None,
    slowmo_factor: float = 0.5,
) -> List[SelectedClip]:
    """
    Run a 0/1 knapsack to select the best clips within target duration.

    Parameters
    ----------
    clip_scores : List[ClipScore]
        Scored clips from the fusion stage.
    cfg : dict
        Full pipeline config. Uses:
        - ``cfg["selection"]["target_duration_sec"]``
        - ``cfg["selection"]["bin_size_sec"]``
        - ``cfg["rendering"]["slowmo_threshold"]``
        - ``cfg["rendering"]["slowmo_factor"]``
    slowmo_threshold : float, optional
        Override for the slow-motion threshold (defaults to config value).
    slowmo_factor : float, optional
        Slow-motion speed factor (default 0.5 = half speed).

    Returns
    -------
    List[SelectedClip]
        Ordered (chronological) list of selected clips.
    """
    target_sec: float = float(cfg["selection"].get("target_duration_sec", 60.0))
    bin_size: float = float(cfg["selection"].get("bin_size_sec", 0.1))
    slow_thresh: float = float(
        cfg["rendering"].get("slowmo_threshold", 0.85)
        if slowmo_threshold is None else slowmo_threshold
    )
    slowmo_factor = float(cfg["rendering"].get("slowmo_factor", slowmo_factor))

    logger.info(
        f"[Selector] Target={target_sec}s, bin={bin_size}s, "
        f"slowmo_threshold={slow_thresh}"
    )

    # ── Pre-filter: discard zero-saliency clips ───────────────────────────────
    candidates = [cs for cs in clip_scores if cs.S_i > 0.0]
    logger.info(
        f"[Selector] Candidates after quality gate: "
        f"{len(candidates)}/{len(clip_scores)}"
    )

    if not candidates:
        logger.warning("[Selector] No candidates passed the quality gate.")
        return []

    # ── Convert to integer bins ───────────────────────────────────────────────
    capacity = int(round(target_sec / bin_size))
    weights = [max(1, math.ceil(cs.scene.duration_sec / bin_size)) for cs in candidates]
    values = [int(round(cs.S_i * _SCALE)) for cs in candidates]

    n = len(candidates)

    # ── 0/1 Knapsack DP ──────────────────────────────────────────────────────
    # dp[w] = max total value achievable with capacity w
    dp = np.zeros(capacity + 1, dtype=np.int64)

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        # Traverse backwards to maintain 0/1 property
        for w in range(capacity, w_i - 1, -1):
            dp[w] = max(dp[w], dp[w - w_i] + v_i)

    # ── Backtrack to find selected items ──────────────────────────────────────
    selected_indices: List[int] = []
    remaining = capacity
    for i in range(n - 1, -1, -1):
        w_i = weights[i]
        v_i = values[i]
        if remaining >= w_i and dp[remaining] == dp[remaining - w_i] + v_i:
            selected_indices.append(i)
            remaining -= w_i

    # ── Build SelectedClip objects (chronological order) ─────────────────────
    selected: List[SelectedClip] = []
    for idx in selected_indices:
        cs = candidates[idx]
        apply_slowmo = cs.S_i >= slow_thresh

        selected.append(
            SelectedClip(
                clip_score=cs,
                trim_start_sec=cs.scene.start_sec,
                trim_end_sec=cs.scene.end_sec,
                apply_slowmo=apply_slowmo,
            )
        )

    # Sort chronologically by scene index
    selected.sort(key=lambda sc: sc.scene_index)

    total_dur = sum(
        (sc.trim_end_sec - sc.trim_start_sec) * (slowmo_factor if sc.apply_slowmo else 1.0)
        for sc in selected
    )

    logger.info(
        f"[Selector] Selected {len(selected)} clips, "
        f"estimated output duration ≈ {total_dur:.1f}s."
    )
    for sc in selected:
        logger.debug(
            f"  Scene {sc.scene_index:03d}: "
            f"S={sc.S_i:.4f}, "
            f"dur={sc.trim_end_sec - sc.trim_start_sec:.2f}s, "
            f"slowmo={sc.apply_slowmo}"
        )

    return selected
