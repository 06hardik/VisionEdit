"""Shared dataclasses used across every module of the VisionEdit pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class SceneInfo:
    """
    Output of the temporal segmentation stage (Phase 1).

    Attributes
    ----------
    index : int
        Zero-based scene index.
    start_sec : float
        Scene start time in seconds.
    end_sec : float
        Scene end time in seconds.
    duration_sec : float
        Scene duration in seconds (end_sec - start_sec).
    frames : np.ndarray
        Sampled frames of shape (N, H, W, C) in BGR order.
        Set after frame extraction.
    """
    index: int
    start_sec: float
    end_sec: float
    frames: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    @property
    def duration_sec(self) -> float:
        return max(self.end_sec - self.start_sec, 0.0)

    def __repr__(self) -> str:
        n = len(self.frames) if self.frames.ndim > 1 else 0
        return (
            f"SceneInfo(index={self.index}, "
            f"start={self.start_sec:.2f}s, "
            f"end={self.end_sec:.2f}s, "
            f"duration={self.duration_sec:.2f}s, "
            f"frames={n})"
        )


@dataclass
class StreamScores:
    """
    Per-clip raw scores output by the three intelligence streams (Phases 2-4).

    Attributes
    ----------
    scene_index : int
        Index of the corresponding SceneInfo.
    O_i : float
        Object/semantic relevance score from Stream A -- range [0, 1].
        Now computed as decomposed salience (conf + track_stability + persistence)
        rather than raw YOLO confidence, per Pattern 1 cross-domain finding.
    E_i : float
        Emotion score from Stream B -- range [-1, 1].
    Q_i : float
        Blur quality score from Stream C -- Laplacian variance.
    M_i : float
        Motion energy score from Stream C -- range [0, 1].
    passes_gate : bool
        True if Q_i >= blur_threshold (quality gate FR4).

    Rich OD metadata (from score_rich(), optional):
    ------------------------------------------------
    od_top_class : str
        Class name of highest-salience detected object.
    od_track_count : int
        Number of active Kalman tracks at end of clip.
    od_stage2_used : bool
        True if the expensive YOLOv9-E stage was triggered for this clip.
    od_cascade_savings_pct : float
        Estimated compute reduction (%) vs. always-on heavy model.

    Rich FER metadata (from affective.score_rich(), optional):
    -----------------------------------------------------------
    fer_label : str
        Dominant emotion label for this clip.
    fer_intensity : str
        MINIMAL | AVERAGE | STRONG (from Grad-CAM, per Punuri et al. A3).
    fer_expert_used : bool
        True if a binary expert CNN fired for this clip.
    fer_num_faces : int
        Total face detections across clip frames.
    """
    scene_index: int
    O_i: float = 0.0
    E_i: float = 0.0
    Q_i: float = 0.0
    M_i: float = 0.0
    passes_gate: bool = True
    # Rich OD fields
    od_top_class: str = "none"
    od_track_count: int = 0
    od_stage2_used: bool = False
    od_cascade_savings_pct: float = 0.0
    # Rich FER fields
    fer_label: str = "neutral"
    fer_intensity: str = "MINIMAL"
    fer_expert_used: bool = False
    fer_num_faces: int = 0

    def update_from_od_rich(self, rich: Dict[str, Any]) -> None:
        """Populate rich OD fields from score_rich() dict."""
        self.O_i = float(rich.get("o_score", self.O_i))
        self.od_top_class = str(rich.get("top_class", self.od_top_class))
        self.od_track_count = int(rich.get("track_count", self.od_track_count))
        self.od_stage2_used = bool(rich.get("stage2_used", self.od_stage2_used))
        self.od_cascade_savings_pct = float(rich.get("cascade_savings_pct", 0.0))

    def update_from_fer_rich(self, rich: Dict[str, Any]) -> None:
        """Populate rich FER fields from affective.score_rich() dict."""
        self.E_i = float(rich.get("e_score", self.E_i))
        self.fer_label = str(rich.get("label", self.fer_label))
        self.fer_intensity = str(rich.get("intensity", self.fer_intensity))
        self.fer_expert_used = bool(rich.get("expert_used", self.fer_expert_used))
        self.fer_num_faces = int(rich.get("num_faces", self.fer_num_faces))


@dataclass
class ClipScore:
    """
    Per-clip fused saliency score output by the fusion scorer (Phase 5).

    Attributes
    ----------
    scene : SceneInfo
        The originating scene.
    stream_scores : StreamScores
        Raw per-stream scores used to compute S_i.
    S_i : float
        Master saliency score: S_i = (w1·E_i + w2·O_i + w3·M_i) · gate.
    """
    scene: SceneInfo
    stream_scores: StreamScores
    S_i: float = 0.0


@dataclass
class SelectedClip:
    """
    A clip chosen by the knapsack selector (Phase 5) and fed to the assembler.

    Attributes
    ----------
    clip_score : ClipScore
        The scored clip.
    trim_start_sec : float
        Start trim offset within the source video (seconds).
    trim_end_sec : float
        End trim offset within the source video (seconds).
    apply_slowmo : bool
        True if this clip should be slowed to slowmo_factor speed.
    """
    clip_score: ClipScore
    trim_start_sec: float
    trim_end_sec: float
    apply_slowmo: bool = False

    @property
    def scene_index(self) -> int:
        return self.clip_score.scene.index

    @property
    def S_i(self) -> float:
        return self.clip_score.S_i
