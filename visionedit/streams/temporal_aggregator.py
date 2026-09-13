"""
Stream B -- Temporal Aggregator for Emotion Predictions
========================================================
Provides clip-level emotion scoring by aggregating per-frame CNN
predictions using a rolling LSTM window.

Motivation (from literature review)
------------------------------------
Pattern 4 across both FER and OD literature: temporal/video modeling
lags behind static/image modeling.  Only 1 of 5 FER papers (Salman
et al., 2025 -- MoEDE) does any temporal modeling, and even then at
high compute cost (8 backbone passes per frame).  This module provides
temporal aggregation at a fraction of that cost using a single LSTM
over pre-computed per-frame feature vectors.

Intensity ranking
-----------------
Adapted from Punuri et al. (2024) "Decoding Human Facial Emotions:
A Ranking Approach Using Explainable AI", IEEE Access 2024.
Their 3-tier ranking (Minimal / Average / Strong) is reused here,
driven by the mean Grad-CAM activation score from FERModel.predict_with_cam()
rather than LRP relevance bins.  This removes the key limitation of LRP
(only valid on correctly classified images) since Grad-CAM works on all
predictions.

Usage
-----
aggregator = EmotionTemporalAggregator(window=10)

# Per-frame update:
aggregator.update(face_id="face_0", probs={"happy": 0.8, ...}, cam_score=0.65)

# Clip-level result:
result = aggregator.get_clip_score("face_0")
# -> {"label": "happy", "intensity": "AVERAGE", "confidence": 0.78, "window_len": 8}
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from visionedit.streams.fer_model import EMOTION_LABELS, NUM_CLASSES

# --------------------------------------------------------------------------

# Intensity ranking thresholds (from Punuri et al., 2024 -- A3)
# Mean Grad-CAM activation in [0, 1]:
#   < MINIMAL_THRESHOLD  -> "MINIMAL"
#   < STRONG_THRESHOLD   -> "AVERAGE"
#   >= STRONG_THRESHOLD  -> "STRONG"
MINIMAL_THRESHOLD: float = 0.40
STRONG_THRESHOLD: float = 0.70

# Default LSTM hidden size (small enough for real-time inference)
DEFAULT_HIDDEN_SIZE: int = 64
DEFAULT_WINDOW: int = 10


def intensity_rank(
    cam_score: float,
    minimal: float = MINIMAL_THRESHOLD,
    strong: float = STRONG_THRESHOLD,
) -> str:
    """
    Convert a Grad-CAM intensity score to a 3-tier label.

    Adapted from Punuri et al. (2024) who used LRP relevance scores.
    We use mean Grad-CAM activation as a faster, equivalent signal.

    Parameters
    ----------
    cam_score : float
        Mean heatmap activation in [0, 1] from GradCAM.compute().
    minimal, strong : float
        Thresholds separating the three tiers.

    Returns
    -------
    str
        "MINIMAL", "AVERAGE", or "STRONG"
    """
    if cam_score < minimal:
        return "MINIMAL"
    if cam_score < strong:
        return "AVERAGE"
    return "STRONG"


class _TemporalLSTM(nn.Module):
    """
    Single-layer LSTM that maps a sequence of emotion probability vectors
    to a clip-level prediction.

    Input  : (batch=1, seq_len, NUM_CLASSES)
    Output : (batch=1, NUM_CLASSES)  -- softmax probabilities
    """

    def __init__(self, input_size: int = NUM_CLASSES, hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, seq_len, NUM_CLASSES) -> (1, NUM_CLASSES) softmax."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # last hidden state
        logits = self.head(last)
        return torch.softmax(logits, dim=-1)


class EmotionTemporalAggregator:
    """
    LSTM-based rolling window aggregator over per-frame emotion predictions.

    Maintains per-face deques of per-frame probability vectors.
    When get_clip_score() is called, uses either:
      - The loaded LSTM (if weights available), or
      - A simple mean fallback (if no LSTM weights)
    to produce a clip-level emotion label + intensity ranking.

    This directly addresses Pattern 4 from the literature synthesis:
    temporal modeling is absent from 4/5 FER papers, and the one that
    has it (MoEDE) pays 8x backbone cost.  Here we pay only LSTM cost on
    top of already-computed frame features.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        lstm_weights: Optional[str] = None,
        device: Optional[str] = None,
        intensity_minimal: float = MINIMAL_THRESHOLD,
        intensity_strong: float = STRONG_THRESHOLD,
    ):
        """
        Parameters
        ----------
        window : int
            Rolling window size (number of frames to accumulate per face).
        hidden_size : int
            LSTM hidden layer size.
        lstm_weights : str, optional
            Path to pre-trained LSTM .pt weights file.
            If None or file missing, falls back to simple-mean aggregation.
        device : str, optional
            "cuda" or "cpu".  Auto-detected if None.
        intensity_minimal, intensity_strong : float
            Thresholds for MINIMAL / AVERAGE / STRONG ranking.
        """
        self.window = window
        self.intensity_minimal = intensity_minimal
        self.intensity_strong = intensity_strong
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Per-face rolling buffers: face_id -> deque of (probs_array, cam_score)
        self._probs_buf: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._cam_buf: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

        # LSTM (lazy loaded)
        self._lstm: Optional[_TemporalLSTM] = None
        self._lstm_weights = lstm_weights
        self._lstm_loaded = False

    # -- LSTM loading ------------------------------------------------------

    def _ensure_lstm(self) -> bool:
        """Load LSTM weights if available.  Returns True if LSTM is ready."""
        if self._lstm_loaded:
            return self._lstm is not None

        self._lstm_loaded = True
        if not self._lstm_weights:
            logger.debug("[Temporal] No LSTM weights path configured; using mean fallback.")
            return False

        from pathlib import Path
        p = Path(self._lstm_weights)
        if not p.exists():
            logger.warning(
                f"[Temporal] LSTM weights not found at {str(p)!r}. "
                "Using simple-mean fallback."
            )
            return False

        try:
            model = _TemporalLSTM().to(self.device)
            sd = torch.load(p, map_location=self.device, weights_only=True)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            model.load_state_dict(sd)
            model.eval()
            self._lstm = model
            logger.info(f"[Temporal] LSTM loaded from {p}.")
            return True
        except Exception as exc:
            logger.warning(f"[Temporal] Failed to load LSTM ({exc}); using mean fallback.")
            return False

    # -- Per-frame update --------------------------------------------------

    def update(
        self,
        face_id: str,
        probs: Dict[str, float],
        cam_score: float = 0.0,
    ) -> None:
        """
        Record one frame's emotion probabilities for a tracked face.

        Parameters
        ----------
        face_id : str
            Unique identifier for this face track (e.g. "face_0").
        probs : dict
            Emotion label -> probability from FERModel.predict().
        cam_score : float
            Mean Grad-CAM activation from FERModel.predict_with_cam().
            Pass 0.0 if using mini_xception backend (no GradCAM).
        """
        arr = np.array([probs.get(label, 0.0) for label in EMOTION_LABELS], dtype=np.float32)
        self._probs_buf[face_id].append(arr)
        self._cam_buf[face_id].append(float(cam_score))

    # -- Clip-level scoring ------------------------------------------------

    def get_clip_score(self, face_id: str) -> Dict:
        """
        Compute clip-level emotion score for a tracked face.

        Returns a dict with keys:
            label       : str   -- dominant emotion label
            confidence  : float -- top-1 probability
            intensity   : str   -- "MINIMAL" | "AVERAGE" | "STRONG"
            window_len  : int   -- number of frames in the window

        Falls back to returning all-neutral if no frames have been seen.
        """
        buf = list(self._probs_buf.get(face_id, []))
        cam_buf = list(self._cam_buf.get(face_id, []))

        if not buf:
            return {
                "label": "neutral",
                "confidence": 0.0,
                "intensity": "MINIMAL",
                "window_len": 0,
            }

        use_lstm = self._ensure_lstm() and len(buf) >= 2

        if use_lstm:
            agg_probs = self._lstm_aggregate(buf)
        else:
            agg_probs = self._mean_aggregate(buf)

        top_idx = int(np.argmax(agg_probs))
        label = EMOTION_LABELS[top_idx]
        confidence = float(agg_probs[top_idx])

        mean_cam = float(np.mean(cam_buf)) if cam_buf else 0.0
        # Blend cam_score with confidence for intensity: high confidence
        # AND high cam activation -> STRONG
        intensity_signal = 0.6 * mean_cam + 0.4 * confidence
        rank = intensity_rank(
            intensity_signal,
            minimal=self.intensity_minimal,
            strong=self.intensity_strong,
        )

        return {
            "label": label,
            "confidence": confidence,
            "intensity": rank,
            "window_len": len(buf),
        }

    # -- Aggregation methods -----------------------------------------------

    def _mean_aggregate(self, buf: list) -> np.ndarray:
        """Simple mean over frames (fallback when no LSTM weights)."""
        return np.mean(np.stack(buf, axis=0), axis=0)

    def _lstm_aggregate(self, buf: list) -> np.ndarray:
        """LSTM over frame sequence."""
        seq = torch.tensor(np.stack(buf, axis=0), dtype=torch.float32)
        seq = seq.unsqueeze(0).to(self.device)   # (1, seq_len, NUM_CLASSES)
        with torch.no_grad():
            out = self._lstm(seq)                 # (1, NUM_CLASSES)
        return out.squeeze(0).cpu().numpy().astype(float)

    # -- Utility -----------------------------------------------------------

    def reset(self, face_id: Optional[str] = None) -> None:
        """
        Clear buffered predictions.

        Parameters
        ----------
        face_id : str, optional
            If provided, clears only that face.  If None, clears all faces.
        """
        if face_id is not None:
            self._probs_buf.pop(face_id, None)
            self._cam_buf.pop(face_id, None)
        else:
            self._probs_buf.clear()
            self._cam_buf.clear()

    def active_faces(self) -> list:
        """Return list of face IDs currently in the buffer."""
        return list(self._probs_buf.keys())


# --------------------------------------------------------------------------
_aggregator_singleton: Optional[EmotionTemporalAggregator] = None


def get_aggregator(
    window: int = DEFAULT_WINDOW,
    lstm_weights: Optional[str] = None,
    device: Optional[str] = None,
    intensity_minimal: float = MINIMAL_THRESHOLD,
    intensity_strong: float = STRONG_THRESHOLD,
) -> EmotionTemporalAggregator:
    """
    Return the module-level EmotionTemporalAggregator singleton.

    Parameters
    ----------
    window : int
        Rolling window size in frames.
    lstm_weights : str, optional
        Path to LSTM .pt weights.  Only used on first call.
    device : str, optional
        "cuda" or "cpu".  Auto-detected if None.
    intensity_minimal, intensity_strong : float
        Intensity ranking thresholds.

    Returns
    -------
    EmotionTemporalAggregator
    """
    global _aggregator_singleton
    if _aggregator_singleton is None:
        _aggregator_singleton = EmotionTemporalAggregator(
            window=window,
            lstm_weights=lstm_weights,
            device=device,
            intensity_minimal=intensity_minimal,
            intensity_strong=intensity_strong,
        )
    return _aggregator_singleton
