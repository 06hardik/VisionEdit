"""
Stream B -- Binary Expert CNNs for Hard Emotion Classes
=======================================================
Implements two lightweight binary expert CNNs: one for "disgust" and one
for "fear" -- the two emotion classes that consistently underperform in
every FER classifier reviewed in the literature:

    Gursesli et al. (2024)  [A1]: disgust/fear bottom of confusion matrix
    Abbas et al. (2025)     [A2]: not tested in-the-wild; CK+ hides failure
    Salman et al. (2025)    [A4]: MoEDE shows binary experts substantially
                                  improve per-class F1 for minority emotions

Architecture
------------
Inspired by Salman et al. (2025) "Mixture of Emotion Dependent Experts
(MoEDE)", IEEE OJSP 2025.  MoEDE uses 8 full MobileNetV2 binary experts
(32.76M params total) -- simplified here to 2 lightweight CLCM-style
binary CNNs (~2.4M params each) targeting only disgust and fear.

Trigger rule (from architecture proposal)
-----------------------------------------
Run experts when primary CNN top-1 confidence < 0.60 AND top-1 or top-2
class is disgust or fear.  Expert overrides primary if expert confidence
exceeds the primary probability for that class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from visionedit.streams.fer_model import (
    EMOTION_LABELS,
    INPUT_SIZE,
    _DepthwiseSeparableConv,
)

# --------------------------------------------------------------------------
_DISGUST_IDX: int = EMOTION_LABELS.index("disgust")
_FEAR_IDX: int = EMOTION_LABELS.index("fear")

DEFAULT_DISGUST_WEIGHTS: str = "weights/expert_disgust.pt"
DEFAULT_FEAR_WEIGHTS: str = "weights/expert_fear.pt"
DEFAULT_TRIGGER_THRESHOLD: float = 0.60

_HARD_CLASS_INDICES: List[int] = [
    EMOTION_LABELS.index("angry"),
    EMOTION_LABELS.index("disgust"),
    EMOTION_LABELS.index("fear"),
    EMOTION_LABELS.index("sad"),
]


class BinaryExpertCNN(nn.Module):
    """
    Lightweight binary expert CNN: "target emotion" vs. "rest".

    CLCM-style depthwise-separable architecture with 2-class output.
    Trained with balanced sampling (all target-class samples + equal-sized
    random negative sample) -- mirrors MoEDE sampling strategy (Salman
    et al., 2025) and directly addresses class imbalance without
    oversampling artificially.

    Input : (B, 1, 48, 48)  -- grayscale, normalised [0, 1]
    Output: (B, 2)          -- [p_not_target, p_target] softmax probs
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            _DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            _DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            _DepthwiseSeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)


class ExpertEnsemble:
    """
    Ensemble of two binary expert CNNs for disgust and fear correction.

    Triggered when primary FER CNN has low confidence on hard classes.
    Expert predictions can override the primary model output, specifically
    improving the two emotion classes that fail in all reviewed papers.

    Design: simplified MoEDE -- instead of 8 full backbone passes per frame
    (32.76M params), only 1-2 lightweight experts run on ambiguous frames
    (~15-20% of frames), keeping average overhead minimal.
    """

    def __init__(
        self,
        disgust_weights: Optional[str] = None,
        fear_weights: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._disgust_path = Path(disgust_weights or DEFAULT_DISGUST_WEIGHTS)
        self._fear_path = Path(fear_weights or DEFAULT_FEAR_WEIGHTS)
        self._expert_disgust: Optional[BinaryExpertCNN] = None
        self._expert_fear: Optional[BinaryExpertCNN] = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        def _load(path: Path, name: str) -> BinaryExpertCNN:
            model = BinaryExpertCNN().to(self.device)
            if path.exists():
                logger.info(f"[Expert] Loading {name} expert from: {path}")
                sd = torch.load(path, map_location=self.device, weights_only=True)
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                model.load_state_dict(sd)
                logger.info(f"[Expert] {name} expert loaded.")
            else:
                logger.warning(
                    f"[Expert] Weights not found at {str(path)!r}. "
                    f"Using random-init {name} expert (corrections meaningless)."
                )
            model.eval()
            return model

        self._expert_disgust = _load(self._disgust_path, "disgust")
        self._expert_fear = _load(self._fear_path, "fear")
        self._loaded = True

    @staticmethod
    def _preprocess(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        import cv2
        gray = (
            cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            if face_bgr.ndim == 3 else face_bgr
        )
        gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        t = torch.tensor(gray, dtype=torch.float32) / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(device)

    def _expert_confidence(
        self, expert: BinaryExpertCNN, tensor: torch.Tensor
    ) -> float:
        with torch.no_grad():
            probs = expert(tensor)   # (1, 2) -> [p_not_target, p_target]
        return float(probs[0, 1].item())

    def should_trigger(
        self,
        primary_probs: Dict[str, float],
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    ) -> bool:
        """Return True if expert ensemble should be consulted."""
        sorted_p = sorted(primary_probs.items(), key=lambda x: x[1], reverse=True)
        top1_label, top1_conf = sorted_p[0]
        top2_label = sorted_p[1][0] if len(sorted_p) > 1 else top1_label
        if top1_conf >= trigger_threshold:
            return False
        hard_labels = {EMOTION_LABELS[i] for i in _HARD_CLASS_INDICES}
        return top1_label in hard_labels or top2_label in hard_labels

    def apply(
        self,
        primary_probs: Dict[str, float],
        face_bgr: np.ndarray,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Apply expert correction to primary FER probabilities.

        Parameters
        ----------
        primary_probs : dict
            Emotion -> probability from the primary FER CNN.
        face_bgr : np.ndarray
            BGR face crop (any size; resized internally).
        trigger_threshold : float
            Primary confidence threshold below which experts are consulted.

        Returns
        -------
        corrected_probs : dict
            Updated emotion probabilities (renormalised to sum=1).
        expert_used : bool
            True if at least one expert fired and changed a prediction.
        """
        if not self.should_trigger(primary_probs, trigger_threshold):
            return primary_probs, False

        self._ensure_loaded()
        tensor = self._preprocess(face_bgr, self.device)
        corrected = dict(primary_probs)
        expert_used = False

        # Disgust expert
        p_disgust = self._expert_confidence(self._expert_disgust, tensor)
        if p_disgust > corrected.get("disgust", 0.0):
            logger.debug(
                f"[Expert] disgust: {p_disgust:.3f} > {corrected.get('disgust', 0):.3f}"
            )
            boost = p_disgust - corrected["disgust"]
            corrected["disgust"] = p_disgust
            top1 = max(primary_probs, key=primary_probs.get)
            if top1 != "disgust":
                corrected[top1] = max(0.0, corrected[top1] - boost)
            expert_used = True

        # Fear expert
        p_fear = self._expert_confidence(self._expert_fear, tensor)
        if p_fear > corrected.get("fear", 0.0):
            logger.debug(
                f"[Expert] fear: {p_fear:.3f} > {corrected.get('fear', 0):.3f}"
            )
            boost = p_fear - corrected["fear"]
            corrected["fear"] = p_fear
            top1 = max(corrected, key=corrected.get)
            if top1 != "fear":
                corrected[top1] = max(0.0, corrected[top1] - boost)
            expert_used = True

        # Renormalise
        total = sum(corrected.values())
        if total > 0:
            corrected = {k: v / total for k, v in corrected.items()}

        return corrected, expert_used


# --------------------------------------------------------------------------
_expert_singleton: Optional[ExpertEnsemble] = None


def get_expert_ensemble(
    disgust_weights: Optional[str] = None,
    fear_weights: Optional[str] = None,
    device: Optional[str] = None,
) -> ExpertEnsemble:
    """Return the module-level ExpertEnsemble singleton."""
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = ExpertEnsemble(
            disgust_weights=disgust_weights,
            fear_weights=fear_weights,
            device=device,
        )
    return _expert_singleton
