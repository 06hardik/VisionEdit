"""
visionedit.streams.video_analyzer
==================================
Frame-by-frame Video & Live Webcam Emotion Recognition Analyzer.
Integrates:
  1. Face Detection (OpenCV Haar Cascade + MediaPipe support)
  2. CLCM_Attention_Hybrid (MobileNetV2 + Spatial Attention + Regularized Head)
  3. Real-time Grad-CAM Heatmap Visualization
  4. LRP/Grad-CAM Intensity Ranking (Minimal, Average, Strong)
  5. Confidence-Weighted Temporal Fusion for Video-level Emotion Aggregation
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionedit.streams.explainability import (
    GradCAMExplainer,
    calculate_intensity_score,
    get_intensity_weight,
)
from visionedit.streams.keras_fer_model import (
    EMOTION_LABELS,
    load_trained_keras_model,
)


class EmotionVideoAnalyzer:
    """
    Video & Webcam Emotion Recognition Analyzer with Explainable AI & Temporal Fusion.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.class_names = class_names if class_names else EMOTION_LABELS
        self.model = load_trained_keras_model(model_path=model_path, weights_path=weights_path)

        # Grad-CAM target layer
        target_layer = None
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                target_layer = layer.name
                break

        self.explainer = GradCAMExplainer(self.model, target_layer)

        # OpenCV Face detector (Haar cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Input resolution
        self.input_size = (
            self.model.input_shape[1]
            if self.model.input_shape and self.model.input_shape[1] is not None
            else 128
        )

        # State caching for real-time smoothness
        self.frame_count = 0
        self.cached_preds = None
        self.cached_pred_class_name = "neutral"
        self.cached_confidence = 0.0
        self.cached_intensity_label = "Minimal"
        self.cached_intensity_weight = 0.2
        self.cached_heatmap_overlay = None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances face crop with CLAHE and normalizes to [0, 1].
        """
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(enhanced_rgb, (self.input_size, self.input_size))
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def process_frame(
        self,
        frame: np.ndarray,
        skip_gradcam: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Processes a single BGR frame, detects faces, predicts emotions,
        computes Grad-CAM and overlays telemetry.
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        frame_data = None
        annotated_frame = frame.copy()

        if len(faces) == 0:
            cv2.putText(
                annotated_frame, "No Face Detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return annotated_frame, None

        # Pick largest face
        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face

        # Add 15% margin
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x_min = max(0, x - margin_x)
        y_min = max(0, y - margin_y)
        x_max = min(frame.shape[1], x + w + margin_x)
        y_max = min(frame.shape[0], y + h + margin_y)

        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return annotated_frame, None

        # Predict every 2 frames for smooth UI
        if self.frame_count % 2 == 1 or self.cached_preds is None:
            norm_face = self.preprocess_face(face_crop)
            batch = np.expand_dims(norm_face, axis=0)
            preds = self.model.predict(batch, verbose=0)[0]

            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            pred_name = self.class_names[pred_idx]

            if not skip_gradcam:
                try:
                    heatmap = self.explainer.generate_heatmap(norm_face, pred_idx)
                    score_val, label_val = calculate_intensity_score(heatmap)
                    overlaid = self.explainer.overlay_heatmap(
                        cv2.resize(face_crop, (self.input_size, self.input_size)),
                        heatmap,
                        alpha=0.45,
                    )
                except Exception:
                    score_val, label_val = 0.5, "Average"
                    overlaid = cv2.resize(face_crop, (self.input_size, self.input_size))
            else:
                score_val, label_val = 0.5, "Average"
                overlaid = cv2.resize(face_crop, (self.input_size, self.input_size))

            self.cached_preds = preds
            self.cached_pred_class_name = pred_name
            self.cached_confidence = confidence
            self.cached_intensity_label = label_val
            self.cached_intensity_weight = get_intensity_weight(label_val)
            self.cached_heatmap_overlay = overlaid

        # Bounding box & telemetry
        color = (0, 255, 0) if self.cached_confidence > 0.6 else (0, 165, 255)
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

        tag = f"{self.cached_pred_class_name.upper()} {self.cached_confidence * 100:.1f}% ({self.cached_intensity_label})"
        cv2.putText(annotated_frame, tag, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Picture-in-picture Grad-CAM overlay in top-right corner
        if self.cached_heatmap_overlay is not None:
            pip_size = 128
            pip = cv2.resize(self.cached_heatmap_overlay, (pip_size, pip_size))
            h_f, w_f = annotated_frame.shape[:2]
            y1, y2 = 20, 20 + pip_size
            x1, x2 = w_f - pip_size - 20, w_f - 20
            annotated_frame[y1:y2, x1:x2] = pip
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(annotated_frame, "Grad-CAM XAI", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        frame_data = {
            "emotion": self.cached_pred_class_name,
            "confidence": self.cached_confidence,
            "intensity_label": self.cached_intensity_label,
            "intensity_weight": self.cached_intensity_weight,
            "probabilities": self.cached_preds,
        }
        return annotated_frame, frame_data

    def analyze_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        output_plot_path: Optional[str] = None,
    ) -> Dict:
        """
        Analyzes a video file end-to-end, computing Confidence-Weighted Temporal Fusion.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        history = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, data = self.process_frame(frame)
            if writer is not None:
                writer.write(annotated)
            if data is not None:
                history.append((frame_idx, data))
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Confidence-Weighted Temporal Fusion
        if len(history) == 0:
            return {"dominant_emotion": "neutral", "confidence": 0.0, "frames_analyzed": 0}

        weighted_probs = np.zeros(len(self.class_names), dtype=np.float64)
        total_weight = 0.0

        for _, d in history:
            w = d["intensity_weight"] * d["confidence"]
            weighted_probs += d["probabilities"] * w
            total_weight += w

        if total_weight > 0:
            final_probs = weighted_probs / total_weight
        else:
            final_probs = np.ones(len(self.class_names)) / len(self.class_names)

        dominant_idx = int(np.argmax(final_probs))
        dominant_emotion = self.class_names[dominant_idx]
        final_confidence = float(final_probs[dominant_idx])

        # Generate output plot
        if output_plot_path:
            self._save_temporal_plot(history, final_probs, dominant_emotion, output_plot_path)

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": final_confidence,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, final_probs)},
            "frames_analyzed": len(history),
            "total_frames": total_frames,
        }

    def _save_temporal_plot(self, history, final_probs, dominant_emotion, save_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        frames = [h[0] for h in history]
        confs = [h[1]["confidence"] * 100 for h in history]
        weights = [h[1]["intensity_weight"] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(frames, confs, color="#1976d2", lw=2, label="Confidence (%)")
        ax1.set_ylabel("Confidence (%)", fontweight="bold")
        ax1.set_title(f"Temporal Emotion Progression — Dominant: {dominant_emotion.upper()}", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left")

        ax2.plot(frames, weights, color="#e65100", lw=2, label="LRP/Grad-CAM Weight")
        ax2.set_xlabel("Frame Index", fontweight="bold")
        ax2.set_ylabel("Intensity Weight", fontweight="bold")
        ax2.set_yticks([0.2, 1.0, 1.5])
        ax2.set_yticklabels(["Minimal (0.2)", "Average (1.0)", "Strong (1.5)"])
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
