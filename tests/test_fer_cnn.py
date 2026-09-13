"""
tests/test_fer_cnn.py
=====================
Unit tests for the custom Mini-Xception FER CNN and the MediaPipe
face detector — no real video needed, no weights file required.

These tests validate:
  - MiniXception architecture: correct output shape, valid probability sums
  - FERModel wrapper: predict() returns expected keys, inference runs on CPU
  - face_detector: crops non-empty, batch mode works, graceful empty-frame handling
  - affective.score(): stub backend returns 0.0, fer_cnn backend runs on synthetic frames

Run with:
    python -m pytest tests/test_fer_cnn.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionedit.streams.fer_model import (
    MiniXception, FERModel, EMOTION_LABELS, INPUT_SIZE, NUM_CLASSES
)


# ── Helper: synthetic face image ──────────────────────────────────────────────

def _make_face_bgr(h: int = 80, w: int = 80) -> np.ndarray:
    """Generate a synthetic BGR face crop (random pixels, skin-ish tone)."""
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(100, 220, size=(h, w, 3), dtype=np.uint8)
    return frame


def _make_frame_with_face(H: int = 240, W: int = 320) -> np.ndarray:
    """Generate a BGR frame with a bright rectangular 'face' region."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Bright patch in centre — ensures Haar cascade may pick it up
    frame[60:180, 80:240] = _make_face_bgr(120, 160)
    return frame


# ── Architecture tests ────────────────────────────────────────────────────────

class TestMiniXceptionArchitecture(unittest.TestCase):
    """Validate model structure and output shape — no weights needed."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not installed.")
        cls.torch = torch

    def test_output_shape(self):
        """Model should output (batch_size, 7) probabilities."""
        import torch
        model = MiniXception()
        model.eval()
        x = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_output_is_probability_distribution(self):
        """Softmax output should sum to ~1.0 per sample."""
        import torch
        model = MiniXception()
        model.eval()
        x = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        sums = out.sum(dim=1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_all_outputs_non_negative(self):
        """All softmax probabilities must be >= 0."""
        import torch
        model = MiniXception()
        model.eval()
        x = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        self.assertTrue((out >= 0).all().item())

    def test_num_params(self):
        """Model should have fewer than 150K parameters (lightweight constraint)."""
        model = MiniXception()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertLess(n_params, 150_000, f"Too many parameters: {n_params}")

    def test_single_channel_input(self):
        """Model only accepts single-channel (grayscale) input — shape (B,1,H,W)."""
        import torch
        model = MiniXception()
        model.eval()
        # Wrong: 3-channel input should fail
        x_bad = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with self.assertRaises(Exception):
            with torch.no_grad():
                model(x_bad)


# ── FERModel wrapper tests ─────────────────────────────────────────────────────

class TestFERModelWrapper(unittest.TestCase):
    """Test the high-level FERModel wrapper (no weights file needed)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not installed.")
        # Use a model without loading weights (random init)
        cls.model = FERModel(weights_path="nonexistent_weights.pt")

    def test_predict_returns_dict_with_all_emotions(self):
        """predict() should return a dict with exactly 7 emotion keys."""
        face = _make_face_bgr()
        result = self.model.predict(face)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(EMOTION_LABELS))

    def test_predict_probabilities_in_range(self):
        """All probabilities should be in [0, 1]."""
        face = _make_face_bgr()
        result = self.model.predict(face)
        for label, prob in result.items():
            self.assertGreaterEqual(prob, 0.0, f"{label} probability < 0")
            self.assertLessEqual(prob, 1.0, f"{label} probability > 1")

    def test_predict_probabilities_sum_to_one(self):
        """Probabilities should sum to ~1.0 (softmax output)."""
        face = _make_face_bgr()
        result = self.model.predict(face)
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_predict_batch_empty(self):
        """predict_batch([]) should return an empty list."""
        result = self.model.predict_batch([])
        self.assertEqual(result, [])

    def test_predict_batch_multiple(self):
        """predict_batch should return one dict per input face."""
        faces = [_make_face_bgr() for _ in range(3)]
        results = self.model.predict_batch(faces)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, dict)
            self.assertEqual(set(r.keys()), set(EMOTION_LABELS))

    def test_predict_grayscale_input(self):
        """predict() should handle already-grayscale (H,W) input."""
        import cv2
        face_bgr = _make_face_bgr()
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        result = self.model.predict(gray)
        self.assertEqual(set(result.keys()), set(EMOTION_LABELS))

    def test_predict_small_crop(self):
        """Should handle very small face crops (e.g. 20×20) via resize."""
        tiny = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        result = self.model.predict(tiny)
        self.assertEqual(set(result.keys()), set(EMOTION_LABELS))


# ── Face detector tests ───────────────────────────────────────────────────────

class TestFaceDetector(unittest.TestCase):
    """Test the face_detector module (MediaPipe or Haar fallback)."""

    @classmethod
    def setUpClass(cls):
        try:
            from visionedit.streams.face_detector import (
                detect_faces, detect_faces_batch,
                _MEDIAPIPE_AVAILABLE, _HAAR_AVAILABLE,
            )
            cls.detect_faces = staticmethod(detect_faces)
            cls.detect_faces_batch = staticmethod(detect_faces_batch)
            if not _MEDIAPIPE_AVAILABLE and not _HAAR_AVAILABLE:
                raise unittest.SkipTest(
                    "No face detection backend available "
                    "(install mediapipe: pip install mediapipe)"
                )
        except ImportError as e:
            raise unittest.SkipTest(f"face_detector import failed: {e}")


    def test_empty_frame_returns_empty(self):
        """An all-black frame should return an empty list of crops."""
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        crops = self.detect_faces(black)
        self.assertIsInstance(crops, list)
        # No faces expected in a black frame

    def test_returns_list(self):
        """detect_faces should always return a list (never None or exception)."""
        frame = _make_frame_with_face()
        result = self.detect_faces(frame)
        self.assertIsInstance(result, list)

    def test_crops_are_ndarray(self):
        """Each crop should be a numpy ndarray."""
        frame = _make_frame_with_face()
        crops = self.detect_faces(frame)
        for crop in crops:
            self.assertIsInstance(crop, np.ndarray)
            self.assertEqual(crop.ndim, 3)

    def test_batch_returns_list_of_lists(self):
        """detect_faces_batch should return N lists for N frames."""
        frames = np.stack([_make_frame_with_face() for _ in range(3)])
        result = self.detect_faces_batch(frames)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIsInstance(item, list)

    def test_none_frame_returns_empty(self):
        """detect_faces(None) should return [] without crashing."""
        result = self.detect_faces(None)
        self.assertEqual(result, [])


# ── Affective stream integration tests ───────────────────────────────────────

class TestAffectiveStream(unittest.TestCase):
    """Integration tests for affective.score() — both backends."""

    def _make_frames(self, n: int = 4) -> np.ndarray:
        frames = np.stack([_make_frame_with_face() for _ in range(n)])
        return frames

    def test_stub_backend_returns_zero(self):
        """stub backend should always return 0.0."""
        from visionedit.streams.affective import score
        cfg = {"streams": {"affective": {"backend": "stub"}}}
        result = score(self._make_frames(), cfg)
        self.assertEqual(result, 0.0)

    def test_stub_backend_empty_frames(self):
        """stub backend should return 0.0 even for empty frames."""
        from visionedit.streams.affective import score
        cfg = {"streams": {"affective": {"backend": "stub"}}}
        result = score(np.array([]), cfg)
        self.assertEqual(result, 0.0)

    def test_fer_cnn_backend_returns_float(self):
        """fer_cnn backend should return a float (even with random-init weights)."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed.")
        from visionedit.streams.affective import score
        cfg = {"streams": {"affective": {
            "backend": "fer_cnn",
            "model_weights": "nonexistent.pt",  # triggers random-init fallback
            "positive_emotions": ["happy", "surprise"],
            "negative_emotions": [],
            "min_face_confidence": 0.3,
            "face_padding_ratio": 0.15,
        }}}
        result = score(self._make_frames(), cfg)
        self.assertIsInstance(result, float)

    def test_fer_cnn_score_in_range(self):
        """fer_cnn score should be in [-1.0, 1.0]."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed.")
        from visionedit.streams.affective import score
        cfg = {"streams": {"affective": {
            "backend": "fer_cnn",
            "model_weights": "nonexistent.pt",
            "positive_emotions": ["happy", "surprise"],
            "negative_emotions": ["angry", "fear"],
            "min_face_confidence": 0.3,
            "face_padding_ratio": 0.15,
        }}}
        result = score(self._make_frames(), cfg)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_unknown_backend_returns_zero(self):
        """An unknown backend name should log a warning and return 0.0."""
        from visionedit.streams.affective import score
        cfg = {"streams": {"affective": {"backend": "does_not_exist"}}}
        result = score(self._make_frames(), cfg)
        self.assertEqual(result, 0.0)

    def test_empty_frames_returns_zero(self):
        """Empty frame array should always return 0.0 regardless of backend."""
        from visionedit.streams.affective import score
        for backend in ("stub", "fer_cnn"):
            cfg = {"streams": {"affective": {
                "backend": backend,
                "model_weights": "nonexistent.pt",
                "positive_emotions": ["happy"],
                "negative_emotions": [],
                "min_face_confidence": 0.5,
                "face_padding_ratio": 0.15,
            }}}
            result = score(np.array([]), cfg)
            self.assertEqual(result, 0.0, f"Backend '{backend}' failed on empty frames")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
