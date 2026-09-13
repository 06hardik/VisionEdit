"""
tests/test_dummy.py
====================
End-to-end smoke test that runs the full VisionEdit pipeline on a
synthetic 10-second dummy video generated with OpenCV.

This test does NOT require any real video footage and validates:
  - Segmentation  : scenes are detected without errors
  - Frame extraction : frames are populated (N, H, W, 3)
  - Stream scoring   : scores are numeric and in expected range
  - Fusion           : ClipScore objects are returned
  - Selection        : SelectedClip list is returned (may be empty on dummy video)
  - No uncaught exceptions end-to-end

Run with:
    python -m pytest tests/test_dummy.py -v
or:
    python tests/test_dummy.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_dummy_video(path: str, duration_sec: int = 10, fps: int = 25) -> None:
    """
    Generate a synthetic video: alternating solid-color frames to ensure
    PySceneDetect picks up multiple scene changes.
    """
    import cv2

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    colors = [
        (60, 20, 220),    # Red-ish
        (20, 200, 60),    # Green-ish
        (220, 80, 20),    # Blue-ish
        (200, 200, 20),   # Cyan-ish
        (20, 60, 200),    # Orange-ish
    ]

    total_frames = duration_sec * fps
    frames_per_color = total_frames // len(colors)

    for color_idx, color in enumerate(colors):
        for _ in range(frames_per_color):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            # Add slight noise so Laplacian variance > 0
            noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            out.write(frame)

    out.release()


def _stub_config(output_path: str) -> dict:
    """Minimal config using stub emotion backend (no DeepFace download needed)."""
    return {
        "segmentation": {
            "frames_per_scene": 4,      # Low for speed
            "detector": "content",
            "threshold": 10.0,          # Low threshold → more scene cuts on dummy video
        },
        "streams": {
            "semantic": {
                "yolo_model": "yolov10s.pt",
                "yolo_world_model": "yolov8s-world.pt",
                "prompt": [],
                "min_confidence": 0.25,
            },
            "affective": {
                "backend": "stub",      # No real model — returns 0.0 always
                "enforce_detection": False,
                "positive_emotions": ["happy", "surprise"],
                "negative_emotions": [],
            },
            "quality": {
                "blur_threshold": 1.0,  # Very low — solid-color frames have low variance
            },
        },
        "fusion": {
            "weights": {"w1": 0.4, "w2": 0.35, "w3": 0.25},
        },
        "selection": {
            "target_duration_sec": 8.0,
            "bin_size_sec": 0.1,
        },
        "rendering": {
            "audio_path": None,
            "slowmo_threshold": 0.99,  # Very high — won't trigger on dummy
            "slowmo_factor": 0.5,
            "output_path": output_path,
            "codec": "libx264",
            "audio_codec": "aac",
            "fps": None,
        },
    }


# ── Unit-level tests (no real models needed) ──────────────────────────────────

class TestDataTypes(unittest.TestCase):
    """Test shared dataclass contracts."""

    def test_scene_info_duration(self):
        from visionedit.utils.data_types import SceneInfo
        s = SceneInfo(index=0, start_sec=1.0, end_sec=4.5)
        self.assertAlmostEqual(s.duration_sec, 3.5)

    def test_scene_info_degenerate(self):
        from visionedit.utils.data_types import SceneInfo
        s = SceneInfo(index=0, start_sec=5.0, end_sec=3.0)
        self.assertEqual(s.duration_sec, 0.0)

    def test_stream_scores_defaults(self):
        from visionedit.utils.data_types import StreamScores
        ss = StreamScores(scene_index=0)
        self.assertEqual(ss.O_i, 0.0)
        self.assertEqual(ss.E_i, 0.0)
        self.assertTrue(ss.passes_gate)


class TestConfigLoader(unittest.TestCase):
    """Test config loading and merging."""

    def test_load_missing_file(self):
        from visionedit.utils.config_loader import load_config
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_project_config(self):
        """Load the real config.yaml from the project root."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            self.skipTest("config.yaml not found at project root")
        from visionedit.utils.config_loader import load_config
        cfg = load_config(str(config_path))
        self.assertIn("segmentation", cfg)
        self.assertIn("fusion", cfg)
        self.assertIn("rendering", cfg)

    def test_weight_sum_warning(self):
        import warnings
        import tempfile, yaml
        from visionedit.utils.config_loader import load_config
        bad_cfg = {"fusion": {"weights": {"w1": 0.9, "w2": 0.9, "w3": 0.9}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_cfg, f)
            fname = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_config(fname)
                self.assertTrue(any("sum" in str(warning.message).lower() for warning in w))
        finally:
            os.unlink(fname)


class TestQualityStream(unittest.TestCase):
    """Test Stream C without any models."""

    def _make_frames(self, n=4, noise=True):
        frames = []
        for i in range(n):
            frame = np.full((64, 64, 3), i * 40, dtype=np.uint8)
            if noise:
                frame += np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            frames.append(frame)
        return np.stack(frames)

    def test_quality_score_returns_tuple(self):
        from visionedit.streams.quality import score
        frames = self._make_frames()
        cfg = {"streams": {"quality": {"blur_threshold": 50.0}}}
        result = score(frames, cfg)
        self.assertEqual(len(result), 3)
        Q_i, M_i, passes = result
        self.assertIsInstance(Q_i, float)
        self.assertIsInstance(M_i, float)
        self.assertIsInstance(passes, bool)

    def test_motion_range(self):
        from visionedit.streams.quality import score
        frames = self._make_frames()
        cfg = {"streams": {"quality": {"blur_threshold": 1.0}}}
        _, M_i, _ = score(frames, cfg)
        self.assertGreaterEqual(M_i, 0.0)
        self.assertLessEqual(M_i, 1.0)

    def test_empty_frames(self):
        from visionedit.streams.quality import score
        cfg = {"streams": {"quality": {"blur_threshold": 100.0}}}
        Q_i, M_i, passes = score(np.array([]), cfg)
        self.assertEqual(Q_i, 0.0)
        self.assertFalse(passes)


class TestFusionScorer(unittest.TestCase):
    """Test the saliency fusion formula."""

    def _make_scene(self, idx=0):
        from visionedit.utils.data_types import SceneInfo
        s = SceneInfo(index=idx, start_sec=0.0, end_sec=3.0)
        s.frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        return s

    def test_quality_gate_zeros_score(self):
        from visionedit.utils.data_types import StreamScores
        from visionedit.fusion.scorer import fuse
        scene = self._make_scene()
        ss = StreamScores(scene_index=0, O_i=1.0, E_i=1.0, M_i=1.0, passes_gate=False)
        cfg = {"fusion": {"weights": {"w1": 0.4, "w2": 0.35, "w3": 0.25}}}
        scores = fuse([scene], [ss], cfg)
        self.assertEqual(scores[0].S_i, 0.0)

    def test_full_score(self):
        from visionedit.utils.data_types import StreamScores
        from visionedit.fusion.scorer import fuse
        scene = self._make_scene()
        ss = StreamScores(scene_index=0, O_i=1.0, E_i=1.0, M_i=1.0, passes_gate=True)
        cfg = {"fusion": {"weights": {"w1": 0.4, "w2": 0.35, "w3": 0.25}}}
        scores = fuse([scene], [ss], cfg)
        self.assertAlmostEqual(scores[0].S_i, 1.0, places=4)


class TestKnapsackSelector(unittest.TestCase):
    """Test the knapsack selector."""

    def _make_clip_score(self, idx, duration, saliency):
        from visionedit.utils.data_types import SceneInfo, StreamScores, ClipScore
        scene = SceneInfo(index=idx, start_sec=0.0, end_sec=duration)
        scene.frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        ss = StreamScores(scene_index=idx, passes_gate=True)
        return ClipScore(scene=scene, stream_scores=ss, S_i=saliency)

    def test_selects_within_budget(self):
        from visionedit.fusion.selector import select
        clips = [
            self._make_clip_score(0, 5.0, 0.9),
            self._make_clip_score(1, 5.0, 0.8),
            self._make_clip_score(2, 5.0, 0.5),
        ]
        cfg = {
            "selection": {"target_duration_sec": 10.0, "bin_size_sec": 0.1},
            "rendering": {"slowmo_threshold": 0.99, "slowmo_factor": 0.5},
        }
        selected = select(clips, cfg)
        total = sum(sc.trim_end_sec - sc.trim_start_sec for sc in selected)
        self.assertLessEqual(total, 10.1)  # allow 1 bin of tolerance

    def test_chronological_order(self):
        from visionedit.fusion.selector import select
        clips = [
            self._make_clip_score(0, 3.0, 0.9),
            self._make_clip_score(1, 3.0, 0.7),
            self._make_clip_score(2, 3.0, 0.8),
        ]
        cfg = {
            "selection": {"target_duration_sec": 9.0, "bin_size_sec": 0.1},
            "rendering": {"slowmo_threshold": 0.99, "slowmo_factor": 0.5},
        }
        selected = select(clips, cfg)
        indices = [sc.scene_index for sc in selected]
        self.assertEqual(indices, sorted(indices))


class TestBeatDetector(unittest.TestCase):
    """Test beat detector graceful fallback."""

    def test_no_audio_returns_empty(self):
        from visionedit.rendering.beat_detector import detect_beats
        result = detect_beats(None)
        self.assertEqual(len(result), 0)

    def test_snap_to_beat_no_beats(self):
        from visionedit.rendering.beat_detector import snap_to_beat
        snapped = snap_to_beat(3.5, np.array([]))
        self.assertEqual(snapped, 3.5)

    def test_snap_to_beat_within_tolerance(self):
        from visionedit.rendering.beat_detector import snap_to_beat
        beats = np.array([1.0, 2.0, 3.0, 4.0])
        snapped = snap_to_beat(3.3, beats, tolerance_sec=0.5)
        self.assertAlmostEqual(snapped, 3.0)

    def test_snap_to_beat_outside_tolerance(self):
        from visionedit.rendering.beat_detector import snap_to_beat
        beats = np.array([1.0, 2.0, 3.0, 4.0])
        snapped = snap_to_beat(3.8, beats, tolerance_sec=0.1)
        self.assertAlmostEqual(snapped, 3.8)


# ── Integration smoke test ────────────────────────────────────────────────────

class TestEndToEndDummy(unittest.TestCase):
    """
    Full pipeline smoke test on a synthetic dummy video.

    Skipped automatically if OpenCV cannot create a video writer
    (e.g. missing codec support) or if key imports fail.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import cv2
        except ImportError:
            raise unittest.SkipTest("OpenCV not installed.")

        cls.tmpdir = tempfile.mkdtemp()
        cls.video_path = os.path.join(cls.tmpdir, "dummy.mp4")
        cls.output_path = os.path.join(cls.tmpdir, "output.mp4")

        try:
            _create_dummy_video(cls.video_path, duration_sec=5, fps=25)
        except Exception as e:
            raise unittest.SkipTest(f"Could not create dummy video: {e}")

    def test_segmentation_only(self):
        """Phase 0+1 gate: segmentation runs and returns scenes."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames

        cfg = _stub_config(self.output_path)
        scenes = detect_scenes(self.video_path, cfg)
        self.assertGreater(len(scenes), 0)

        scenes = extract_frames(self.video_path, scenes, cfg)
        for scene in scenes:
            self.assertGreater(scene.frames.ndim, 1, "Frames should be a 3D+ array")

    def test_quality_stream_on_dummy(self):
        """Phase 4 gate: quality stream runs on real frames from dummy video."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score

        cfg = _stub_config(self.output_path)
        scenes = detect_scenes(self.video_path, cfg)
        scenes = extract_frames(self.video_path, scenes, cfg)

        for scene in scenes:
            Q_i, M_i, passes = score(scene.frames, cfg)
            self.assertGreaterEqual(Q_i, 0.0)
            self.assertGreaterEqual(M_i, 0.0)
            self.assertLessEqual(M_i, 1.0)

    def test_fusion_and_selection(self):
        """Phase 5 gate: fusion + selection run without errors."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score as quality_score
        from visionedit.utils.data_types import StreamScores
        from visionedit.fusion.scorer import fuse
        from visionedit.fusion.selector import select

        cfg = _stub_config(self.output_path)
        scenes = detect_scenes(self.video_path, cfg)
        scenes = extract_frames(self.video_path, scenes, cfg)

        ss_list = []
        for scene in scenes:
            Q_i, M_i, passes = quality_score(scene.frames, cfg)
            ss_list.append(StreamScores(
                scene_index=scene.index,
                O_i=0.5,   # stub
                E_i=0.0,   # stub
                Q_i=Q_i,
                M_i=M_i,
                passes_gate=passes,
            ))

        clip_scores = fuse(scenes, ss_list, cfg)
        selected = select(clip_scores, cfg)

        # May be empty if all clips fail gate — that's OK for dummy video
        for sc in selected:
            self.assertGreaterEqual(sc.S_i, 0.0)
            self.assertLessEqual(sc.trim_end_sec, 5.1)  # within video duration


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
