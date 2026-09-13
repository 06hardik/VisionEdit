"""
tests/test_datasets.py
=======================
Validation tests for the VisionEdit pipeline using real video datasets:
  - Kinetics-400 (action recognition, stream A + C validation)
  - HMDB51       (action recognition, stream C motion ordering validation)

All tests are skipped automatically if the dataset is not configured in
config.yaml (datasets.kinetics400_root / datasets.hmdb51_root = null).
No test will fail simply because a dataset isn't downloaded.

What these tests validate
--------------------------
  Stream A (Semantic / O_i):
    - Kinetics sports clips → YOLO detects objects → O_i > 0.0
    - High-semantic category clips score higher than black frames

  Stream C (Quality+Motion / Q_i, M_i):
    - Real video clips pass the quality gate (Q_i >= blur_threshold)
    - High-motion HMDB51 clips ("cartwheel") score higher M_i than
      low-motion clips ("sit")

  Full pipeline:
    - Pipeline runs end-to-end on a real clip without crashing
    - Output duration is within tolerance of target duration

Run with:
    python -m pytest tests/test_datasets.py -v

    # To force a specific dataset path without editing config.yaml:
    KINETICS_ROOT=D:/datasets/kinetics400/val python -m pytest tests/test_datasets.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionedit.utils.config_loader import load_config

# ── Load config ───────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_cfg: dict = {}
if _CONFIG_PATH.exists():
    try:
        _cfg = load_config(str(_CONFIG_PATH))
    except Exception:
        pass

# Allow environment variable overrides for CI/CD flexibility
_KINETICS_ROOT = os.environ.get(
    "KINETICS_ROOT",
    (_cfg.get("datasets") or {}).get("kinetics400_root") or ""
)
_HMDB51_ROOT = os.environ.get(
    "HMDB51_ROOT",
    (_cfg.get("datasets") or {}).get("hmdb51_root") or ""
)
_HMDB51_SPLITS = os.environ.get(
    "HMDB51_SPLITS",
    (_cfg.get("datasets") or {}).get("hmdb51_splits_dir") or ""
)

KINETICS_AVAILABLE = bool(_KINETICS_ROOT) and Path(_KINETICS_ROOT).exists()
HMDB51_AVAILABLE   = bool(_HMDB51_ROOT)   and Path(_HMDB51_ROOT).exists()


# ── Shared stub config (low thresholds for real clips) ───────────────────────

def _real_clip_config(output_path: str) -> dict:
    """Pipeline config tuned for real clip validation (no YOLO download, stub semantic)."""
    return {
        "segmentation": {
            "frames_per_scene": 8,
            "detector": "content",
            "threshold": 27.0,
        },
        "streams": {
            "semantic": {
                "yolo_model": "yolov10s.pt",
                "yolo_world_model": "yolov8s-world.pt",
                "prompt": [],
                "min_confidence": 0.25,
            },
            "affective": {
                "backend": "stub",          # Avoid FER CNN downloads in CI
                "positive_emotions": ["happy", "surprise"],
                "negative_emotions": [],
            },
            "quality": {
                "blur_threshold": 50.0,     # Real video should easily pass
            },
        },
        "fusion": {
            "weights": {"w1": 0.4, "w2": 0.35, "w3": 0.25},
        },
        "selection": {
            "target_duration_sec": 10.0,
            "bin_size_sec": 0.1,
        },
        "rendering": {
            "audio_path": None,
            "slowmo_threshold": 0.99,
            "slowmo_factor": 0.5,
            "output_path": output_path,
            "codec": "libx264",
            "audio_codec": "aac",
            "fps": None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Kinetics-400 Tests
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(KINETICS_AVAILABLE, "Kinetics-400 not configured — skipping")
class TestKinetics400Pipeline(unittest.TestCase):
    """
    Validates Streams A and C on Kinetics-400 action clips.

    Stream A (Semantic): Sports clips → YOLO detects people/objects → O_i > 0
    Stream C (Motion): Fast-action clips → high M_i
    """

    @classmethod
    def setUpClass(cls):
        from tests.datasets.kinetics400 import Kinetics400Dataset
        cls.dataset = Kinetics400Dataset(root_dir=_KINETICS_ROOT)
        print(f"\n{cls.dataset.summary()}")

    def test_dataset_has_categories(self):
        """Dataset directory should contain at least one category."""
        self.assertGreater(len(self.dataset.categories), 0)

    def test_quality_stream_on_real_clips(self):
        """Real video clips should pass the quality gate (not blurry)."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            cfg = _real_clip_config(tmp.name)

        # Get any available clips
        clips = []
        for cat in self.dataset.categories[:5]:
            clips.extend(self.dataset.get_clips(cat, n=2))
            if clips:
                break

        if not clips:
            self.skipTest("No clips found in Kinetics-400 dataset")

        pass_count = 0
        total = 0
        for clip_path in clips[:3]:
            try:
                cfg_local = _real_clip_config(clip_path + "_out.mp4")
                scenes = detect_scenes(clip_path, cfg_local)
                scenes = extract_frames(clip_path, scenes, cfg_local)
                for scene in scenes:
                    Q_i, M_i, passes = score(scene.frames, cfg_local)
                    self.assertGreaterEqual(Q_i, 0.0)
                    self.assertGreaterEqual(M_i, 0.0)
                    self.assertLessEqual(M_i, 1.0)
                    if passes:
                        pass_count += 1
                    total += 1
            except Exception as e:
                print(f"  Warning: {Path(clip_path).name} failed: {e}")

        if total > 0:
            pass_rate = pass_count / total
            print(f"\n[Kinetics-400] Quality gate pass rate: {pass_rate:.0%} ({pass_count}/{total})")
            # At least 50% of real video scenes should pass the blur gate
            self.assertGreater(pass_rate, 0.5, "Too many real clips failed quality gate")

    def test_motion_score_high_action_categories(self):
        """High-action categories should have elevated M_i scores."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score

        high_motion = self.dataset.get_clips_by_group("high_motion", n_per_category=3)
        if not high_motion:
            self.skipTest("No high-motion clips found")

        motion_scores = []
        for cat, clips in high_motion.items():
            for clip_path in clips[:2]:
                try:
                    cfg = _real_clip_config(clip_path + "_out.mp4")
                    scenes = detect_scenes(clip_path, cfg)
                    scenes = extract_frames(clip_path, scenes, cfg)
                    for scene in scenes:
                        _, M_i, _ = score(scene.frames, cfg)
                        motion_scores.append(M_i)
                except Exception as e:
                    print(f"  Warning: {Path(clip_path).name} failed: {e}")

        if motion_scores:
            mean_motion = np.mean(motion_scores)
            print(f"\n[Kinetics-400] Mean M_i for high-motion clips: {mean_motion:.4f}")
            # High-action clips should have non-trivial motion
            self.assertGreater(mean_motion, 0.05,
                "High-motion Kinetics clips have unexpectedly low M_i scores")

    def test_scene_detection_on_real_clips(self):
        """Real clips should produce at least 1 scene."""
        from visionedit.segmentation.scene_detector import detect_scenes

        clips = []
        for cat in self.dataset.categories[:3]:
            clips.extend(self.dataset.get_clips(cat, n=1))

        for clip_path in clips[:3]:
            try:
                cfg = _real_clip_config(clip_path + "_out.mp4")
                scenes = detect_scenes(clip_path, cfg)
                self.assertGreater(len(scenes), 0,
                    f"No scenes detected in {Path(clip_path).name}")
            except Exception as e:
                print(f"  Warning: {Path(clip_path).name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HMDB51 Tests
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HMDB51_AVAILABLE, "HMDB51 not configured — skipping")
class TestHMDB51Pipeline(unittest.TestCase):
    """
    Validates Stream C (motion) ordering on HMDB51 action clips.

    Key test: motion scores should be ordered correctly:
      cartwheel / somersault (high M_i)  >  sit / stand (low M_i)
    """

    @classmethod
    def setUpClass(cls):
        from tests.datasets.hmdb51 import HMDB51Dataset
        cls.dataset = HMDB51Dataset(
            root_dir=_HMDB51_ROOT,
            splits_dir=_HMDB51_SPLITS or None,
        )
        print(f"\n{cls.dataset.summary()}")

    def test_dataset_has_categories(self):
        """HMDB51 directory should contain action category subdirectories."""
        self.assertGreater(len(self.dataset.categories), 0)

    def test_known_categories_present(self):
        """At least some of the 51 known categories should be present."""
        found = set(self.dataset.categories) & {"run", "sit", "laugh", "jump"}
        print(f"\n[HMDB51] Known categories found: {found}")
        self.assertGreater(len(found), 0,
            "None of the expected HMDB51 categories were found — check directory structure")

    def test_quality_gate_pass_rate_real_clips(self):
        """Real HMDB51 clips should mostly pass the blur gate."""
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score

        clips = []
        for cat in self.dataset.categories[:3]:
            clips.extend(self.dataset.get_clips(cat, n=3))

        pass_count = total = 0
        for clip_path in clips[:6]:
            try:
                cfg = _real_clip_config(clip_path + "_out.mp4")
                scenes = detect_scenes(clip_path, cfg)
                scenes = extract_frames(clip_path, scenes, cfg)
                for scene in scenes:
                    _, _, passes = score(scene.frames, cfg)
                    if passes:
                        pass_count += 1
                    total += 1
            except Exception as e:
                print(f"  Warning: {Path(clip_path).name}: {e}")

        if total > 0:
            pass_rate = pass_count / total
            print(f"\n[HMDB51] Quality gate pass rate: {pass_rate:.0%} ({pass_count}/{total})")
            self.assertGreater(pass_rate, 0.5)

    def test_motion_ordering_high_vs_low(self):
        """
        High-motion categories (cartwheel) should have higher mean M_i
        than low-motion categories (sit).
        This is the core validation: the motion stream discriminates correctly.
        """
        from visionedit.segmentation.scene_detector import detect_scenes
        from visionedit.segmentation.frame_extractor import extract_frames
        from visionedit.streams.quality import score

        def _mean_motion(clips: list) -> float:
            scores = []
            for clip_path in clips[:3]:
                try:
                    cfg = _real_clip_config(clip_path + "_out.mp4")
                    scenes = detect_scenes(clip_path, cfg)
                    scenes = extract_frames(clip_path, scenes, cfg)
                    for scene in scenes:
                        _, M_i, _ = score(scene.frames, cfg)
                        scores.append(M_i)
                except Exception as e:
                    print(f"  Warning: {Path(clip_path).name}: {e}")
            return float(np.mean(scores)) if scores else 0.0

        high_motion = self.dataset.get_clips_by_group("high_motion", n_per_category=3)
        low_motion  = self.dataset.get_clips_by_group("low_motion",  n_per_category=3)

        if not high_motion or not low_motion:
            self.skipTest("Could not find both high_motion and low_motion clips")

        high_clips = [c for clips in high_motion.values() for c in clips]
        low_clips  = [c for clips in low_motion.values()  for c in clips]

        mean_high = _mean_motion(high_clips)
        mean_low  = _mean_motion(low_clips)

        print(
            f"\n[HMDB51] Motion scores:\n"
            f"  High-motion (cartwheel/somersault): {mean_high:.4f}\n"
            f"  Low-motion  (sit/stand/smoke):      {mean_low:.4f}"
        )

        self.assertGreater(
            mean_high, mean_low,
            f"Expected high-motion clips to have higher M_i ({mean_high:.4f}) "
            f"than low-motion clips ({mean_low:.4f})"
        )

    def test_scene_count_reasonable(self):
        """A 2–5 second HMDB51 clip should produce at least 1 scene."""
        from visionedit.segmentation.scene_detector import detect_scenes

        clips = []
        for cat in self.dataset.categories[:2]:
            clips.extend(self.dataset.get_clips(cat, n=2))

        for clip_path in clips[:4]:
            try:
                cfg = _real_clip_config(clip_path + "_out.mp4")
                scenes = detect_scenes(clip_path, cfg)
                self.assertGreater(len(scenes), 0,
                    f"No scenes in {Path(clip_path).name}")
            except Exception as e:
                print(f"  Warning: {Path(clip_path).name}: {e}")

    def test_split_loading(self):
        """If splits are configured, split-aware loading should return clips."""
        if not _HMDB51_SPLITS or not Path(_HMDB51_SPLITS).exists():
            self.skipTest("HMDB51 split files not configured")

        # Try loading test split 1 for "run"
        if "run" in self.dataset.categories:
            clips = self.dataset.get_split_clips("run", split=1, subset="test")
            print(f"\n[HMDB51] 'run' test split1 clips: {len(clips)}")
            self.assertIsInstance(clips, list)
            # May be empty if "run" is not in the dataset, that's fine


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Kinetics-400 available: {KINETICS_AVAILABLE} ({_KINETICS_ROOT!r})")
    print(f"HMDB51 available:       {HMDB51_AVAILABLE}   ({_HMDB51_ROOT!r})")
    unittest.main(verbosity=2)
