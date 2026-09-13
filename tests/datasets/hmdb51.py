"""
tests/datasets/hmdb51.py
========================
HMDB51 local dataset loader for VisionEdit validation tests.

HMDB51 (Human Motion DataBase) is an action recognition benchmark:
  - 51 action categories
  - 6,849 clips total (~130 per category)
  - Clips are ~2–5 seconds, 320×240, 30fps
  - Three official train/test splits

How to use with this loader
---------------------------
1. Download HMDB51 from the official source:
     http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
   Direct download (6.8 GB total, in .rar files per category):
     https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

2. Extract all inner .rar files so you have:
     /path/to/hmdb51/
       brush_hair/
         April_09_brush_hair_u_nm_np1_ba_goo_0.avi
         ...
       cartwheel/
         ...
       ...

3. Optionally download the official split files:
     https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
   Extract to:
     /path/to/hmdb51_splits/
       brush_hair_test_split1.txt
       ...

4. Set paths in config.yaml:
     datasets:
       hmdb51_root: "D:/datasets/hmdb51"
       hmdb51_splits_dir: "D:/datasets/hmdb51_splits"

Categories used for VisionEdit stream validation
-------------------------------------------------
  High motion (M_i)      : "cartwheel", "somersault", "jump"
  Low motion (M_i)       : "sit", "stand", "smoke"
  Face-heavy (E_i)       : "laugh", "smile", "cry"
  Object-rich (O_i)      : "shoot_ball", "ride_bike", "play_guitar"
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ── All 51 HMDB51 categories ──────────────────────────────────────────────────

ALL_CATEGORIES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
    "climb_stairs", "cry", "cut_in_kitchen", "dive", "draw_sword", "dribble",
    "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand",
    "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
    "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
    "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit",
    "situp", "smile", "smoke", "somersault", "stand", "swing_baseball",
    "sword", "sword_exercise", "talk", "throw", "turn", "walk",
    "wave",
]

# Predefined stream-validation groups
HIGH_MOTION_CATEGORIES  = ["cartwheel", "somersault", "jump", "dive", "flic_flac"]
LOW_MOTION_CATEGORIES   = ["sit", "stand", "smoke", "eat", "drink"]
HIGH_FACE_CATEGORIES    = ["laugh", "cry", "talk", "smile", "kiss"]
HIGH_OBJECT_CATEGORIES  = ["shoot_ball", "ride_bike", "play_guitar", "dribble", "golf"]


class HMDB51Dataset:
    """
    Loader for a locally-stored HMDB51 dataset directory.

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing one subdirectory per action category.
        Expected structure: root_dir/<category>/<clip>.avi
    splits_dir : str | Path, optional
        Directory containing the official split text files.
        Required for split-aware clip retrieval.
    extensions : list[str]
        Accepted video file extensions (default: [".avi", ".mp4"]).
    """

    def __init__(
        self,
        root_dir: str | Path = r"D:\datasets\hmdb51",
        splits_dir: Optional[str | Path] = None,
        extensions: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.splits_dir = Path(splits_dir) if splits_dir else None
        self.extensions = extensions or [".avi", ".mp4", ".mkv"]

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"HMDB51 root directory not found: {self.root_dir}\n"
                "Set datasets.hmdb51_root in config.yaml to your local path."
            )

        self._clip_cache: Dict[str, List[Path]] = {}
        logger.info(f"[HMDB51] Loaded dataset from: {self.root_dir}")

    # ── Category discovery ────────────────────────────────────────────────────

    @property
    def categories(self) -> List[str]:
        """List all action category directories found in root_dir."""
        return sorted(
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _list_clips(self, category: str) -> List[Path]:
        """Return all clip paths for a given category (cached)."""
        if category not in self._clip_cache:
            cat_dir = self.root_dir / category
            if not cat_dir.is_dir():
                logger.warning(
                    f"[HMDB51] Category '{category}' not found in {self.root_dir}. "
                    f"Available: {self.categories[:10]}..."
                )
                self._clip_cache[category] = []
            else:
                clips = [
                    p for p in sorted(cat_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in self.extensions
                ]
                self._clip_cache[category] = clips
                logger.debug(f"[HMDB51] Found {len(clips)} clips for '{category}'")
        return self._clip_cache[category]

    # ── Split-aware retrieval ─────────────────────────────────────────────────

    def get_split_clips(
        self,
        category: str,
        split: int = 1,
        subset: str = "test",
    ) -> List[str]:
        """
        Get clip paths for an official HMDB51 train/test split.

        Parameters
        ----------
        category : str
            Action category name.
        split : int
            Official split number: 1, 2, or 3.
        subset : str
            "train" or "test".

        Returns
        -------
        list of str
            Absolute clip paths in the requested split/subset.
            Falls back to get_clips() if splits_dir is not configured.
        """
        if self.splits_dir is None:
            logger.warning(
                "[HMDB51] splits_dir not configured — returning all clips. "
                "Set datasets.hmdb51_splits_dir in config.yaml for split-aware loading."
            )
            return self.get_clips(category)

        split_file = self.splits_dir / f"{category}_test_split{split}.txt"
        if not split_file.exists():
            logger.warning(f"[HMDB51] Split file not found: {split_file}")
            return self.get_clips(category)

        # Parse split file: each line is "<filename> <label>"
        # label: 0 = not in split, 1 = train, 2 = test
        subset_label = 1 if subset == "train" else 2
        selected_names = set()
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and int(parts[1]) == subset_label:
                    selected_names.add(parts[0])

        all_clips = self._list_clips(category)
        return [
            str(p) for p in all_clips
            if p.name in selected_names
        ]

    def get_clips(
        self,
        category: str,
        n: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[str]:
        """
        Get up to ``n`` clip paths from a given category (no split filtering).

        Parameters
        ----------
        category : str
            Action category name (e.g. "cartwheel", "laugh").
        n : int
            Maximum number of clips to return.
        shuffle : bool
            If True, randomly sample.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list of str
            Absolute paths to video files.
        """
        clips = self._list_clips(category)
        if not clips:
            return []

        if shuffle:
            rng = random.Random(seed)
            clips = rng.sample(clips, min(n, len(clips)))
        else:
            clips = clips[:n]

        return [str(p) for p in clips]

    def get_clips_by_group(
        self,
        group: str,
        n_per_category: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Get clips for a predefined validation group.

        Parameters
        ----------
        group : str
            One of "high_motion", "low_motion", "high_face", "high_object".
        n_per_category : int
            Number of clips per category.

        Returns
        -------
        dict
            {category: [clip_path, ...]}
        """
        group_map = {
            "high_motion":  HIGH_MOTION_CATEGORIES,
            "low_motion":   LOW_MOTION_CATEGORIES,
            "high_face":    HIGH_FACE_CATEGORIES,
            "high_object":  HIGH_OBJECT_CATEGORIES,
        }
        if group not in group_map:
            raise ValueError(
                f"Unknown group '{group}'. Use: {list(group_map.keys())}"
            )

        result: Dict[str, List[str]] = {}
        for cat in group_map[group]:
            clips = self.get_clips(cat, n=n_per_category)
            if clips:
                result[cat] = clips
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        cats = self.categories
        total = sum(len(self._list_clips(c)) for c in cats)
        return (
            f"HMDB51 @ {self.root_dir}\n"
            f"  Categories : {len(cats)}\n"
            f"  Total clips: {total}\n"
            f"  Splits dir : {self.splits_dir or 'not configured'}"
        )


# ── Convenience helper for tests ──────────────────────────────────────────────

def from_config(cfg: dict) -> Optional[HMDB51Dataset]:
    """
    Create an HMDB51Dataset from the pipeline config dict.
    Returns None if hmdb51_root is not set (tests will be skipped).

    Usage in tests::

        dataset = hmdb51.from_config(cfg)
        if dataset is None:
            pytest.skip("HMDB51 not configured")
    """
    root = cfg.get("datasets", {}).get("hmdb51_root")
    if not root:
        return None
    splits_dir = cfg.get("datasets", {}).get("hmdb51_splits_dir")
    try:
        return HMDB51Dataset(root_dir=root, splits_dir=splits_dir)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return None
