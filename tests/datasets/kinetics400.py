"""
tests/datasets/kinetics400.py
==============================
Kinetics-400 local dataset loader for VisionEdit validation tests.

Kinetics-400 is a large-scale action recognition benchmark:
  - 400 human action categories
  - ~240K training clips, ~20K validation clips
  - Each clip is ~10 seconds at 25fps

How to use with this loader
---------------------------
1. Download the Kinetics-400 validation split.
   Official: https://github.com/google-deepmind/kinetics-dataset
   Recommended tool: https://github.com/cvdfoundation/kinetics-dataset
   (provides pre-downloaded .tar.gz files, ~100GB for the val split)

2. Extract so you have a directory layout like:
     /path/to/kinetics400/val/
       abseiling/
         abc123.mp4
         def456.mp4
         ...
       air drumming/
         xyz789.mp4
         ...

3. Set the root path in config.yaml:
     datasets:
       kinetics400_root: "D:/datasets/kinetics400/val"

   Or pass it directly to Kinetics400Dataset(root_dir=...) in tests.

Categories covered by VisionEdit validation tests
--------------------------------------------------
We focus on categories that exercise specific pipeline streams:
  - High-motion  : "running", "dancing", "gymnastics", "swimming"
  - High-semantic: "playing basketball", "soccer", "playing guitar"
  - Face-heavy   : "laughing", "crying", "singing"
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


# ── Categories aligned to official Kinetics-400 label_map.txt ───────────────
# Source: https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt

# HIGH MOTION (M_i) -- vigorous physical activity, full-body movement
# Kinetics dir names use spaces (e.g. "gymnastics tumbling"), not underscores
HIGH_MOTION_CATEGORIES = [
    "gymnastics tumbling",
    "breakdancing",
    "somersaulting",
    "parkour",
    "skateboarding",
    "snowboarding",
    "surfing water",
    "springboard diving",
    "hurdling",
    "long jump",
    "triple jump",
    "high jump",
    "pole vault",
    "hammer throw",
    "javelin throw",
    "shot put",
    "bungee jumping",
    "skydiving",
    "rock climbing",
    "ice skating",
    "skiing (not slalom or crosscountry)",
    "skiing slalom",
    "snowkiting",
    "bouncing on trampoline",
    "cartwheeling",
]

# HIGH SEMANTIC / OBJECT (O_i) -- prominent objects detectable by YOLO
# (sports equipment, instruments, animals, vehicles)
HIGH_SEMANTIC_CATEGORIES = [
    "shooting basketball",
    "dribbling basketball",
    "dunking basketball",
    "playing basketball",
    "playing tennis",
    "playing volleyball",
    "playing cricket",
    "playing badminton",
    "playing ice hockey",
    "kicking soccer ball",
    "shooting goal (soccer)",
    "playing guitar",
    "playing piano",
    "playing drums",
    "playing violin",
    "riding a bike",
    "riding mountain bike",
    "riding horse",
    "riding camel",
    "riding elephant",
    "driving car",
    "driving tractor",
    "sled dog racing",
    "walking the dog",
    "feeding birds",
    "catching fish",
    "archery",
    "bowling",
    "golf driving",
    "golf putting",
]

# HIGH FACE / EMOTION (E_i) -- visible faces with readable expressions
# These directly exercise Stream B (FER) — what we care most about
HIGH_FACE_CATEGORIES = [
    "laughing",
    "crying",
    "singing",
    "hugging",
    "kissing",
    "celebrating",
    "applauding",
    "clapping",
    "giving or receiving award",
    "blowing out candles",
    "opening present",
    "yawning",
    "sneezing",
    "headbanging",
    "pumping fist",
    "news anchoring",
    "presenting weather forecast",
    "testifying",
    "answering questions",
]

# LOW ACTIVITY (baseline / control group -- should have low S_i overall)
LOW_ACTIVITY_CATEGORIES = [
    "reading book",
    "reading newspaper",
    "texting",
    "using computer",
    "writing",
    "waiting in line",
    "sitting",
    "sleeping",
]

# All targeted categories (used by iter_targeted_samples)
ALL_TARGETED_CATEGORIES = (
    HIGH_MOTION_CATEGORIES
    + HIGH_SEMANTIC_CATEGORIES
    + HIGH_FACE_CATEGORIES
    + LOW_ACTIVITY_CATEGORIES
)

# Convenience alias map: our short names → official label names
# Use this when doing category lookups in results tables
CATEGORY_GROUP_MAP = {
    cat: "HIGH_MOTION"   for cat in HIGH_MOTION_CATEGORIES
}
CATEGORY_GROUP_MAP.update({cat: "HIGH_SEMANTIC" for cat in HIGH_SEMANTIC_CATEGORIES})
CATEGORY_GROUP_MAP.update({cat: "HIGH_FACE"     for cat in HIGH_FACE_CATEGORIES})
CATEGORY_GROUP_MAP.update({cat: "LOW_ACTIVITY"  for cat in LOW_ACTIVITY_CATEGORIES})


class Kinetics400Dataset:
    """
    Loader for a locally-stored Kinetics-400 split directory.

    Parameters
    ----------
    root_dir : str | Path
        Root directory of the Kinetics-400 split.
        Expected structure: root_dir/<category>/<clip>.mp4
    extensions : list[str]
        Accepted video file extensions (default: [".mp4", ".mkv", ".avi"]).
    """

    def __init__(
        self,
        root_dir: str | Path = r"D:\datasets\kinetics400",
        extensions: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.extensions = extensions or [".mp4", ".mkv", ".avi", ".webm"]

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Kinetics-400 root directory not found: {self.root_dir}\n"
                "Set datasets.kinetics400_root in config.yaml to your local path."
            )

        self._category_cache: Dict[str, List[Path]] = {}
        logger.info(f"[Kinetics-400] Loaded dataset from: {self.root_dir}")

    # ── Category discovery ────────────────────────────────────────────────────

    @property
    def categories(self) -> List[str]:
        """List all available category directory names."""
        return sorted(
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _find_category_dir(self, category: str) -> Optional[Path]:
        """
        Find a category directory, tolerating case / space / underscore differences.
        Returns None if not found.
        """
        # Direct match first
        exact = self.root_dir / category
        if exact.is_dir():
            return exact

        # Normalised match (lowercase, spaces → underscores)
        normalised = category.lower().replace(" ", "_")
        for d in self.root_dir.iterdir():
            if d.is_dir() and d.name.lower().replace(" ", "_") == normalised:
                return d

        # Alias match
        for canonical, aliases in _CATEGORY_ALIASES.items():
            if category.lower() in [a.lower() for a in aliases]:
                for alias in aliases:
                    alias_dir = self.root_dir / alias
                    if alias_dir.is_dir():
                        return alias_dir

        return None

    # ── Clip retrieval ────────────────────────────────────────────────────────

    def get_clips(
        self,
        category: str,
        n: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[str]:
        """
        Get up to ``n`` clip paths from a given category.

        Parameters
        ----------
        category : str
            Action category name (e.g. "running", "playing basketball").
        n : int
            Maximum number of clips to return.
        shuffle : bool
            If True, randomly sample from the available clips.
        seed : int
            Random seed for reproducible sampling.

        Returns
        -------
        list of str
            Absolute paths to video files. Empty list if category not found.
        """
        if category not in self._category_cache:
            cat_dir = self._find_category_dir(category)
            if cat_dir is None:
                logger.warning(
                    f"[Kinetics-400] Category '{category}' not found in {self.root_dir}. "
                    f"Available: {self.categories[:10]}..."
                )
                self._category_cache[category] = []
            else:
                clips = [
                    p for p in cat_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in self.extensions
                ]
                self._category_cache[category] = clips
                logger.debug(
                    f"[Kinetics-400] Found {len(clips)} clips for '{category}'"
                )

        clips = self._category_cache[category]
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
        group: str = "high_motion",
        n_per_category: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Get clips grouped by a predefined category group.

        Parameters
        ----------
        group : str
            One of "high_motion", "high_semantic", "high_face".
        n_per_category : int
            Number of clips per category.

        Returns
        -------
        dict
            {category: [clip_path, ...]}
        """
        group_map = {
            "high_motion": HIGH_MOTION_CATEGORIES,
            "high_semantic": HIGH_SEMANTIC_CATEGORIES,
            "high_face": HIGH_FACE_CATEGORIES,
        }
        if group not in group_map:
            raise ValueError(f"Unknown group '{group}'. Use: {list(group_map.keys())}")

        result = {}
        for cat in group_map[group]:
            clips = self.get_clips(cat, n=n_per_category)
            if clips:
                result[cat] = clips
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        cats = self.categories
        total = sum(
            len(list(p for p in (self.root_dir / c).iterdir()
                     if p.suffix.lower() in self.extensions))
            for c in cats
            if (self.root_dir / c).is_dir()
        )
        return (
            f"Kinetics-400 @ {self.root_dir}\n"
            f"  Categories : {len(cats)}\n"
            f"  Total clips: {total}"
        )


# ── Convenience helper for tests ──────────────────────────────────────────────

def from_config(cfg: dict) -> Optional[Kinetics400Dataset]:
    """
    Create a Kinetics400Dataset from the pipeline config dict.
    Returns None if kinetics400_root is not set (tests will be skipped).

    Usage in tests::

        dataset = kinetics400.from_config(cfg)
        if dataset is None:
            pytest.skip("Kinetics-400 not configured")
    """
    root = cfg.get("datasets", {}).get("kinetics400_root")
    if not root:
        return None
    try:
        return Kinetics400Dataset(root_dir=root)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return None
