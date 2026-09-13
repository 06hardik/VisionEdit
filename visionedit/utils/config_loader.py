"""Config loader — reads and validates config.yaml."""

import copy
from pathlib import Path
from typing import Any

import yaml


# ── Defaults (merged under any user-provided config) ──────────────────────────
_DEFAULTS: dict = {
    "segmentation": {
        "frames_per_scene": 16,
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
            "backend": "deepface",
            "enforce_detection": False,
            "positive_emotions": ["happy", "surprise"],
            "negative_emotions": [],
        },
        "quality": {
            "blur_threshold": 100.0,
        },
    },
    "fusion": {
        "weights": {
            "w1": 0.40,
            "w2": 0.35,
            "w3": 0.25,
        },
    },
    "selection": {
        "target_duration_sec": 60.0,
        "bin_size_sec": 0.1,
    },
    "rendering": {
        "audio_path": None,
        "slowmo_threshold": 0.85,
        "slowmo_factor": 0.5,
        "output_path": "output/highlight.mp4",
        "codec": "libx264",
        "audio_codec": "aac",
        "fps": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(path: str = "config.yaml") -> dict:
    """
    Load config from a YAML file and merge it over the built-in defaults.

    Parameters
    ----------
    path : str
        Path to config.yaml (default: ``config.yaml`` in the working directory).

    Returns
    -------
    dict
        Fully resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path.resolve()}\n"
            "Copy config.yaml from the project root or specify --config <path>."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        user_cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    merged = _deep_merge(_DEFAULTS, user_cfg)
    _validate(merged)
    return merged


def _validate(cfg: dict) -> None:
    """Light-weight validation — catches obvious misconfiguration early."""
    weights = cfg["fusion"]["weights"]
    total = weights["w1"] + weights["w2"] + weights["w3"]
    if not (0.99 <= total <= 1.01):
        import warnings
        warnings.warn(
            f"Fusion weights sum to {total:.3f} (expected ~1.0). "
            "Scores will still be computed but may not be in [0,1].",
            UserWarning,
            stacklevel=2,
        )

    fps = cfg["rendering"]["fps"]
    if fps is not None and fps <= 0:
        raise ValueError(f"rendering.fps must be positive or null, got {fps}.")

    bin_size = cfg["selection"]["bin_size_sec"]
    if bin_size <= 0:
        raise ValueError(f"selection.bin_size_sec must be positive, got {bin_size}.")
