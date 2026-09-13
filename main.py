"""
VisionEdit — CLI Entry Point
============================
Usage examples
--------------
  # Basic run with default config:
  python main.py --input raw_footage.mp4

  # Zero-shot prompt + custom target duration:
  python main.py --input raw_footage.mp4 --prompt "dog celebration" --duration 90

  # Use a custom config file + background music:
  python main.py --input clip.mp4 --config my_config.yaml --audio music.mp3

  # Override output path:
  python main.py --input clip.mp4 --output results/reel.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

from visionedit.pipeline import run
from visionedit.utils.config_loader import load_config
from visionedit.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visionedit",
        description="VisionEdit — Automated Video Editing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="VIDEO",
        help="Path to the raw input video file.",
    )

    # ── Optional overrides ────────────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        metavar="CONFIG",
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="OUTPUT",
        help="Output .mp4 path (overrides config rendering.output_path).",
    )
    parser.add_argument(
        "--prompt", "-p",
        nargs="+",
        default=None,
        metavar="TERM",
        help="Zero-shot text prompt terms for YOLO-World (e.g. --prompt dog celebration).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Target highlight reel duration in seconds (overrides config).",
    )
    parser.add_argument(
        "--audio", "-a",
        default=None,
        metavar="AUDIO",
        help="Path to background audio track for beat-matched pacing.",
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Emotion fusion weight (overrides config fusion.weights.w1).",
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Semantic fusion weight (overrides config fusion.weights.w2).",
    )
    parser.add_argument(
        "--w3",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Motion fusion weight (overrides config fusion.weights.w3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI flag values on top of config.yaml values."""
    if args.output:
        cfg["rendering"]["output_path"] = args.output
    if args.prompt is not None:
        cfg["streams"]["semantic"]["prompt"] = args.prompt
    if args.duration is not None:
        cfg["selection"]["target_duration_sec"] = args.duration
    if args.audio:
        cfg["rendering"]["audio_path"] = args.audio
    if args.w1 is not None:
        cfg["fusion"]["weights"]["w1"] = args.w1
    if args.w2 is not None:
        cfg["fusion"]["weights"]["w2"] = args.w2
    if args.w3 is not None:
        cfg["fusion"]["weights"]["w3"] = args.w3
    return cfg


def print_summary(result: dict) -> None:
    """Pretty-print the pipeline run summary."""
    print("\n" + "═" * 60)
    print("  ✅  VisionEdit — Pipeline Complete")
    print("═" * 60)
    print(f"  Input          : {result.get('input_path')}")
    print(f"  Scenes found   : {result.get('num_scenes', '?')}")
    print(f"  Clips selected : {result.get('num_selected', '?')}")
    print(f"  Output duration: {result.get('output_duration_sec', 0):.1f}s")
    print(f"  Elapsed time   : {result.get('elapsed_sec', 0):.1f}s")
    print(f"  Output file    : {result.get('output_path')}")
    print("═" * 60 + "\n")


def main() -> int:
    args = parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    log = setup_logging(verbose=args.verbose)

    # ── Validate input path ───────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return 1

    # ── Load & patch config ───────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Ensure output directory exists
    output_path = Path(cfg["rendering"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting VisionEdit pipeline on: {input_path}")
    log.info(f"Target duration: {cfg['selection']['target_duration_sec']}s")
    if cfg["streams"]["semantic"]["prompt"]:
        log.info(f"Zero-shot prompt: {cfg['streams']['semantic']['prompt']}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        result = run(video_path=str(input_path), cfg=cfg)
    except Exception as exc:
        log.exception(f"Pipeline failed: {exc}")
        return 1

    result["elapsed_sec"] = time.time() - t0
    result["input_path"] = str(input_path)

    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
