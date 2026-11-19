"""
VisionEdit — Master Unified CLI Entry Point
===========================================
Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)

Autonomous Video Intelligence & Highlight Extraction via Dual Multi-Stream Saliency Fusion
(Facial Emotion Recognition + Two-Stage Object Detection + Quality Gating)

Supported Modes:
  1. pipeline      -- Full multi-stream video highlight detection (OD + FER + Quality + Knapsack)
  2. webcam        -- Real-time webcam emotion recognition with live Grad-CAM overlay
  3. emotion-video -- Frame-by-frame emotion inference on any video with temporal fusion plot
  4. predict-image -- Single image emotion prediction with 3-panel Grad-CAM visualization
  5. train-fer     -- Two-phase curriculum training on FER dataset
  6. benchmark     -- Generate publication-grade research comparison tables and figures
  7. presentation  -- Build interactive presentation HTML

Usage examples:
  # 1. Full Highlight Extraction Pipeline
  python main.py --input raw_footage.mp4 --duration 60

  # 2. Live Webcam Emotion Recognition Demo
  python main.py --mode webcam

  # 3. Emotion Video Inference with Temporal Fusion Plot
  python main.py --mode emotion-video --input data/sample_test_video.mp4

  # 4. Predict Single Image with Grad-CAM
  python main.py --mode predict-image --input path/to/face.jpg

  # 5. Generate Academic Benchmark Graphs
  python main.py --mode benchmark
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from visionedit.utils.config_loader import load_config
from visionedit.utils.logging_setup import setup_logging


def print_banner():
    print("═" * 72)
    print("  VisionEdit — Autonomous Multi-Stream Video Intelligence System")
    print("  Students : Hardik Jindal (Roll No. 2024UCA1878)")
    print("             Ayush Agarwal (Roll No. 2024UCA1822)")
    print("  Research : 10 Papers Synthesized (5 OD + 5 FER) — Final Submission")
    print("═" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visionedit",
        description="VisionEdit — Automated Multi-Stream Video Editing & Emotion Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode", "-m",
        default="pipeline",
        choices=["pipeline", "webcam", "emotion-video", "predict-image", "train-fer", "benchmark", "presentation"],
        help="Execution mode (default: pipeline).",
    )

    parser.add_argument(
        "--input", "-i",
        default=None,
        metavar="FILE",
        help="Input video or image file path.",
    )

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
        help="Output path for rendered video or visualization.",
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
        help="Emotion fusion weight w1 (overrides config).",
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Semantic object fusion weight w2 (overrides config).",
    )
    parser.add_argument(
        "--w3",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Motion fusion weight w3 (overrides config).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
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
    print("\n" + "═" * 60)
    print("  ✅  VisionEdit — Highlight Pipeline Complete")
    print("═" * 60)
    print(f"  Input Video    : {result.get('input_path')}")
    print(f"  Scenes Detected: {result.get('num_scenes', '?')}")
    print(f"  Clips Selected : {result.get('num_selected', '?')}")
    print(f"  Final Duration : {result.get('output_duration_sec', 0):.1f}s")
    print(f"  Elapsed Time   : {result.get('elapsed_sec', 0):.1f}s")
    print(f"  Rendered Reel  : {result.get('output_path')}")
    print("═" * 60 + "\n")


def run_pipeline_mode(args) -> int:
    from visionedit.pipeline import run

    log = setup_logging(verbose=args.verbose)

    if not args.input:
        sample_path = PROJECT_ROOT / "data" / "sample_test_video.mp4"
        if sample_path.exists():
            log.info(f"No --input specified. Using sample test video: {sample_path}")
            args.input = str(sample_path)
        else:
            log.error("Error: --input <video_path> is required for pipeline mode.")
            return 1

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return 1

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    output_path = Path(cfg["rendering"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting VisionEdit highlight pipeline on: {input_path}")
    log.info(f"Target duration: {cfg['selection']['target_duration_sec']}s")

    t0 = time.time()
    try:
        result = run(video_path=str(input_path), cfg=cfg)
    except Exception as exc:
        log.exception(f"Pipeline execution error: {exc}")
        return 1

    result["elapsed_sec"] = time.time() - t0
    result["input_path"] = str(input_path)
    print_summary(result)
    return 0


def main() -> int:
    print_banner()
    args = parse_args()

    if args.mode == "webcam":
        from scripts.demo_webcam import main as webcam_main
        sys.argv = [sys.argv[0]]
        if args.output:
            sys.argv.extend(["--output", args.output])
        webcam_main()
        return 0

    elif args.mode == "emotion-video":
        from scripts.demo_video_inference import main as video_main
        sys.argv = [sys.argv[0]]
        if args.input:
            sys.argv.extend(["--input", args.input])
        if args.output:
            sys.argv.extend(["--output-video", args.output])
        video_main()
        return 0

    elif args.mode == "predict-image":
        from scripts.predict_image import main as predict_main
        sys.argv = [sys.argv[0]]
        if args.input:
            sys.argv.extend(["--image", args.input])
        else:
            sys.argv.append("--test-sample")
        if args.output:
            sys.argv.extend(["--output", args.output])
        predict_main()
        return 0

    elif args.mode == "train-fer":
        from scripts.train_fer import main as train_main
        sys.argv = [sys.argv[0]]
        train_main()
        return 0

    elif args.mode == "benchmark":
        print("\n[+] Generating all academic comparison graphs & benchmark tables...")
        from scripts.generate_comparison_graphs import main as graphs_main
        graphs_main()
        from scripts.generate_results import main as results_main
        try:
            results_main()
        except Exception as e:
            print(f"[!] Results generator note: {e}")
        return 0

    elif args.mode == "presentation":
        from scripts.build_presentation import main as pres_main
        pres_main()
        return 0

    else:
        return run_pipeline_mode(args)


if __name__ == "__main__":
    sys.exit(main())
