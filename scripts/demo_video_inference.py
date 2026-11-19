"""
scripts/demo_video_inference.py - Full Video Emotion Recognition & Temporal Fusion
=================================================================================
Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)

Processes any video file:
  - Detects faces across frames
  - Infers emotion probabilities with CLCM_Attention_Hybrid
  - Applies Grad-CAM XAI
  - Computes Confidence-Weighted Temporal Fusion
  - Generates annotated video + temporal emotion progression plot

Usage:
  python scripts/demo_video_inference.py --input data/sample_test_video.mp4
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from visionedit.streams.video_analyzer import EmotionVideoAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Video Emotion Recognition & Temporal Fusion")
    parser.add_argument("--input", "-i", type=str, default=str(BASE_DIR / "data" / "sample_test_video.mp4"), help="Input video path")
    parser.add_argument("--model", type=str, default=str(BASE_DIR / "weights" / "final_model.keras"), help="Model path")
    parser.add_argument("--weights", type=str, default=str(BASE_DIR / "weights" / "best_model_weights.h5"), help="Weights path")
    parser.add_argument("--output-video", type=str, default=str(BASE_DIR / "outputs" / "demos" / "video_emotion_output.mp4"), help="Output video path")
    parser.add_argument("--output-plot", type=str, default=str(BASE_DIR / "outputs" / "graphs" / "video_emotion_temporal_progression.png"), help="Output plot path")
    args = parser.parse_args()

    print("=" * 70)
    print("  VisionEdit — Video Emotion Inference & Temporal Fusion Demo")
    print("  Hardik Jindal (2024UCA1878) & Ayush Agarwal (2024UCA1822)")
    print("=" * 70)
    print(f"  Input Video   : {args.input}")
    print(f"  Output Video  : {args.output_video}")
    print(f"  Output Plot   : {args.output_plot}")
    print("=" * 70)

    if not os.path.exists(args.input):
        print(f"[-] Input video file not found at: {args.input}")
        return

    analyzer = EmotionVideoAnalyzer(model_path=args.model, weights_path=args.weights)
    results = analyzer.analyze_video(
        args.input,
        output_video_path=args.output_video,
        output_plot_path=args.output_plot,
    )

    print("\n" + "=" * 50)
    print("  VIDEO EMOTION ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"  Dominant Emotion : {results['dominant_emotion'].upper()}")
    print(f"  Confidence       : {results['confidence'] * 100:.2f}%")
    print(f"  Frames Analyzed  : {results['frames_analyzed']} / {results['total_frames']}")
    print("-" * 50)
    print("  Aggregated Probabilities (Confidence-Weighted Temporal Fusion):")
    for name, p in results.get("probabilities", {}).items():
        bar = "#" * int(p * 30)
        print(f"    {name:10s} : {p * 100:5.2f}% {bar}")
    print("=" * 50)
    print(f"[+] Output video saved: {args.output_video}")
    print(f"[+] Output plot saved:  {args.output_plot}")


if __name__ == "__main__":
    main()
