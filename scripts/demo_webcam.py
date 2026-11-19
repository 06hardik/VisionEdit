"""
scripts/demo_webcam.py - Live Webcam Emotion Recognition Demo
=============================================================
Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)

Demonstrates:
  - Real-time face detection
  - CLCM_Attention_Hybrid inference (MobileNetV2 + Spatial Attention)
  - Real-time Grad-CAM Explainable AI overlay
  - Intensity Ranking (Minimal / Average / Strong)
  - Confidence-Weighted Temporal Fusion

Usage:
  python scripts/demo_webcam.py
  python scripts/demo_webcam.py --camera 0 --output outputs/demos/webcam_recording.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
from visionedit.streams.video_analyzer import EmotionVideoAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Live Webcam Emotion Recognition Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--model", type=str, default=str(BASE_DIR / "weights" / "final_model.keras"), help="Model path")
    parser.add_argument("--weights", type=str, default=str(BASE_DIR / "weights" / "best_model_weights.h5"), help="Weights path")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "outputs" / "demos" / "live_webcam_demo.mp4"), help="Output video path")
    args = parser.parse_args()

    print("=" * 70)
    print("  VisionEdit — Live Webcam Emotion Recognition Demo")
    print("  Students: Hardik Jindal (2024UCA1878) & Ayush Agarwal (2024UCA1822)")
    print("=" * 70)
    print("  Model: CLCM_Attention_Hybrid (MobileNetV2 + Spatial Attention)")
    print("  XAI: Real-Time Grad-CAM Heatmap Overlay")
    print("  Intensity: 3-Tier Categorization (Minimal / Average / Strong)")
    print("  Aggregation: Confidence-Weighted Temporal Fusion")
    print("=" * 70)
    print("  [!] Press 'q' or ESC in the webcam window to stop and view results.\n")

    analyzer = EmotionVideoAnalyzer(model_path=args.model, weights_path=args.weights)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[-] Could not open camera {args.camera}. Checking fallback sample video...")
        fallback_video = BASE_DIR / "data" / "sample_test_video.mp4"
        if fallback_video.exists():
            print(f"[+] Running fallback video demo on: {fallback_video}")
            analyzer.analyze_video(
                str(fallback_video),
                output_video_path=args.output,
                output_plot_path=str(BASE_DIR / "outputs" / "graphs" / "webcam_sim_temporal_progression.png"),
            )
            print(f"[+] Output saved to {args.output}")
            return
        else:
            print("[-] No camera or fallback video found.")
            return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("[+] Camera active. Stream starting...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, data = analyzer.process_frame(frame)
        writer.write(annotated)
        cv2.imshow("VisionEdit — Live Emotion Recognition (Grad-CAM XAI)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Demo completed successfully. Recorded video saved to: {args.output}")


if __name__ == "__main__":
    main()
