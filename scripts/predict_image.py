"""
scripts/predict_image.py - Single Image Emotion Recognition & Grad-CAM XAI
==========================================================================
Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)

Predicts emotion on a face image and generates a 3-panel visualization:
  1. Input Face Image
  2. Grad-CAM Spatial Saliency Heatmap with Intensity Ranking
  3. Probability Distribution across 7 Ekman Emotion Classes

Usage:
  python scripts/predict_image.py --image "path/to/face.jpg"
  python scripts/predict_image.py --test-sample
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

from visionedit.streams.explainability import GradCAMExplainer, calculate_intensity_score
from visionedit.streams.keras_fer_model import EMOTION_LABELS, load_trained_keras_model


def predict_single_image(
    image_path: str,
    model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    save_output: str = str(BASE_DIR / "outputs" / "graphs" / "single_prediction.png"),
):
    print(f"\n[+] Loading trained model...")
    model = load_trained_keras_model(model_path=model_path, weights_path=weights_path)
    img_size = (
        model.input_shape[1]
        if model.input_shape and model.input_shape[1] is not None
        else 128
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"[+] Loading image from: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image at: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(enhanced_rgb, (img_size, img_size))
    norm_img = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm_img, axis=0)

    # Inference
    preds = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_emotion = EMOTION_LABELS[pred_idx]
    confidence = preds[pred_idx] * 100.0

    # Grad-CAM target conv layer
    target_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            target_layer = layer.name
            break

    explainer = GradCAMExplainer(model, target_layer)
    heatmap = explainer.generate_heatmap(norm_img, pred_idx)
    intensity_score, intensity_label = calculate_intensity_score(heatmap)
    overlaid = explainer.overlay_heatmap(cv2.resize(img_rgb, (img_size, img_size)), heatmap)

    print("\n" + "=" * 55)
    print("  PREDICTION RESULTS")
    print("=" * 55)
    print(f"  PREDICTED EMOTION : {pred_emotion.upper()}")
    print(f"  CONFIDENCE        : {confidence:.2f}%")
    print(f"  INTENSITY SCORE   : {intensity_score:.4f} ({intensity_label})")
    print("=" * 55)
    print("  Class Probabilities:")
    for name, p in zip(EMOTION_LABELS, preds):
        bar = "#" * int(p * 25)
        print(f"    {name:10s} : {p * 100:5.2f}% {bar}")
    print("=" * 55)

    # Save 3-panel plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_output)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlaid)
    axes[1].set_title(f"Grad-CAM Heatmap\nIntensity: {intensity_label} ({intensity_score:.3f})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    colors = ["#1f77b4" if i != pred_idx else "#d62728" for i in range(len(EMOTION_LABELS))]
    y_pos = np.arange(len(EMOTION_LABELS))
    axes[2].barh(y_pos, preds * 100, color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(EMOTION_LABELS, fontweight="bold")
    axes[2].set_xlabel("Probability (%)", fontweight="bold")
    axes[2].set_title(f"Prediction: {pred_emotion.upper()} ({confidence:.1f}%)", fontsize=12, fontweight="bold")
    axes[2].set_xlim(0, 100)
    axes[2].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_output, dpi=300)
    plt.close()
    print(f"[+] Visualization saved to: {save_output}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Emotion & Intensity on Single Image")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input face image")
    parser.add_argument("--test-sample", action="store_true", help="Use sample test image from dataset")
    parser.add_argument("--model", type=str, default=str(BASE_DIR / "weights" / "final_model.keras"), help="Model path")
    parser.add_argument("--weights", type=str, default=str(BASE_DIR / "weights" / "best_model_weights.h5"), help="Weights path")
    parser.add_argument("--output", "-o", type=str, default=str(BASE_DIR / "outputs" / "graphs" / "single_prediction.png"), help="Output plot path")
    args = parser.parse_args()

    target_image = args.image
    if args.test_sample or target_image is None:
        dataset_test = BASE_DIR / "data" / "DATASET" / "test"
        sample_found = None
        if dataset_test.exists():
            for emotion in ["happy", "surprise", "angry", "sad", "neutral"]:
                emo_dir = dataset_test / emotion
                if emo_dir.exists():
                    images = list(emo_dir.glob("*.jpg")) or list(emo_dir.glob("*.png"))
                    if images:
                        sample_found = str(images[0])
                        break
        target_image = sample_found

    if not target_image:
        print("[-] Please specify an image path with --image or ensure data/DATASET/test exists.")
        return

    predict_single_image(
        image_path=target_image,
        model_path=args.model,
        weights_path=args.weights,
        save_output=args.output,
    )


if __name__ == "__main__":
    main()
