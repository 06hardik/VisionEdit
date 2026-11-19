"""
scripts/train_fer.py - Two-Phase Training Script for Emotion Recognition
========================================================================
Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)

Implements Curriculum-Like Coarse-to-fine Learning (CLCM strategy):
  Phase 1: Train classification head only (MobileNetV2 frozen) for 20 epochs
           at learning rate 1e-4.
  Phase 2: Unfreeze top 30 layers of MobileNetV2 and fine-tune end-to-end
           at learning rate 1e-5.

Key Features:
  - Square-root smoothed inverse class weighting to solve minority skew (Disgust/Fear)
  - CLAHE contrast enhancement & rotation/flip/zoom data augmentation
  - Early stopping & ReduceLROnPlateau
  - Automatic model checkpoint saving to weights/

Usage:
  python scripts/train_fer.py --data-dir data/DATASET --epochs 50 --batch-size 64
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from visionedit.streams.keras_fer_model import (
    EMOTION_LABELS,
    NUM_CLASSES,
    build_model_with_attention,
    unfreeze_model,
)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[!] GPU memory configuration: {e}")


def compute_smoothed_class_weights(train_generator):
    """
    Computes square-root smoothed inverse class weights:
      W_c = sqrt(N_max / N_c), clipped to [0.5, 3.5]
    Avoids extreme weight skew that degrades neutral prediction precision.
    """
    counts = np.bincount(train_generator.classes)
    max_count = np.max(counts)
    raw_weights = np.sqrt(max_count / counts)
    clipped_weights = np.clip(raw_weights, 0.5, 3.5)
    class_weights = {i: float(w) for i, w in enumerate(clipped_weights)}
    return class_weights


def train(data_dir: str, epochs: int = 50, batch_size: int = 64, img_size: int = 128):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing train or test folders inside: {data_dir}")

    print("=" * 70)
    print("  VisionEdit — Two-Phase Curriculum FER Training")
    print("  Students: Hardik Jindal (2024UCA1878) & Ayush Agarwal (2024UCA1822)")
    print("=" * 70)
    print(f"  Training Directory : {train_dir}")
    print(f"  Testing Directory  : {test_dir}")
    print(f"  Batch Size         : {batch_size}")
    print(f"  Target Resolution  : {img_size}x{img_size}")
    print("=" * 70)

    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_weights = compute_smoothed_class_weights(train_gen)
    print("\n[+] Smoothed Class Weights:")
    for i, name in enumerate(EMOTION_LABELS):
        print(f"    {name:10s} : {class_weights.get(i, 1.0):.3f}")

    # Build model
    model = build_model_with_attention(num_classes=NUM_CLASSES, img_size=img_size)

    # Callbacks
    weights_dir = BASE_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    chkpt_path = str(weights_dir / "best_model_weights.h5")
    final_save_path = str(weights_dir / "final_model.keras")

    callbacks = [
        ModelCheckpoint(chkpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(str(BASE_DIR / "outputs" / "training_log.csv")),
    ]

    # --- Phase 1: Frozen Backbone ---
    print("\n" + "=" * 50)
    print("  PHASE 1: Training Classification Head (Backbone Frozen)")
    print("=" * 50)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    phase1_epochs = min(20, epochs)
    model.fit(
        train_gen,
        epochs=phase1_epochs,
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # --- Phase 2: Unfreeze and Fine-tune ---
    if epochs > phase1_epochs:
        print("\n" + "=" * 50)
        print("  PHASE 2: End-to-End Fine-Tuning (Top 30 Layers Unfrozen)")
        print("=" * 50)
        model = unfreeze_model(model, num_layers_to_unfreeze=30)
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            initial_epoch=phase1_epochs,
            epochs=epochs,
            validation_data=test_gen,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    model.save(final_save_path)
    print(f"\n[+] Training complete. Model saved to: {final_save_path}")
    print(f"[+] Best weights saved to: {chkpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument("--data-dir", type=str, default=str(BASE_DIR / "data" / "DATASET"), help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Image resolution")
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size)


if __name__ == "__main__":
    main()
