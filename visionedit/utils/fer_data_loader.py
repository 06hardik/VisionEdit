"""
data_loader.py - Data Loading and Augmentation for Facial Emotion Recognition
==============================================================================

This module handles all data ingestion, augmentation, and preprocessing for the
emotion recognition pipeline. It provides:

    1. create_data_generators() — Builds Keras ImageDataGenerators for training
       (with augmentation) and testing (without augmentation).
    2. compute_class_weights()  — Computes balanced class weights using sklearn
       to counteract class imbalance during training.
    3. get_dataset_stats()      — Scans the dataset directory and reports the
       number of images per emotion class.

Data augmentation strategy is inspired by Paper 4 (CLCM — Cross-Layer Context
Mixing), which demonstrated that moderate geometric augmentations significantly
improve generalization on FER datasets without introducing unrealistic
distortions that could alter the perceived emotion.
"""

import os
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Import centralized configuration
import config


# =============================================================================
# 1. DATA GENERATORS
# =============================================================================

def create_data_generators():
    """
    Create training and test data generators using Keras ImageDataGenerator.

    Training Generator:
        - Applies real-time data augmentation to artificially expand the
          training set and improve generalization. Each augmentation simulates
          a real-world variation that the model might encounter at inference:

          • rotation_range=15°:
              Simulates slight head tilts. People rarely hold their heads
              perfectly upright when expressing emotions. 15° is conservative
              enough to avoid flipping the semantic meaning of asymmetric
              expressions (e.g., a contempt smirk).

          • width_shift_range=0.1 (10% of image width):
              Simulates horizontal face displacement — the subject may not
              be perfectly centered in the frame, especially in unconstrained
              (in-the-wild) settings.

          • height_shift_range=0.1 (10% of image height):
              Simulates vertical face displacement — accounts for variations
              in face detection bounding box placement.

          • horizontal_flip=True:
              Facial expressions are approximately symmetric. Flipping
              doubles the effective dataset size with minimal semantic change.
              NOTE: We do NOT use vertical_flip because upside-down faces
              are not a realistic test-time scenario.

          • shear_range=0.1:
              Simulates mild perspective distortion, as if the camera or
              subject is at a slight angle. This helps the model handle
              non-frontal face captures.

          • zoom_range=0.1:
              Simulates slight scale variations — the face may appear closer
              or farther from the camera across different frames or subjects.

          • rescale=1.0/255:
              Normalizes pixel values from [0, 255] to [0, 1]. MobileNetV2
              expects inputs in this range when using its default preprocessing.

    Test Generator:
        - Only applies rescaling (no augmentation). Test data must remain
          unmodified to provide an unbiased estimate of model performance.

    Returns:
        tuple: (train_generator, test_generator)
            - train_generator: DirectoryIterator yielding augmented batches
            - test_generator: DirectoryIterator yielding unmodified batches

    Notes:
        - Both generators use flow_from_directory, which infers class labels
          from subdirectory names (angry/, disgust/, fear/, etc.).
        - class_mode='categorical' produces one-hot encoded label vectors
          of shape (NUM_CLASSES,) = (7,).
        - shuffle=False for the test generator ensures reproducible evaluation
          and correct alignment of predictions with ground truth labels.
    """

    # -------------------------------------------------------------------------
    # Training generator with augmentation (Paper 4: CLCM augmentation strategy)
    # -------------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,            # Normalize pixels to [0, 1]
        rotation_range=10,            # ±10° rotation — simulates head tilts
        width_shift_range=0.08,       # ±8% horizontal shift
        height_shift_range=0.08,      # ±8% vertical shift
        horizontal_flip=True,         # Mirror flip — expressions are ~symmetric
        zoom_range=0.08,              # ±8% zoom
        fill_mode='nearest',
    )

    # -------------------------------------------------------------------------
    # Test generator — rescaling only, NO augmentation
    # -------------------------------------------------------------------------
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,            # Same normalization as training
    )

    # -------------------------------------------------------------------------
    # Create directory iterators
    # -------------------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        directory=config.TRAIN_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),  # Resize to 224x224
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',     # One-hot encoded labels for softmax output
        classes=config.CLASS_NAMES,   # Explicit class ordering for consistency
        shuffle=True,                 # Shuffle training data each epoch
        color_mode='rgb',             # MobileNetV2 expects 3-channel input
    )

    test_generator = test_datagen.flow_from_directory(
        directory=config.TEST_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),  # Same size as training
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        classes=config.CLASS_NAMES,   # Same class ordering as training
        shuffle=False,                # No shuffle — preserves order for evaluation
        color_mode='rgb',
    )

    return train_generator, test_generator


# =============================================================================
# 2. CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights_from_directory(train_dir):
    """
    Compute balanced class weights from the training directory structure.

    Class imbalance is a critical issue in FER datasets. For example, in our
    dataset 'disgust' has only 436 samples while 'happy' has 7215 — a 16.5x
    imbalance. Without correction, the model would learn to predict 'happy'
    disproportionately because it minimizes the average loss.

    This function uses sklearn's compute_class_weight with strategy='balanced',
    which computes weights as:

        w_c = N_total / (K * N_c)

    where:
        N_total = total number of training samples
        K       = number of classes (7)
        N_c     = number of samples in class c

    This ensures each class contributes equally to the total loss, effectively
    upweighting rare classes (disgust) and downweighting common ones (happy).

    Args:
        train_dir (str): Path to the training directory containing class
                         subdirectories (e.g., TRAIN_DIR/angry/, TRAIN_DIR/happy/).

    Returns:
        dict: Mapping from class index (int) to weight (float).
              Example: {0: 1.03, 1: 9.48, 2: 1.01, 3: 0.57, ...}
    """
    # Count the number of images in each class subdirectory
    class_counts = {}
    for class_name in config.CLASS_NAMES:
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            # Count all files in the class directory (assumes all are valid images)
            num_images = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
            class_counts[class_name] = num_images
        else:
            print(f"[WARNING] Class directory not found: {class_path}")
            class_counts[class_name] = 0

    # Build arrays for sklearn
    all_labels = []
    for idx, class_name in enumerate(config.CLASS_NAMES):
        all_labels.extend([idx] * class_counts[class_name])
    all_labels = np.array(all_labels)

    # Compute balanced weights using sklearn
    unique_classes = np.arange(config.NUM_CLASSES)
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels,
    )

    # Square-root smoothing to prevent extreme weights (e.g. disgust 9.48x)
    smoothed_weights = np.sqrt(raw_weights)
    normalized_weights = smoothed_weights / np.mean(smoothed_weights)

    # Convert to dict format expected by Keras model.fit(class_weight=...)
    class_weight_dict = {i: float(round(w, 4)) for i, w in enumerate(normalized_weights)}

    # Log the computed weights for transparency
    print("\n[INFO] Computed class weights (square-root smoothed):")
    print("-" * 45)
    for idx, class_name in enumerate(config.CLASS_NAMES):
        print(
            f"  {idx} ({class_name:>8s}): "
            f"weight = {class_weight_dict[idx]:.4f}  "
            f"(n = {class_counts[class_name]:,})"
        )
    print("-" * 45)

    return class_weight_dict


# =============================================================================
# 3. DATASET STATISTICS
# =============================================================================

def get_dataset_stats(data_dir):
    """
    Scan a dataset directory and print detailed statistics about the class
    distribution. This is useful for:
        - Verifying the dataset was downloaded/extracted correctly
        - Understanding the severity of class imbalance
        - Debugging data pipeline issues (e.g., missing classes)

    Args:
        data_dir (str): Path to the dataset split directory (train or test).
                        Expected structure: data_dir/<class_name>/<image_files>

    Returns:
        dict: Mapping from class name (str) to image count (int).
              Example: {'angry': 3995, 'disgust': 436, ...}
    """
    print(f"\n{'='*55}")
    print(f"  Dataset Statistics: {os.path.basename(data_dir)}")
    print(f"{'='*55}")

    class_counts = {}
    total = 0

    for class_name in config.CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
        else:
            count = 0
            print(f"  [WARNING] Missing directory: {class_path}")

        class_counts[class_name] = count
        total += count

    # Print per-class statistics with percentage bars
    max_count = max(class_counts.values()) if class_counts else 1
    for class_name in config.CLASS_NAMES:
        count = class_counts[class_name]
        pct = (count / total * 100) if total > 0 else 0
        bar_len = int(30 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"  {class_name:>8s}: {count:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"  {'TOTAL':>8s}: {total:>6,}  (100.0%)")
    print(f"{'='*55}\n")

    # Report imbalance ratio (most common / least common)
    if class_counts:
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        if class_counts[min_class] > 0:
            imbalance_ratio = class_counts[max_class] / class_counts[min_class]
            print(
                f"  Imbalance ratio: {imbalance_ratio:.1f}x "
                f"('{max_class}' / '{min_class}')"
            )
        else:
            print(f"  [WARNING] Class '{min_class}' has 0 samples!")

    return class_counts


# =============================================================================
# 4. MAIN — Quick sanity check when run directly
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  Facial Emotion Recognition — Data Pipeline Check")
    print("=" * 55)

    # Print dataset statistics for both splits
    print("\n--- Training Set ---")
    train_stats = get_dataset_stats(config.TRAIN_DIR)

    print("\n--- Test Set ---")
    test_stats = get_dataset_stats(config.TEST_DIR)

    # Compute class weights from the actual directory
    print("\n--- Class Weights ---")
    weights = compute_class_weights_from_directory(config.TRAIN_DIR)

    # Create generators and verify shapes
    print("\n--- Data Generators ---")
    train_gen, test_gen = create_data_generators()
    print(f"  Training batches per epoch:   {len(train_gen)}")
    print(f"  Test batches per epoch:       {len(test_gen)}")

    # Verify a single batch
    sample_batch_x, sample_batch_y = next(train_gen)
    print(f"  Sample batch — images shape:  {sample_batch_x.shape}")
    print(f"  Sample batch — labels shape:  {sample_batch_y.shape}")
    print(f"  Pixel value range:            [{sample_batch_x.min():.3f}, {sample_batch_x.max():.3f}]")
    print("\n[OK] Data pipeline is working correctly.")
