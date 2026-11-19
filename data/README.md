# VisionEdit Data & Media Assets
**Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)**

This directory houses test video footage and dataset splits used for training and benchmarking.

## Structure

```
data/
├── sample_test_video.mp4       # 267 KB sample video for testing and pipeline demonstrations
├── README.md                   # This documentation file
└── DATASET/                    # Facial Emotion Recognition & AffectNet Datasets
    ├── train/                  # FER-2013 Training Split (28,709 images across 7 classes)
    │   ├── angry/              # 3,995 images
    │   ├── disgust/            # 436 images (minority class addressed via smoothed weights)
    │   ├── fear/               # 4,097 images
    │   ├── happy/              # 7,215 images
    │   ├── neutral/            # 4,965 images
    │   ├── sad/                # 4,830 images
    │   └── surprise/           # 3,171 images
    ├── test/                   # FER-2013 Test/Validation Split (7,178 images)
    │   ├── angry/ (958)
    │   ├── disgust/ (111)
    │   ├── fear/ (1,024)
    │   ├── happy/ (1,774)
    │   ├── neutral/ (1,233)
    │   ├── sad/ (1,247)
    │   └── surprise/ (831)
    └── YOLO_format/            # AffectNet formatted for YOLOv8/v9 object and facial analysis
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml           # YOLO configuration defining 8 affective classes
```
