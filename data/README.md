# VisionEdit Data & Media Assets
**Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)**

This directory houses test video footage and dataset splits used for training and benchmarking VisionEdit across **6 landmark computer vision datasets** (3 in Facial Emotion Recognition, 3 in Object Detection & Video Intelligence).

For the complete technical breakdown, academic citations, class statistics, and preprocessing recipes, refer to [**`docs/datasets.md`**](../docs/datasets.md).

---

## Directory Structure

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
    │   ├── angry/              # 958 images
    │   ├── disgust/            # 111 images
    │   ├── fear/               # 1,024 images
    │   ├── happy/              # 1,774 images
    │   ├── neutral/            # 1,233 images
    │   ├── sad/                # 1,247 images
    │   └── surprise/           # 831 images
    └── YOLO_format/            # AffectNet formatted for YOLOv8/v9 object and facial analysis
        ├── train/
        ├── valid/
        ├── test/
        └── data.yaml           # YOLO configuration defining 8 affective classes
```

---

## The 6 Benchmark Datasets Summary

| Dataset | Modality | Scale / Scope | Role in VisionEdit |
|:---|:---:|:---|:---|
| **FER-2013** | Emotion (FER) | 35,887 images ($48\times 48$, 7 classes) | Primary training split in `data/DATASET/`; smoothed loss validation |
| **AffectNet** | Emotion (FER) | $>1\text{M}$ images (450k annotated, 8 classes) | YOLO format split in `data/DATASET/YOLO_format/`; cross-dataset test |
| **CK+** | Emotion (FER) | 593 sequences (123 subjects, FACS AUs) | Action Unit verification & Grad-CAM spatial calibration |
| **MS-COCO 2017** | Detection (OD) | 330k images (1.5M instances, 80 classes) | Pretrained cascade weights; 80-class real-time detection |
| **Pascal VOC** | Detection (OD) | 11,530 images (27,450 objects, 20 classes) | IoU alignment calibration & SaIS aspect-ratio verification |
| **Kinetics-400 / HMDB51**| Video Intelligence | 306k clips (10s) + 6.8k clips | End-to-end multi-stream pipeline & Knapsack highlight validation |

> **Note on Raw Dataset Ingestion**:  
> The full raw image corpus (`data/DATASET/`, ~305 MB, 86,412 files) is maintained on local disk and ignored in `.gitignore` to prevent GitHub HTTP payload limits. The lightweight test video `sample_test_video.mp4` and all metadata are tracked.
