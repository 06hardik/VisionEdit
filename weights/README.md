# VisionEdit Model Weights & Checkpoints Directory
**Hardik Jindal (Roll No. 2024UCA1878) & Ayush Agarwal (Roll No. 2024UCA1822)**

This directory contains the trained weights, checkpoints, and models utilized across the VisionEdit pipeline.

## 1. Trained Emotion Recognition Models (Stream B)

| Filename | Architecture | Format | Parameters | Input Shape | Provenance / Training Details |
|---|---|---|---|---|---|
| `best_model_weights.h5` | `CLCM_Attention_Hybrid` | Keras HDF5 Weights | 2,414,888 | (128, 128, 3) | Checkpoint saved at epoch with peak validation accuracy (Two-phase training on FER-2013). |
| `final_model.keras` | `CLCM_Attention_Hybrid` | Keras 3 Native Model | 2,414,888 | (128, 128, 3) | Full end-to-end serialized model including spatial attention convolution and 4-tier regularized classification head. |
| `saved_model_tf/` | SavedModel ProtocolBuffer | TensorFlow PB format | 2,414,888 | (128, 128, 3) | TensorFlow C++ / production serving format (`saved_model.pb` + variables). |
| `fer_mini_xception.pt` | `Mini-Xception` | PyTorch checkpoint | 58,231 | (1, 48, 48) | Lightweight PyTorch fallback model for low-compute environments. |

## 2. Object Detection Models (Stream A)

Ultralytics models (`yolov8s.pt`, `yolov9e.pt`, `yolov8s-world.pt`) are cached automatically in the working directory upon initial execution:
- **Stage 1 (Always-On)**: `yolov8s.pt` (28.6 GFLOPs, 11.2M parameters) — Fast anchor-free candidate detection (~9 ms/frame).
- **Stage 2 (Gated Candidate Verification)**: `yolov9e.pt` (189.0 GFLOPs, 57.3M parameters) — High-precision verification triggered when top candidate confidence $\ge 0.60$.
