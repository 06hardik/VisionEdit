# Research Findings — 10 Papers Reviewed

## Facial Emotion Recognition (5 Papers)

### [A1] Gursesli et al. 2024
**Title:** Multitask, Multi-label and Multi-domain Learning with CNNs for Emotion Recognition

**What they do:** Multi-task CNN trained on multiple datasets simultaneously.

**Strengths:**
- Learns shared representations across datasets
- Multi-label output allows co-occurring emotions

**Limitations identified:**
- Confusion matrix shows F1 ≈ 40% for disgust and fear
- No temporal modeling for video
- Training data leakage between tasks

**Our fix:** Binary expert CNNs dedicated to disgust and fear only.

---

### [A2] Abbas et al. 2025 ★ (Primary Paper)
**Title:** Facial Emotion Recognition (FER) Through Custom Lightweight CNN Model

**What they do:** CLCM — a custom lightweight CNN (~2.4M params) evaluated on CK+, JAFFE, FER2013.

**Strengths:**
- 97.78% accuracy on CK+ (controlled lab data)
- Lightweight — deployable on edge devices

**Limitations identified:**
- Tested only on controlled lab datasets (CK+ poses, JAFFE frontal faces)
- Class imbalance not addressed — happy/neutral dominate
- No video-temporal modeling
- No explainability

**Our fixes:**
- WeightedFERLoss with inverse-sqrt-frequency class weights
- LSTM temporal aggregator for video
- Grad-CAM explainability layer

---

### [A3] Punuri et al. 2024
**Title:** Decoding Human Facial Emotions: A Ranking Approach Using Explainable AI

**What they do:** VGG16 + Layer-wise Relevance Propagation (LRP) for FER explainability. Relevance scores binned into intensity ranks.

**Strengths:**
- Novel intensity ranking (MINIMAL/AVERAGE/STRONG)
- Pixel-level attribution

**Limitations identified:**
- LRP only computed on correctly classified images — cannot explain wrong predictions
- VGG16 is 138M params — heavy for video inference
- 10-interval binning is arbitrary

**Our fixes:**
- Grad-CAM (works on ALL frames including misclassified)
- 3-tier threshold: mean(CAM) < 0.30 / 0.60 / ≥ 0.60
- Lighter backbone (CLCM 2.4M vs VGG16 138M)

---

### [A4] Salman et al. 2025
**Title:** Mixture of Emotion-Dependent Experts: Facial Expressions Recognition in Videos

**What they do:** 8 MobileNetV2 expert networks — one per emotion class — all running on every frame.

**Strengths:**
- Per-emotion specialists improve minority class accuracy
- 96.26% on AffectNet (7-class)

**Limitations identified:**
- 8 × ~4.1M = 32.76M parameters always active — very expensive
- All 8 experts run on every frame regardless of context
- No Grad-CAM or interpretability

**Our fix:** 2 lightweight experts (disgust, fear only) triggered only when primary confidence < 0.60 (~15-20% of frames).

---

### [A5] Kosta et al. 2023
**Title:** EmotionNet-X: An Optimized CNN Architecture for Robust Facial Emotion Analysis

**What they do:** Compact CNN (EmotionNet-X) optimised for FER efficiency. Achieves competitive accuracy at lower FLOP count.

**Strengths:**
- Efficient architecture with depthwise separable convolutions
- Good accuracy/parameter trade-off

**Limitations identified:**
- Frame-level inference only — no temporal modeling
- Per-frame predictions are noisy in video
- No aggregation strategy for clip-level labels

**Our fix:** LSTM rolling window aggregator (window=10 frames per face track).

---

## Object Detection (5 Papers)

### [B1] Hua et al. 2025
**Title:** A Benchmark Review of YOLO Algorithm Developments for Object Detection

**What they do:** Comprehensive benchmark comparing YOLO versions v1–v10 on COCO, VOC, custom datasets.

**Strengths:**
- Systematic comparison of all YOLO variants
- Clear speed-accuracy Pareto curves

**Limitations identified:**
- Benchmarks use single-model inference — no cascade considered
- Does not address variable frame-rate video scenarios

**Our fix:** Two-stage cascade (lightweight always-on + heavy gated) achieves ~69.5% FLOP savings.

---

### [B2] Miri et al. 2025
**Title:** A Guide to Image- and Video-Based Small Object Detection Using Deep Learning

**What they do:** Survey of small object detection techniques for maritime surveillance.

**Strengths:**
- Multi-scale feature analysis
- Attention mechanisms for small objects

**Limitations identified:**
- Salience defined purely by confidence score — no stability or persistence
- Surveillance-domain focused — may not generalise to highlight detection

**Our fix:** Decomposed salience: O_i = 0.5·conf + 0.3·stability + 0.2·persistence.

---

### [B3] Shah et al. 2026
**Title:** A Two-Stage Spatiotemporal CNN-YOLOv9 Framework for Abandoned Object Detection

**What they do:** Spatiotemporal CNN + YOLOv9 cascade for detecting abandoned objects in surveillance video.

**Strengths:**
- Temporal consistency modeling across frames
- Published RR (FLOP reduction ratio) formula

**Limitations identified:**
- Tracking horizon τ fixed at 75 frames regardless of video frame rate
- 75 frames at 15fps = 5s (correct), but at 25fps = only 3s

**Our fix:** Adaptive τ = τ_sec × fps (τ_sec = 5s default).

---

### [B4] Yang et al. 2024
**Title:** A²Net: An Anchor-Free Alignment Network for Oriented Object Detection

**What they do:** Anchor-free alignment network with SaLA (Sample Assignment with Label Alignment) training strategy. Introduces SaIS (Sample-wise Intersection Score).

**Strengths:**
- SaIS combines IoU with shape score for better assignment
- Handles oriented bounding boxes (remote sensing)

**Limitations identified:**
- SaIS adds complexity; shape_score computation needs clarification
- Domain-specific to remote sensing (aerial images)

**Our adaptation:** SaIS = IoU + 0.5 × shape_score as a training-time label assignment — zero inference cost.

---

### [B5] Ramos et al. 2025
**Title:** A Decade of You Only Look Once (YOLO) for Object Detection: A Review

**What they do:** Comprehensive review of YOLO evolution, anchor-free paradigms, decoupled heads.

**Strengths:**
- Clear taxonomy of architectural improvements
- Analysis of anchor-free advantages

**Limitations identified:**
- Review only — no experimental validation of anchor-free inference
- Does not address cascade architectures

**Our fix:** Validates YOLOv8-S (anchor-free, decoupled head) as Stage 1 of our cascade, confirming best speed-accuracy from B1/B5 analysis.
