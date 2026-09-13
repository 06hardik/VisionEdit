# VisionEdit

> **Automatic Video Highlight Detection via Multi-Stream Saliency Scoring**

A research-grade video intelligence pipeline that fuses **facial emotion recognition (FER)**, **object detection (OD)**, and **technical quality assessment** into a single saliency score, then assembles a highlight reel from the top-scoring scenes.

Built as part of a research assignment reviewing 10 published papers across FER and object detection and addressing their key limitations.

---

## Architecture

`
Source Video
    │
    ▼
Phase 1: Scene Detector (PySceneDetect ContentDetector, threshold=27)
    │
    ├──── Stream A: Semantic / Object Detection (O_i)
    │         YOLOv8-S  →  gated YOLOv9-E  →  Kalman Tracker  →  Decomposed Salience
    │
    ├──── Stream B: Affective / Facial Emotion Recognition (E_i)
    │         CLCM Backbone  →  Grad-CAM Intensity  →  Binary Experts  →  LSTM Aggregator
    │
    └──── Stream C: Quality Assessment (Q_i / M_i)
              Laplacian Blur Gate  →  Motion Energy
                                │
                                ▼
              Phase 5: Weighted Fusion
              S_i = (0.40·E_i + 0.35·O_i + 0.25·M_i) × 𝟙(Q_i ≥ θ)
                                │
                                ▼
              Phase 6: Knapsack Selection → output/highlight.mp4
`

See [outputs/architecture_diagram.html](outputs/architecture_diagram.html) for the full interactive diagram.

---

## Research Foundation

| Paper | Finding | Our Fix |
|---|---|---|
| [A1] Gursesli 2024 | disgust/fear F1 ~40% | Binary expert CNNs for hard classes |
| [A2] Abbas 2025 ★ | CLCM tested on lab-only CK+ | WeightedFERLoss (inverse-sqrt-freq) |
| [A3] Punuri 2024 | LRP fails on misclassified frames | Grad-CAM (works on ALL frames) |
| [A4] Salman 2025 | 8 full backbones always-on (32.76M) | 2 experts, triggered <60% conf only |
| [A5] Kosta 2023 | No temporal emotion modeling | LSTM rolling window (10 frames) |
| [B1] Hua 2025 | Single heavy YOLO on every frame | Two-stage cascade (~69.5% FLOP saving) |
| [B2] Miri 2025 | Raw confidence only, no stability | Decomposed salience = conf+stability+persist |
| [B3] Shah 2026 | Fixed τ=75 frames (ignores fps) | Adaptive τ = τ_sec × fps |
| [B4] Yang 2024 | IoU-only label assignment | SaIS = IoU + 0.5×shape_score |
| [B5] Ramos 2025 | No anchor-free validation | YOLOv8-S anchor-free, decoupled head |

★ Primary paper — CLCM is our main FER backbone.

---

## Quick Start

### 1. Install dependencies

`ash
pip install -r requirements.txt
`

### 2. Download model weights

`ash
python scripts/download_weights.py
`

### 3. Configure paths

Edit [config.yaml](config.yaml) — set your video source path:

`yaml
pipeline:
  input_path: "path/to/your/video.mp4"
`

### 4. Run the pipeline

`ash
python main.py
`

Output highlight reel is saved to output/highlight.mp4.

---

## Generating Results / Presentation

`ash
# Score real Kinetics-400 clips + generate Grad-CAM heatmaps
python scripts/generate_presentation_outputs.py

# Build the full HTML presentation (metrics table + embedded heatmaps)
python scripts/build_presentation.py

# Open the presentation
start outputs/presentation.html
`

### Downloading Kinetics-400 clips

`ash
# Install yt-dlp first
pip install yt-dlp

# Download our 80 targeted classes (uses python -m yt_dlp, no exe needed)
python scripts/download_kinetics400.py
`

The script downloads only the 80 classes relevant to our three streams (~30 clips/class) from the official Kinetics-400 validation split.

---

## Project Structure

`
VisionEdit/
├── visionedit/                  # Main package
│   ├── streams/                 # Three intelligence streams
│   │   ├── affective.py         # Stream B: FER pipeline
│   │   ├── fer_model.py         # CLCM backbone + binary experts
│   │   ├── object_detection.py  # Stream A: Two-stage OD cascade
│   │   ├── semantic.py          # OD salience scoring
│   │   ├── temporal.py          # Stream C: Motion + quality
│   │   └── gradcam.py           # Grad-CAM visualiser
│   ├── fusion/
│   │   └── scorer.py            # Weighted fusion (fuse_rich)
│   ├── segmentation/
│   │   └── scene_detector.py    # Phase 1: scene cuts
│   ├── rendering/
│   │   └── assembler.py         # Phase 6: highlight assembly
│   ├── utils/
│   │   └── data_types.py        # StreamScores dataclass
│   └── pipeline.py              # End-to-end orchestrator
│
├── scripts/
│   ├── generate_presentation_outputs.py   # Grad-CAM + metrics from real clips
│   ├── build_presentation.py              # Full HTML presentation builder
│   ├── generate_results.py                # Academic results tables
│   ├── download_kinetics400.py            # Dataset download (yt-dlp)
│   └── download_weights.py                # Model weights downloader
│
├── tests/
│   └── datasets/
│       ├── kinetics400.py        # Kinetics-400 loader + class groups
│       └── hmdb51.py             # HMDB51 loader
│
├── outputs/                      # Generated outputs (tracked selectively)
│   ├── architecture_diagram.html
│   ├── presentation.html
│   ├── real_clip_results.csv
│   ├── metrics_summary.csv
│   └── gradcam/                  # Grad-CAM PNGs (gitignored — large)
│
├── CV_PAPERS/                    # 10 research papers reviewed
├── config.yaml                   # All pipeline hyperparameters
├── requirements.txt
└── main.py
`

---

## Results (Intermediate — 7 Kinetics-400 Classes, 116 Clips)

| Class | E_i | O_i | M_i | **S_i** | Emotion |
|---|---|---|---|---|---|
| applauding | +0.751 | 0.585 | 0.738 | **0.690** | happy / STRONG |
| bouncing on trampoline | +0.541 | 0.856 | 0.688 | **0.578** | surprise / AVERAGE |
| bowling | +0.104 | 0.919 | 0.691 | **0.536** | neutral / MINIMAL |
| blowing out candles | +0.630 | 0.900 | 0.757 | **0.438** | happy / STRONG |
| javelin throw | +0.058 | 0.858 | 0.668 | **0.491** | neutral / MINIMAL |
| archery | +0.019 | 0.753 | 0.505 | **0.405** | neutral / MINIMAL |
| answering questions | +0.082 | 0.898 | 0.459 | **0.387** | neutral / MINIMAL |

O_i and M_i are computed from **real YOLOv8-nano inference** on real Kinetics-400 video frames.

---

## License

For academic/educational purposes only.
