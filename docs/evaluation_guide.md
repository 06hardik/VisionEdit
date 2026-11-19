# VisionEdit — Teacher Evaluation & Grading Quickstart Guide
**Students:**  
- **Hardik Jindal** — Roll No. `2024UCA1878`  
- **Ayush Agarwal** — Roll No. `2024UCA1822`  

---

## 🎯 Quick Verification Checklist for Evaluator / Teacher

This guide enables the evaluating teacher to verify all claims, run demos, inspect trained models, and review comparative benchmarks in under 5 minutes.

---

### 1. One-Click Verification (Windows)
Pre-configured `.bat` scripts are in the root of `VisionEdit/`:
- 🟢 **`run_webcam_demo.bat`**: Launches the live webcam facial emotion recognition demo with real-time Grad-CAM XAI and intensity ranking.
- 🟢 **`run_video_pipeline.bat`**: Executes the complete 6-stage video highlight pipeline on `data/sample_test_video.mp4` and exports `outputs/highlight.mp4`.
- 🟢 **`run_benchmark.bat`**: Generates all comparative academic benchmark tables and figures against the 10 reviewed papers.

---

### 2. CLI Execution Commands

From the `VisionEdit/` directory:

```bash
# 1. Full Multi-Stream Video Highlight Pipeline
python main.py --input data/sample_test_video.mp4 --duration 30

# 2. Live Webcam Emotion Recognition Demo (Grad-CAM XAI)
python main.py --mode webcam

# 3. Video Emotion Analysis with Temporal Progression Plot
python main.py --mode emotion-video --input data/sample_test_video.mp4

# 4. Single Image Prediction with 3-Panel Grad-CAM Visualization
python main.py --mode predict-image

# 5. Generate Benchmark Comparison Figures & Tables
python main.py --mode benchmark

# 6. Run Automated Test Suite
python -m unittest tests/test_integration.py -v
```

---

### 3. Key Deliverables & Where to Inspect Them

| Evaluated Deliverable | Location in Repository | Description |
|:---|:---|:---|
| **Trained Emotion Model** | `weights/final_model.keras` & `weights/best_model_weights.h5` | Trained `CLCM_Attention_Hybrid` model ($2.41\text{M}$ params). |
| **Recorded Webcam Demo** | `outputs/demos/my_webcam_demo.mp4` | Live recording showing real-time Grad-CAM PiP overlay. |
| **Verified Run Video** | `outputs/demos/verified_run.mp4` | Verification recording of highlight extraction. |
| **Comparative Benchmark Graphs** | `outputs/graphs/` | 24 high-resolution plots comparing against papers. |
| **Full Project Academic Report** | `docs/SUBMISSION_REPORT.md` | Formal capstone submission report for grading. |
| **10 Research Papers Analysis** | `docs/RESEARCH_PAPERS_ANALYSIS.md` | In-depth critique of the 10 reviewed papers (5 OD + 5 FER). |
| **Reviewed Research Paper PDFs** | `CV_PAPERS/` | Full original PDF publications for both domains. |
| **Full Saliency Results Table** | `outputs/results_table.csv` & `outputs/metrics_summary.csv` | Numerical test-set metrics and FLOP counts. |

---

### 4. Summary of Research Breakthroughs for Grading

1. **Parameter Reduction (ER-1 MoEDE)**: Reduced parameter count from $32.8\text{M}$ down to **$2.41\text{M}$** ($92.7\%$ reduction) via Spatial Attention Hybrid.
2. **Class Imbalance Resolution (ER-3 Gursesli)**: Rescued disgust recall from $22.3\%$ to **$62.2\%$** ($+39.9\%$ gain) via square-root smoothed inverse loss.
3. **Real-Time XAI & Temporal Attention (ER-2 Punuri)**: Replaced brittle VGG-16 LRP with lightweight Grad-CAM ($12\text{ms}$ vs $142\text{ms}$), linking intensity ranks to temporal fusion.
4. **Compute Savings (OD-1 Hua)**: Two-stage gated detection cascade (YOLOv8s + YOLOv9e) achieves **$69.5\%$ FLOP reduction**.
5. **Adaptive Horizon (OD-3 Shah)**: Framerate-adaptive tracking horizon ($\tau = \tau_{\text{base}} \times \text{FPS}$) eliminates framerate tracking bias.
6. **Combinatorial Assembly**: Solves highlight curation via exact $0/1$ Knapsack Dynamic Programming under target duration budget $T$.
