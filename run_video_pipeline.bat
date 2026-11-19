@echo off
TITLE VisionEdit - Full Video Highlight Pipeline
echo ======================================================================
echo   VisionEdit: Full Multi-Stream Video Highlight Detection Pipeline
echo   Students: Hardik Jindal (2024UCA1878) ^& Ayush Agarwal (2024UCA1822)
echo ======================================================================
echo   - Stream A: Two-Stage Cascade Object Detection (YOLOv8s + YOLOv9e)
echo   - Stream B: Affective Saliency (CLCM_Attention_Hybrid + Grad-CAM)
echo   - Stream C: Quality Gating ^& Optical Flow Motion Energy
echo   - Fusion: Multi-Stream Saliency + 0/1 Knapsack Optimization
echo ======================================================================
echo.

python main.py --input data/sample_test_video.mp4 --duration 30

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] 'python' command failed. Trying 'py -3.10'...
    py -3.10 main.py --input data/sample_test_video.mp4 --duration 30
)

echo.
echo Highlight extraction finished. Output video saved in outputs/highlight.mp4.
pause
