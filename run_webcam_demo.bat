@echo off
TITLE VisionEdit - Live Webcam Emotion Recognition Demo
echo ======================================================================
echo   VisionEdit: Live Webcam Facial Emotion Recognition Demo
echo   Students: Hardik Jindal (2024UCA1878) ^& Ayush Agarwal (2024UCA1822)
echo ======================================================================
echo   - Model: CLCM_Attention_Hybrid (MobileNetV2 + Spatial Attention)
echo   - XAI: Real-Time Grad-CAM Heatmap Overlay
echo   - Scoring: LRP/Grad-CAM Intensity Ranking (Minimal/Average/Strong)
echo   - Fusion: Confidence-Weighted Temporal Aggregation
echo.
echo   [!] Click on camera window and press 'q' or ESC to stop and view results.
echo ======================================================================
echo.

python main.py --mode webcam

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] 'python' command failed. Trying 'py -3.10'...
    py -3.10 main.py --mode webcam
)

echo.
echo Demo finished. Recorded video saved in outputs/demos/.
pause
