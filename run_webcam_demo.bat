@echo off
setlocal enabledelayedexpansion
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

:: Detect valid Python executable
set "PY_BIN="

if exist "%USERPROFILE%\.local\bin\python3.15.exe" (
    set "PY_BIN=%USERPROFILE%\.local\bin\python3.15.exe"
    goto :RunScript
)

if exist "%APPDATA%\uv\python\cpython-3.15-windows-x86_64-none\python.exe" (
    set "PY_BIN=%APPDATA%\uv\python\cpython-3.15-windows-x86_64-none\python.exe"
    goto :RunScript
)

if exist "%APPDATA%\uv\python\cpython-3.15.2-windows-x86_64-none\python.exe" (
    set "PY_BIN=%APPDATA%\uv\python\cpython-3.15.2-windows-x86_64-none\python.exe"
    goto :RunScript
)

python -c "import sys" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_BIN=python"
    goto :RunScript
)

py -c "import sys" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_BIN=py"
    goto :RunScript
)

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
    if exist "%%D\python.exe" (
        set "PY_BIN=%%D\python.exe"
        goto :RunScript
    )
)

:RunScript
if not defined PY_BIN (
    echo [ERROR] No working Python installation found!
    pause
    exit /b 1
)

"%PY_BIN%" main.py --mode webcam

echo.
echo Demo finished. Recorded video saved in outputs/demos/.
pause
