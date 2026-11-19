@echo off
setlocal enabledelayedexpansion
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

"%PY_BIN%" main.py --input data/sample_test_video.mp4 --duration 30

echo.
echo Highlight extraction finished. Output video saved in outputs/highlight.mp4.
pause
