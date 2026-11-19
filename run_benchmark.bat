@echo off
setlocal enabledelayedexpansion
TITLE VisionEdit - Academic Benchmark & Papers Comparison Generator

echo ======================================================================
echo   VisionEdit: Academic Benchmark Generator
echo   Students: Hardik Jindal (2024UCA1878) ^& Ayush Agarwal (2024UCA1822)
echo ======================================================================
echo   Synthesizes empirical outcomes across all 10 researched papers:
echo     - 5 Emotion Recognition Papers (Salman, Punuri, Gursesli, Abbas, Kosta)
echo     - 5 Object Detection Papers (Hua, Miri, Shah, Yang, Ramos)
echo   Generates comparative figures, Pareto frontiers, and tables.
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

"%PY_BIN%" main.py --mode benchmark

echo.
echo Benchmark generation complete. Check outputs/graphs/ and outputs/results_table.csv.
pause
