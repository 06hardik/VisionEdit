@echo off
setlocal enabledelayedexpansion
TITLE VisionEdit - AI Command Center Backend Processing Terminal

echo ======================================================================
echo   Launching VisionEdit High-Tech Terminal Dashboard
echo   Students: Hardik Jindal (2024UCA1878) ^& Ayush Agarwal (2024UCA1822)
echo ======================================================================
echo.

:: Detect valid Python executable
set "PY_BIN="

:: 1. Check if uv python in .local\bin exists
if exist "%USERPROFILE%\.local\bin\python3.15.exe" (
    set "PY_BIN=%USERPROFILE%\.local\bin\python3.15.exe"
    goto :RunScript
)

:: 2. Check if uv python in AppData exists
if exist "%APPDATA%\uv\python\cpython-3.15-windows-x86_64-none\python.exe" (
    set "PY_BIN=%APPDATA%\uv\python\cpython-3.15-windows-x86_64-none\python.exe"
    goto :RunScript
)

if exist "%APPDATA%\uv\python\cpython-3.15.2-windows-x86_64-none\python.exe" (
    set "PY_BIN=%APPDATA%\uv\python\cpython-3.15.2-windows-x86_64-none\python.exe"
    goto :RunScript
)

:: 3. Test standard python command
python -c "import sys" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_BIN=python"
    goto :RunScript
)

:: 4. Test py launcher
py -c "import sys" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_BIN=py"
    goto :RunScript
)

:: 5. Search standard AppData Python folders
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
    if exist "%%D\python.exe" (
        set "PY_BIN=%%D\python.exe"
        goto :RunScript
    )
)

:RunScript
if not defined PY_BIN (
    echo [ERROR] No working Python installation found!
    echo Please install Python 3.10+ or disable Windows App execution aliases.
    pause
    exit /b 1
)

"%PY_BIN%" scripts\terminal_dashboard.py %*

echo.
echo ======================================================================
echo   Ready for Teacher Submission Screenshot Capture.
echo ======================================================================
pause
