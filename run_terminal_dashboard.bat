@echo off
TITLE VisionEdit - AI Command Center Backend Processing Terminal
echo ======================================================================
echo   Launching VisionEdit High-Tech Terminal Dashboard
echo   Students: Hardik Jindal (2024UCA1878) ^& Ayush Agarwal (2024UCA1822)
echo ======================================================================
echo.

python scripts/terminal_dashboard.py --fast

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Standard python command failed. Trying py -3.10...
    py -3.10 scripts/terminal_dashboard.py --fast
)

echo.
echo ======================================================================
echo   Ready for Teacher Submission Screenshot Capture.
echo ======================================================================
pause
