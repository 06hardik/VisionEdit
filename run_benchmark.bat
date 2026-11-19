@echo off
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

python main.py --mode benchmark

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] 'python' command failed. Trying 'py -3.10'...
    py -3.10 main.py --mode benchmark
)

echo.
echo Benchmark generation complete. Check outputs/graphs/ and outputs/results_table.csv.
pause
