@echo off
title Apex Sentinel - Automatic Signal Generator
echo ===================================================
echo   APEX SENTINEL - AUTOMATIC SIGNAL GENERATOR
echo ===================================================
echo.
echo Starting background worker for "90%" signals (Dashboard Default)...
echo Press Ctrl+C to stop.
echo.

call venv\Scripts\activate.bat
python -m core.executive --win-rate 90%%

pause
