@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title Apex Sentinel - Automatic Signal Generator
echo ===================================================
echo   APEX SENTINEL - AUTOMATIC SIGNAL GENERATOR
echo ===================================================
echo.
echo Starting background worker for "90%" signals (Dashboard Default)...
echo Press Ctrl+C to stop.
echo.

:loop
echo [%date% %time%] Starting background worker...
call venv\Scripts\activate.bat
python -m core.executive --win-rate 90%% --interval 2
echo [%date% %time%] Worker crashed or stopped. Restarting in 10 seconds...
timeout /t 10
goto loop
