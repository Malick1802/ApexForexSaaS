@echo off
pushd "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=
title Apex Executive - Autonomous Trading Engine

echo =====================================================================
echo   APEX EXECUTIVE - 24/7 AUTONOMOUS RESILIENT RUNNER
echo =====================================================================
echo.

:loop
echo [%DATE% %TIME%] Launching Apex Executive (main.py)...
echo ---------------------------------------------------------------------
.\venv\Scripts\python.exe -X utf8 main.py
set EXITCODE=%ERRORLEVEL%

echo.
echo =====================================================================
echo [WARNING] Apex Executive exited at %TIME% with code: %EXITCODE%
echo The window is kept open to prevent data loss or silent shutdown.
echo Auto-restarting in 5 seconds... (Press Ctrl+C to abort)
echo =====================================================================
timeout /t 5
goto loop
