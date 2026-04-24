@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title Apex Services - Restarter
echo ===================================================
echo   APEX SERVICES - AUTOMATED RESTART
echo ===================================================
echo.

echo 1. Stopping existing services...
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8505 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
powershell -Command "Get-Process python* | Where-Object {$_.CommandLine -like '*main.py*' -or $_.CommandLine -like '*executive.py*' -or $_.CommandLine -like '*sentinel.py*'} | Stop-Process -Force -ErrorAction SilentlyContinue"
echo    Wait 2 seconds for cleanup...
timeout /t 2 /nobreak > nul

echo 1.5. Configuring Windows Firewall...
powershell -Command "New-NetFirewallRule -DisplayName 'Apex Dashboard' -Direction Inbound -LocalPort 8505 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue"

echo 1.7. Recalibrating AI Fleet (Ensuring Environment Synergy)...
echo Ensuring dependencies (pandas-ta)...
.\venv\Scripts\pip.exe install pandas-ta --quiet
.\venv\Scripts\python.exe scripts/recalibrate_fleet.py

echo 2. Starting Apex Sentinel (Watchdog)...
start "Apex Sentinel" cmd /c "start_sentinel.bat"

echo 3. Starting Apex Executive (The Brain)...
start "Apex Executive" .\venv\Scripts\python.exe main.py

echo 4. Starting Apex Dashboard...
start "Apex Dashboard" .\venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0

echo.
echo ===================================================
echo   RESTART COMPLETE
echo   Three windows should now be open on your taskbar.
echo ===================================================
timeout /t 5
exit
