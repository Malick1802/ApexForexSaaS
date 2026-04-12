@echo off
title Apex Services - Restarter
echo ===================================================
echo   APEX SERVICES - AUTOMATED RESTART
echo ===================================================
echo.

echo 1. Stopping existing services...
powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8505 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
powershell -Command "Get-Process python* | Where-Object {$_.CommandLine -like '*executive.py*'} | Stop-Process -Force -ErrorAction SilentlyContinue"
echo    Wait 2 seconds for cleanup...
timeout /t 2 /nobreak > nul

echo 1.5. Configuring Windows Firewall...
powershell -Command "New-NetFirewallRule -DisplayName 'Apex Dashboard' -Direction Inbound -LocalPort 8505 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue"

echo 2. Starting Apex Sentinel (Watchdog)...
start "Apex Sentinel" cmd /c "start_sentinel.bat"

echo 3. Starting Apex Dashboard...
start "Apex Dashboard" .\venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0

echo.
echo ===================================================
echo   RESTART COMPLETE
echo   Two windows should now be open on your taskbar.
echo ===================================================
pause
