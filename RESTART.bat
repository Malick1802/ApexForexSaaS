@echo off
pushd "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=
title Apex Services - Restarter
echo ===================================================
echo   APEX SERVICES - AUTOMATED RESTART
echo ===================================================
echo.

echo 1. Stopping existing services...
powershell -Command "$p = Get-NetTCPConnection -LocalPort 8505 -ErrorAction SilentlyContinue; if($p){Stop-Process -Id $p.OwningProcess -Force -ErrorAction SilentlyContinue}"
powershell -Command "Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*start_sentinel.bat*' -or $_.CommandLine -like '*run_executive.bat*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -Command "Get-WmiObject Win32_Process | Where-Object { $_.Name -like 'python*' -and ($_.CommandLine -like '*main.py*' -or $_.CommandLine -like '*executive*' -or $_.CommandLine -like '*sentinel*' -or $_.CommandLine -like '*app.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo    Wait 2 seconds for cleanup...
powershell -Command "Start-Sleep -Seconds 2"

echo 1.5. Configuring Windows Firewall...
powershell -Command "New-NetFirewallRule -DisplayName 'Apex Dashboard' -Direction Inbound -LocalPort 8505 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue"

echo 1.7. Ensuring Environment Synergy...
echo Ensuring dependencies (pandas-ta)...
.\venv\Scripts\pip.exe install pandas-ta --quiet
echo [DISABLED] Platt Scaling Calibration is disabled per user requirements. Skipping recalibration.

echo 2. Starting Apex Sentinel (Watchdog)...
start "Apex Sentinel" cmd /c "start_sentinel.bat"

echo 3. Starting Apex Executive (The Brain)...
start "Apex Executive" cmd /c "run_executive.bat"

echo 4. Starting Apex Dashboard...
start "Apex Dashboard" .\venv\Scripts\python.exe -X utf8 -m streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0

echo.
echo ===================================================
echo   RESTART COMPLETE
echo   Three windows should now be open on your taskbar.
echo ===================================================
powershell -Command "Start-Sleep -Seconds 5"
exit
