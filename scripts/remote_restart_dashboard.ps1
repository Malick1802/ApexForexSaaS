# remote_restart_dashboard.ps1
# Run this on the Azure VM to kill and restart the dashboard process

Set-Location "C:\Users\Malick1802\ApexForexSaaS"

# Step 1: Pull latest code
Write-Host "Pulling latest code..."
git pull origin main

# Step 2: Kill any process on port 8505
Write-Host "Killing old dashboard process..."
$conn = Get-NetTCPConnection -LocalPort 8505 -ErrorAction SilentlyContinue
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Write-Host "Killed PID $($conn.OwningProcess)"
} else {
    Write-Host "No process found on port 8505"
}

# Step 3: Kill all streamlit / python processes related to dashboard
Write-Host "Killing lingering streamlit processes..."
Get-Process python* -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*\venv\*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

# Step 4: Restart via scheduled task if it exists, otherwise start directly
Write-Host "Restarting dashboard..."
$taskExists = Get-ScheduledTask -TaskName "ApexForexDashboard" -ErrorAction SilentlyContinue
if ($taskExists) {
    Start-ScheduledTask -TaskName "ApexForexDashboard"
    Write-Host "Started via Scheduled Task"
} else {
    $taskList = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -like "*Apex*" }
    if ($taskList) {
        $taskList | ForEach-Object {
            Write-Host "Found task: $($_.TaskName)"
            Start-ScheduledTask -TaskName $_.TaskName
        }
    } else {
        Write-Host "No scheduled task found, starting directly..."
        Start-Job -ScriptBlock {
            Set-Location "C:\Users\Malick1802\ApexForexSaaS"
            & ".\venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0
        }
    }
}

Start-Sleep -Seconds 3

# Step 5: Verify port is now active
$verify = Get-NetTCPConnection -LocalPort 8505 -ErrorAction SilentlyContinue
if ($verify) {
    Write-Host "SUCCESS: Dashboard is listening on port 8505 (PID $($verify.OwningProcess))"
} else {
    Write-Host "WARNING: Port 8505 not yet active - dashboard may still be starting"
}
