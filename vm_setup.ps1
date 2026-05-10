# ================================================================
# ApexForexSaaS — Windows VM Setup & Training Launcher
# Run in PowerShell: powershell -ExecutionPolicy Bypass -File vm_setup.ps1
# ================================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$SEP = "-" * 64

Write-Host ""
Write-Host $SEP -ForegroundColor Cyan
Write-Host "  APEXFOREX VM SETUP" -ForegroundColor Cyan
Write-Host $SEP -ForegroundColor Cyan
Write-Host ""

# 1. Find Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$PYTHON = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") { $PYTHON = $cmd; Write-Host "       Found: $ver" -ForegroundColor Green; break }
    } catch {}
}
if (-not $PYTHON) { Write-Host "[ERROR] Python 3 not found." -ForegroundColor Red; exit 1 }

# 2. Virtual environment
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    & $PYTHON -m venv venv
}
$PIP   = ".\venv\Scripts\pip.exe"
$PYEXE = ".\venv\Scripts\python.exe"
Write-Host "       Done." -ForegroundColor Green

# 3. Install dependencies
Write-Host "[3/6] Installing packages (may take a few minutes)..." -ForegroundColor Yellow
& $PYEXE -m pip install --upgrade pip --quiet --no-warn-script-location 2>$null
& $PYEXE -m pip install -r requirements.txt --quiet --no-warn-script-location
Write-Host "       All packages installed." -ForegroundColor Green

# 4. Create directories
Write-Host "[4/6] Creating required directories..." -ForegroundColor Yellow
foreach ($d in @("logs", "models\foundation_v2", "artifacts", "data_cache", "tmp\v2_training")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "       Done." -ForegroundColor Green

# 5. Environment variables
Write-Host "[5/6] Configuring environment..." -ForegroundColor Yellow
$env:TF_ENABLE_ONEDNN_OPTS = "1"
$env:PYTHONUTF8            = "1"
$env:PYTHONIOENCODING      = "utf-8"
$env:TF_CPP_MIN_LOG_LEVEL  = "3"
Write-Host "       oneDNN enabled." -ForegroundColor Green

# 6. Start training
Write-Host "[6/6] Starting Foundation Brain V2 Training..." -ForegroundColor Yellow
if (Test-Path "logs\foundation_v2_training.log") {
    Clear-Content "logs\foundation_v2_training.log" -ErrorAction SilentlyContinue
}

$proc = Start-Process -FilePath $PYEXE `
    -ArgumentList "models\foundation_trainer_v2.py" `
    -RedirectStandardOutput "logs\foundation_v2_training.log" `
    -RedirectStandardError  "logs\foundation_v2_error.log" `
    -NoNewWindow -PassThru

$proc.Id | Out-File -FilePath "logs\foundation_v2_train.pid" -Encoding ascii

Write-Host ""
Write-Host $SEP -ForegroundColor Green
Write-Host "  Training started! PID: $($proc.Id)" -ForegroundColor Green
Write-Host $SEP -ForegroundColor Green
Write-Host ""
Write-Host "  Watch training:"
Write-Host "    Get-Content logs\foundation_v2_training.log -Wait -Tail 20" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start dashboard (new window):"
Write-Host "    .\venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8505 --server.address 0.0.0.0" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Stop training:"
Write-Host "    Stop-Process -Id (Get-Content logs\foundation_v2_train.pid)" -ForegroundColor Cyan
Write-Host $SEP -ForegroundColor Green
