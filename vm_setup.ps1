# ================================================================
# ApexForexSaaS — Windows VM Setup & Training Launcher
# ================================================================
# Run in PowerShell from the ApexForexSaaS folder:
#   powershell -ExecutionPolicy Bypass -File vm_setup.ps1
# ================================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  APEXFOREX VM SETUP — Windows Environment" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Find Python ---
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$PYTHON = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") {
            $PYTHON = $cmd
            Write-Host "       Found: $ver" -ForegroundColor Green
            break
        }
    } catch {}
}
if (-not $PYTHON) {
    Write-Host "[ERROR] Python 3 not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# --- 2. Create virtual environment ---
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    & $PYTHON -m venv venv
    Write-Host "       Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "       Virtual environment already exists, skipping." -ForegroundColor Green
}

$PIP  = ".\venv\Scripts\pip.exe"
$PYEXE = ".\venv\Scripts\python.exe"

# --- 3. Install dependencies ---
Write-Host "[3/6] Installing Python packages from requirements.txt..." -ForegroundColor Yellow
Write-Host "       (This may take a few minutes on first run)" -ForegroundColor Gray
& $PIP install --upgrade pip --quiet
& $PIP install -r requirements.txt --quiet
Write-Host "       All packages installed." -ForegroundColor Green

# --- 4. Create required directories ---
Write-Host "[4/6] Creating required directories..." -ForegroundColor Yellow
$dirs = @("logs", "models\foundation_v2", "artifacts", "data_cache", "tmp\v2_training")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "       Directories ready." -ForegroundColor Green

# --- 5. Set environment variables ---
Write-Host "[5/6] Configuring environment..." -ForegroundColor Yellow
$env:TF_ENABLE_ONEDNN_OPTS   = "1"
$env:PYTHONUTF8              = "1"
$env:PYTHONIOENCODING        = "utf-8"
$env:TF_CPP_MIN_LOG_LEVEL    = "3"
Write-Host "       oneDNN CPU acceleration enabled." -ForegroundColor Green

# --- 6. Start training ---
Write-Host "[6/6] Starting Foundation Brain V2 Training..." -ForegroundColor Yellow

# Clear any stale log
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
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Training started! PID: $($proc.Id)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Monitor training log:" -ForegroundColor White
Write-Host "    Get-Content logs\foundation_v2_training.log -Wait -Tail 20" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start the dashboard (in a new PowerShell window):" -ForegroundColor White
Write-Host "    .\venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8505 --server.address 0.0.0.0" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Stop training:" -ForegroundColor White
Write-Host "    Stop-Process -Id (Get-Content logs\foundation_v2_train.pid)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Green
