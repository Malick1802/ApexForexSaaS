# ApexForex SaaS - Azure Windows Setup Script
# Run this script on your new Windows VM to prepare the environment.

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   APEXFOREX SAAS - AZURE ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    Write-Host "Python not found. Please install Python 3.10+ and Add to PATH." -ForegroundColor Red
    exit
}
Write-Host "Python detected: $(python --version)" -ForegroundColor Green

# 2. Setup Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 3. Install Dependencies
Write-Host "Installing dependencies... (This may take a few minutes)" -ForegroundColor Yellow
.\venv\Scripts\pip install -r requirements.txt

# 4. Create local directories if missing
Write-Host "Creating system directories..." -ForegroundColor Yellow
$dirs = @("logs", "models", "data_cache", "data")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  -> Created $dir"
    }
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   SETUP COMPLETE - NEXT STEPS:" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "1. MANUALLY COPY your '.env' file to this folder." -ForegroundColor Yellow
Write-Host "2. MANUALLY COPY your 'models/' folder contents to the 'models' folder here." -ForegroundColor Yellow
Write-Host "3. INSTALL MetaTrader 5 Terminal and log in." -ForegroundColor Yellow
Write-Host "4. RUN '.\venv\Scripts\python.exe scripts\verify_deployment.py' to check readiness." -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan
