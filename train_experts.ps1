# ApexForex Expert Trainer Script
$ErrorActionPreference = "Stop"

Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "APEXFOREX EXPERT / SPECIALIST TRAINING" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan

# 1. Activate Virtual Environment
Write-Host "[1/3] Activating environment..." -ForegroundColor Yellow
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . ".\venv\Scripts\Activate.ps1"
    Write-Host "   Done." -ForegroundColor Green
} else {
    Write-Host "   Virtual environment not found! Run vm_setup.ps1 first." -ForegroundColor Red
    exit
}

# 2. Cleanup old logs
Write-Host "[2/3] Cleaning previous expert logs..." -ForegroundColor Yellow
if (Test-Path "logs\expert_training.log") { Remove-Item "logs\expert_training.log" -Force }
if (Test-Path "logs\expert_error.log") { Remove-Item "logs\expert_error.log" -Force }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
Write-Host "   Done." -ForegroundColor Green

# 3. Start the Training Process
Write-Host "[3/3] Starting Expert Model Manufacturing (60%, 70%, 80%, 90%, 100%)..." -ForegroundColor Yellow

# Start the trainer in the background
$process = Start-Process -FilePath "python" `
    -ArgumentList "-u models\win_rate_trainer.py" `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput "logs\expert_training.log" `
    -RedirectStandardError  "logs\expert_error.log"

# Save the PID
$process.Id | Out-File "logs\expert_train.pid"

Write-Host "----------------------------------------------------------------" -ForegroundColor Green
Write-Host "Expert Training started! PID: $($process.Id)" -ForegroundColor Green
Write-Host "----------------------------------------------------------------" -ForegroundColor Green
Write-Host ""
Write-Host "This process trains completely isolated models for every pair and tier."
Write-Host "It uses strict 80/20 temporal splits to prevent Data Leakage."
Write-Host "It uses Early Stopping with 'restore_best_weights' to prevent Overfitting."
Write-Host ""
Write-Host "Watch the training live:" -ForegroundColor Cyan
Write-Host "  Get-Content logs\expert_training.log -Wait -Tail 20"
Write-Host ""
Write-Host "To stop the training:" -ForegroundColor Cyan
Write-Host "  Stop-Process -Id (Get-Content logs\expert_train.pid)"
Write-Host "----------------------------------------------------------------" -ForegroundColor Green
