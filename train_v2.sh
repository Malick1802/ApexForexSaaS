#!/bin/bash
# ============================================================
# Foundation Brain V2 - Training Launcher (Linux VM)
# ============================================================
# Run this on your VM after: git pull origin main
# Usage:  bash train_v2.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  APEX FOUNDATION BRAIN V2 — VM TRAINING LAUNCHER"
echo "============================================================"
echo ""

# --- 1. Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

# --- 2. Setup virtualenv if not present ---
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[SETUP] Activating virtual environment..."
source venv/bin/activate

# --- 3. Install dependencies ---
echo "[SETUP] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# --- 4. Create logs directory ---
mkdir -p logs
mkdir -p models/foundation_v2
mkdir -p artifacts

# --- 5. Enable oneDNN + UTF-8 ---
export TF_ENABLE_ONEDNN_OPTS=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo ""
echo "[INFO] Starting Foundation Brain V2 Training..."
echo "[INFO] Logs will be written to: logs/foundation_v2_training.log"
echo "[INFO] Model will be saved to:  models/foundation_v2/foundation_brain.keras"
echo "[INFO] Press Ctrl+C to stop."
echo ""

# --- 6. Start training with live log output ---
nohup python3 models/foundation_trainer_v2.py > logs/foundation_v2_training.log 2>&1 &
TRAIN_PID=$!
echo "[OK] Training started in background. PID: $TRAIN_PID"
echo "$TRAIN_PID" > logs/foundation_v2_train.pid

echo ""
echo "--------------------------------------------------------------"
echo "  Monitor progress with:"
echo "    tail -f logs/foundation_v2_training.log"
echo ""
echo "  Or start the dashboard (port 8505):"
echo "    streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0"
echo "--------------------------------------------------------------"
