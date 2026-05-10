#!/bin/bash
# ================================================================
# ApexForexSaaS — Full VM Setup & Training Launcher
# ================================================================
# Run this ONCE on the VM after cloning/pulling from GitHub.
# This sets up the exact same Python environment as local.
#
# Usage:
#   git clone https://github.com/Malick1802/ApexForexSaaS.git
#   cd ApexForexSaaS
#   bash vm_setup.sh
# ================================================================

set -e
cd "$(dirname "$0")"

PYTHON=python3

echo ""
echo "================================================================"
echo "  APEXFOREX VM SETUP — Exact Mirror of Local Environment"
echo "================================================================"
echo ""

# --- 1. Python version check ---
echo "[1/6] Checking Python..."
$PYTHON --version || { echo "[ERROR] Python3 not found. Install Python 3.10+"; exit 1; }

# --- 2. Create virtual environment ---
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "       Virtual environment created."
else
    echo "       Virtual environment already exists, skipping."
fi

source venv/bin/activate

# --- 3. Install all dependencies ---
echo "[3/6] Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "       All dependencies installed."

# --- 4. Create required directories ---
echo "[4/6] Creating required directories..."
mkdir -p logs
mkdir -p models/foundation_v2
mkdir -p artifacts
mkdir -p data_cache
mkdir -p tmp/v2_training
echo "       Directories ready."

# --- 5. Set environment variables ---
echo "[5/6] Configuring environment..."
export TF_ENABLE_ONEDNN_OPTS=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL=3
echo "       oneDNN acceleration enabled."

# --- 6. Start Foundation V2 Training ---
echo "[6/6] Starting Foundation Brain V2 Training in background..."
nohup $PYTHON models/foundation_trainer_v2.py \
    > logs/foundation_v2_training.log 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > logs/foundation_v2_train.pid
echo ""
echo "================================================================"
echo "  ✅ Training started! PID: $TRAIN_PID"
echo "================================================================"
echo ""
echo "  Monitor training:"
echo "    tail -f logs/foundation_v2_training.log"
echo ""
echo "  Start dashboard (new terminal):"
echo "    source venv/bin/activate"
echo "    streamlit run dashboard/app.py --server.port 8505 --server.address 0.0.0.0"
echo ""
echo "  Stop training:"
echo "    kill \$(cat logs/foundation_v2_train.pid)"
echo "================================================================"
