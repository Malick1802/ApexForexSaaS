import subprocess
import time
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "scripts" / "python.exe"

def start_process(cmd_args, log_file, env=None):
    print(f"Starting: {' '.join(cmd_args)}")
    with open(log_file, "a") as f:
        return subprocess.Popen(
            cmd_args,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

# Skip internal taskkill to avoid self-termination
time.sleep(1)

# 2. Start 1 Training Worker (Reduced to 1 to avoid OOM)
worker_env = os.environ.copy()
worker_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
worker1 = start_process(
    [str(VENV_PYTHON), "scripts/train_specialists_progressive.py", "--target", "70", "--worker", "1", "--total-workers", "1"],
    "logs/worker1.log",
    env=worker_env
)

# 3. Start Executive Engine
executive = start_process(
    [str(VENV_PYTHON), "run_executive.py", "--confidence", "0.70", "--interval", "15"],
    "logs/executive_stdout.log"
)

# 4. Start Dashboard
dashboard = start_process(
    [str(VENV_PYTHON), "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8504"],
    "logs/dashboard_stdout.log"
)

print("All processes triggered. Waiting 10 seconds for stabilization...")
time.sleep(10)

# 5. Verification
procs = {"Worker 1": worker1, "Executive": executive, "Dashboard": dashboard}
for name, p in procs.items():
    if p.poll() is None:
        print(f"[OK] {name} is running (PID: {p.pid})")
    else:
        print(f"[FAILED] {name} failed with code {p.returncode}")
