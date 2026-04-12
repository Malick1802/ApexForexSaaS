import os
import subprocess
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

# Launch Executive
subprocess.Popen([
    "powershell", "-Command",
    "Start-Process 'venv\\Scripts\\python.exe' -ArgumentList 'run_executive.py --confidence 0.70 --interval 15' -RedirectStandardOutput 'logs\\executive_stdout.log' -RedirectStandardError 'logs\\executive_stderr.log' -WindowStyle Hidden"
], env=env)

# Launch Trainer
subprocess.Popen([
    "powershell", "-Command",
    "Start-Process 'venv\\Scripts\\python.exe' -ArgumentList 'scripts\\train_specialists_progressive.py --target 70 --worker 1 --total-workers 1' -RedirectStandardOutput 'logs\\worker1.log' -RedirectStandardError 'logs\\worker1_err.log' -WindowStyle Hidden"
], env=env)

# Launch Dashboard
subprocess.Popen([
    "powershell", "-Command",
    "Start-Process 'venv\\Scripts\\python.exe' -ArgumentList '-m streamlit run dashboard/app.py --server.port 8505' -RedirectStandardOutput 'logs\\dashboard_stdout.log' -RedirectStandardError 'logs\\executive_stderr.log' -WindowStyle Hidden"
], env=env)

print("All processes launched in fully detached background mode.")
