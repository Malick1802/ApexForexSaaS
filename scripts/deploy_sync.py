import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    try:
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=== ApexForex SaaS Deployment Sync ===")
    
    # 1. Stash any local changes on the VM to prevent merge conflicts
    print("\n[1/4] Stashing local changes...")
    run_command("git stash")
    
    # 2. Pull latest from main
    print("\n[2/4] Pulling latest code from GitHub...")
    if not run_command("git pull origin main"):
        print("!! Git pull failed. Please check your credentials/network.")
        sys.exit(1)
        
    # 3. Update dependencies
    print("\n[3/4] Checking for dependency updates...")
    venv_pip = Path("venv/Scripts/pip.exe")
    if venv_pip.exists():
        run_command(f"{venv_pip} install -r requirements.txt")
    else:
        run_command("pip install -r requirements.txt")
        
    # 4. Final Instructions
    print("\n[4/4] SYNC COMPLETE")
    print("==================================================")
    print("IMPORTANT: Since you are running in Terminal mode,")
    print("please RESTART your DASHBOARD and SENTINEL windows")
    print("to ensure the new code and fixes are active.")
    print("==================================================")

if __name__ == "__main__":
    main()
