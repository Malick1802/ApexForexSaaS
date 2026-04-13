
import os
import subprocess

def kill_sentinel():
    print("Searching for rogue sentinel processes via WMIC...")
    
    # Get PIDs of python processes running sentinel.py
    cmd = 'wmic process where "name=\'python.exe\' and commandline like \'%sentinel.py%\'" get processid'
    
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
        pids = [line.strip() for line in output.splitlines() if line.strip().isdigit()]
        
        if not pids:
            print("No running sentinel.py processes found.")
            return

        print(f"Found PIDs: {pids}")
        
        for pid in pids:
            print(f"Killing PID {pid}...")
            os.system(f"taskkill /PID {pid} /F")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    kill_sentinel()
