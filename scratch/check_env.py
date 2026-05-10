import os
import sys

print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

venv_path = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
print(f"Expected venv python path: {venv_path}")
print(f"Exists: {os.path.exists(venv_path)}")
