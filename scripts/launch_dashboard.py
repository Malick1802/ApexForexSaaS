
import sys
import os
from streamlit.web import cli

if __name__ == '__main__':
    # Set the script path relative to this launcher
    # Assuming launcher is in /scripts, and app is in /dashboard
    # We need to run from project root ideally
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Construct args manually
    sys.argv = [
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.port", "8505",
        "--server.headless", "true"
    ]
    
    print(f"🚀 Launching Dashboard from: {project_root}")
    print(f"🐍 Python Executable: {sys.executable}")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow Check: Found v{tf.__version__} at {tf.__file__}")
    except ImportError as e:
        print(f"❌ TensorFlow Check Failed: {e}")
    
    sys.exit(cli.main())
