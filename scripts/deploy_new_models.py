
import os
import shutil
import time
from datetime import datetime

MODELS_DIR = "models"
SPECIALIST_DIR = os.path.join(MODELS_DIR, "specialist")
BACKUP_DIR = os.path.join(MODELS_DIR, f"specialist_backup_{int(time.time())}")

def deploy_models():
    print("="*60)
    print("🚀 DEPLOYING NEW MODELS (Feb 10)")
    print("="*60)
    
    # 1. Backup
    if os.path.exists(SPECIALIST_DIR):
        print(f"Backing up current models to {BACKUP_DIR}...")
        try:
            shutil.copytree(SPECIALIST_DIR, BACKUP_DIR)
            print("Backup complete.")
        except Exception as e:
            print(f"Backup failed: {e}")
            return
    else:
        print("No existing specialist directory to backup.")
        os.makedirs(SPECIALIST_DIR, exist_ok=True)

    # 2. Deploy
    deployed_count = 0
    errors = 0
    
    # Walk models dir to find "90" folders
    print("Searching for new models...")
    for pair in os.listdir(MODELS_DIR):
        pair_path = os.path.join(MODELS_DIR, pair)
        if not os.path.isdir(pair_path) or pair in ['specialist', 'binary', 'trained', 'enhanced', '__pycache__']:
            continue
            
        # Check for '90' subfolder
        new_model_path = os.path.join(pair_path, "90")
        if os.path.exists(new_model_path):
            print(f"Deploying {pair}...")
            
            # Target dir
            target_pair_dir = os.path.join(SPECIALIST_DIR, pair)
            os.makedirs(target_pair_dir, exist_ok=True)
            
            # Copy BUY and SELL folders
            for direction in ['BUY', 'SELL']:
                src = os.path.join(new_model_path, direction)
                dst = os.path.join(target_pair_dir, direction)
                
                if os.path.exists(src):
                    try:
                        # Remove existing dst if exists
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        # print(f"  - {direction} deployed.")
                    except Exception as e:
                        print(f"  ! Error deploying {pair} {direction}: {e}")
                        errors += 1
                else:
                    # print(f"  - {direction} not found in source.")
                    pass
            
            deployed_count += 1
            
    print("-" * 60)
    print(f"Deployment Complete.")
    print(f"Models Updated: {deployed_count}")
    print(f"Errors: {errors}")
    print("-" * 60)

if __name__ == "__main__":
    deploy_models()
