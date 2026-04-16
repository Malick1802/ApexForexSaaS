import os
import zipfile
from pathlib import Path
import datetime

def zip_directory(folder_path: Path, zip_file: zipfile.ZipFile, root_dir: str):
    for root, dirs, files in os.walk(folder_path):
        # Skip __pycache__ inside models/
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        for file in files:
            file_path = os.path.join(root, file)
            # Add file to zip archive, respecting the relative tree structure
            arcname = os.path.relpath(file_path, start=root_dir)
            zip_file.write(file_path, arcname)

def build_kit():
    print("==================================================")
    print("   APEXFOREX SAAS - BUILDING VM MIGRATION KIT")
    print("==================================================")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"apex_migration_kit_{timestamp}.zip"
    
    files_to_zip = [
        "signals.db",
        ".env",
        "config/trading_whitelist.json"
    ]
    
    dirs_to_zip = [
        "models"
    ]
    
    project_root = Path(__file__).parent.parent
    
    with zipfile.ZipFile(project_root / zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print("Packaging stateful files...")
        for item in files_to_zip:
            target = project_root / item
            if target.exists():
                print(f"  [+] Adding {item}")
                zipf.write(target, item)
            else:
                print(f"  [!] Missing {item}. Skipping.")
                
        print("\nPackaging trained AI models...")
        for d in dirs_to_zip:
            dir_target = project_root / d
            if dir_target.exists():
                print(f"  [+] Adding directory {d}/")
                zip_directory(dir_target, zipf, str(project_root))
            else:
                print(f"  [!] Missing directory {d}/. Skipping.")
                
    print(f"\n✅ Migration Kit Built Successfully: {zip_name}")
    print("Transfer this file to your Windows VPS and extract it over the main directory.")

if __name__ == "__main__":
    build_kit()
