#!/usr/bin/env python
"""
Clear Database and Cache - Fresh Start Script
"""

import os
import sqlite3
import shutil
from pathlib import Path

def clear_database():
    """Clear all signals from the database."""
    db_path = "signals.db"
    
    if not os.path.exists(db_path):
        print(f"✓ Database not found at {db_path} (nothing to clear)")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM signals")
        count = cursor.fetchone()[0]
        
        # Clear all signals
        cursor.execute("DELETE FROM signals")
        conn.commit()
        
        # Reset auto-increment
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='signals'")
        conn.commit()
        
        conn.close()
        
        print(f"✓ Cleared {count} signals from database")
        
    except Exception as e:
        print(f"✗ Error clearing database: {e}")

def clear_cache():
    """Clear the data cache directory."""
    cache_dir = "data_cache"
    
    if not os.path.exists(cache_dir):
        print(f"✓ Cache directory not found (nothing to clear)")
        return
    
    try:
        # Count files
        files = list(Path(cache_dir).glob("*"))
        count = len(files)
        
        # Remove all cached files
        for file in files:
            if file.is_file():
                file.unlink()
        
        print(f"✓ Cleared {count} cached files from {cache_dir}/")
        
    except Exception as e:
        print(f"✗ Error clearing cache: {e}")

def main():
    print("=" * 60)
    print("CLEARING DATABASE AND CACHE - FRESH START")
    print("=" * 60)
    print()
    
    clear_database()
    clear_cache()
    
    print()
    print("=" * 60)
    print("✓ CLEANUP COMPLETE - Ready for fresh start!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Dashboard will start with empty signal history")
    print("  2. Fresh data will be fetched from TwelveData")
    print("  3. AI models will generate new signals")

if __name__ == "__main__":
    main()
