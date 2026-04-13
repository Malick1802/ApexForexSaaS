import sqlite3
import os

db_path = 'signals.db'
if not os.path.exists(db_path):
    print("Database not found.")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Resolve the 'Heatmap Lock' by expiring all ACTIVE signals that are WAIT
# This allows the dashboard to show fresh live inference instead of a locked historical WAIT.
cursor.execute("UPDATE signals SET outcome = 'EXPIRED' WHERE signal = 'WAIT' AND outcome = 'ACTIVE'")
count = cursor.rowcount
conn.commit()
conn.close()

print(f"Successfully expired {count} stuck WAIT signals. Dashboard sentiment heatmap unlocked.")
