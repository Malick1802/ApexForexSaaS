import sqlite3
conn = sqlite3.connect('signals.db')
conn.execute("DELETE FROM signals WHERE symbol='EURAUD' AND outcome='ACTIVE'")
conn.execute("DELETE FROM signals WHERE symbol='GOLD' AND outcome='ACTIVE'")
conn.commit()
print("Cleaned up EURAUD and GOLD active signals for testing.")
