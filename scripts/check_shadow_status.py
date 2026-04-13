import sqlite3
import json

def check_status():
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check EURAUD (Shadow trades)
    print("--- EURAUD Shadow Signals ---")
    cursor.execute("SELECT id, symbol, signal, confidence, outcome, is_hidden, is_proven, timestamp FROM signals WHERE symbol = 'EURAUD' ORDER BY timestamp DESC LIMIT 5")
    [print(dict(row)) for row in cursor.fetchall()]
    
    # Check GOLD (Live Signals)
    print("\n--- GOLD Recent Signals ---")
    cursor.execute("SELECT id, symbol, signal, confidence, outcome, is_hidden, is_proven, timestamp FROM signals WHERE symbol = 'GOLD' ORDER BY timestamp DESC LIMIT 5")
    [print(dict(row)) for row in cursor.fetchall()]

if __name__ == '__main__':
    check_status()
