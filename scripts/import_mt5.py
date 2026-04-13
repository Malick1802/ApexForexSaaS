import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core.mt5_connector import get_mt5

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def import_mt5_positions():
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ MT5 Bridge link failed")
        return

    positions = mt5.positions_get()
    if not positions:
        print("No open positions in MT5.")
        return

    print(f"Found {len(positions)} open positions. Checking DB...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get existing ACTIVE BUY/SELL signals to avoid duplicates
    existing = pd.read_sql_query(
        "SELECT symbol FROM signals WHERE outcome='ACTIVE' AND signal IN ('BUY', 'SELL')", 
        conn
    )
    existing_syms = set(existing['symbol'].tolist())
    
    imported_count = 0
    
    for pos in positions:
        sym = pos.symbol
        if sym in existing_syms:
            print(f"  {sym}: Already active in DB. Skipping.")
            continue
            
        print(f"  {sym}: Missing in DB. Importing...")
        
        # Map MT5 type to Signal
        # 0 = Buy, 1 = Sell
        sig_type = 'BUY' if pos.type == 0 else 'SELL'
        
        # Create new record
        # Use position open time if possible, else now
        # pos.time is unix timestamp
        open_time = datetime.fromtimestamp(pos.time).isoformat()
        
        # Insert
        # We need to match the schema.
        # columns: timestamp, symbol, signal, confidence, price_at_signal, sl, tp, outcome, strategy, timeframe
        # We'll use defaults for unknown
        
        # Check which columns exist in DB first?
        # We'll assume the standard set.
        
        # Schema: timestamp, symbol, signal, confidence, price_at_signal, outcome
        # Removing 'strategy' as it doesn't exist in DB schema
        cols = "timestamp, symbol, signal, confidence, price_at_signal, outcome"
        vals = (open_time, sym, sig_type, 0.99, pos.price_open, 'ACTIVE')
        
        try:
            cursor.execute(f"INSERT INTO signals ({cols}) VALUES (?, ?, ?, ?, ?, ?)", vals)
            imported_count += 1
            print(f"    -> Imported {sym} {sig_type} @ {pos.price_open}")
        except Exception as e:
            print(f"    -> Error importing {sym}: {e}")

    conn.commit()
    conn.close()
    
    print("-" * 30)
    print(f"Import Complete. Imported {imported_count} positions.")

if __name__ == "__main__":
    import_mt5_positions()
