import sqlite3
import pandas as pd
from core.database import SignalDatabase
from data_pipeline import DataEngine

def debug_gold():
    print("=== GOLD Diagnostic ===")
    
    # Check Database
    try:
        db = SignalDatabase()
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM signals WHERE symbol='XAUUSD' ORDER BY timestamp DESC LIMIT 3").fetchall()
        print(f"\nLast 3 DB Signals for GOLD:")
        for r in rows:
            d = dict(r)
            print(f"  TS: {d['timestamp']} | Price: {d['price_at_signal']} | Signal: {d['signal']} | Regime: {d['regime']} | Trades: {d.get('model_trades', 'N/A')}")
    except Exception as e:
        print(f"DB Error: {e}")

    # Check Data Engine
    try:
        engine = DataEngine()
        df = engine.fetch("GOLD", interval="1h", days=3)
        if not df.empty:
            last_price = df['close'].iloc[-1]
            print(f"\nDataEngine 'XAUUSD' Current Price: {last_price}")
            print(f"Data Source: {engine.config['data_provider']['active']}")
        else:
            print("\nDataEngine returned empty DataFrame for GOLD")
    except Exception as e:
        print(f"DataEngine Error: {e}")

if __name__ == "__main__":
    debug_gold()
