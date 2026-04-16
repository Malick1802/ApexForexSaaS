import sqlite3
import pandas as pd
from core.inference import InferenceEngine

def check_nzjpy_history():
    engine = InferenceEngine()
    # Fetch 3 days of data to cover April 14-16
    df = engine.data_engine.fetch("NZDJPY", interval="1h", days=3, use_cache=False)
    if df.empty:
        print("Failed to fetch NZDJPY history.")
        return
        
    print(f"--- NZDJPY Range (Last 3 Days) ---")
    print(f"Max High: {df['high'].max()}")
    print(f"Min Low: {df['low'].min()}")
    print(f"Current Close: {df['close'].iloc[-1]}")
    
    # Entry: 93.705, TP: 94.100, SL: 93.435
    print("\nTarget Check for April 14 Signal:")
    if df['high'].max() >= 94.100:
        print("✅ Should have hit TP (94.100)")
    if df['low'].min() <= 93.435:
        print("✅ Should have hit SL (93.435)")
    if df['high'].max() < 94.100 and df['low'].min() > 93.435:
        print("⏳ Still in the range. Trade is validly ACTIVE.")

if __name__ == "__main__":
    check_nzjpy_history()
