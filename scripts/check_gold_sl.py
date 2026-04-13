import sqlite3
import pandas as pd
from data_pipeline.engine import DataEngine

conn = sqlite3.connect('signals.db')
df = pd.read_sql_query('SELECT id, timestamp, symbol, signal, price_at_signal, tp_price, sl_price, outcome FROM signals WHERE symbol="GOLD" AND outcome="ACTIVE"', conn)
print("=== Active GOLD signals ===")
if not df.empty:
    print(df.to_string())
    for _, row in df.iterrows():
        id_val = row['id']
        sl = float(row['sl_price'])
        tp = float(row['tp_price'])
        direction = row['signal']
        print(f"\nEvaluating GOLD signal ID {id_val}: Direction {direction}, SL: {sl}, TP: {tp}")

        engine = DataEngine()
        data = engine.fetch_data('GOLD', '1m', limit=120)
        if data is not None and not data.empty:
            recent_low = data['low'].min()
            recent_high = data['high'].max()
            current = data['close'].iloc[-1]
            print(f"Data since 2 hours ago: Low={recent_low}, High={recent_high}, Current={current}")
            
            if direction == 'BUY':
                if recent_low <= sl:
                    print("--> YES, Low has breached SL!")
                elif recent_high >= tp:
                    print("--> YES, High has breached TP!")
                else:
                    print("--> NO, neither TP nor SL breached in last 2 hours.")
            else:
                if recent_high >= sl:
                    print("--> YES, High has breached SL!")
                elif recent_low <= tp:
                    print("--> YES, Low has breached TP!")
                else:
                    print("--> NO, neither TP nor SL breached in last 2 hours.")
else:
    print("NO ACTIVE GOLD SIGNALS.")
conn.close()
