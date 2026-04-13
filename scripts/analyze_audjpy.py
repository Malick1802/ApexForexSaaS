
import sqlite3
import pandas as pd
import yfinance as yf

def analyze_audjpy():
    print("--- 1. Inspecting Database for AUDJPY ---")
    try:
        conn = sqlite3.connect("signals.db")
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute("PRAGMA table_info(signals)")
        cols = [info[1] for info in cursor.fetchall()]
        
        # Query AUDJPY
        cursor.execute("SELECT * FROM signals WHERE symbol='AUDJPY' ORDER BY timestamp DESC LIMIT 3")
        rows = cursor.fetchall()
        
        last_signal = None
        
        if not rows:
            print("No records found for AUDJPY.")
        else:
            for row in rows:
                data = dict(zip(cols, row))
                print("\n------------------------------------------------")
                print(f"ID: {data.get('id')} | Time: {data.get('timestamp')}")
                print(f"Signal: {data.get('signal')} | Conf: {data.get('confidence')}")
                print(f"Entry: {data.get('price_at_signal')} | Outcome: {data.get('outcome')}")
                print(f"TP: {data.get('tp_price')} | SL: {data.get('sl_price')}")
                if not last_signal and data.get('signal') in ['BUY', 'SELL']:
                    last_signal = data
            
        conn.close()
        
    except Exception as e:
        print(f"DB Error: {e}")
        return

    if not last_signal:
        print("\nNo recent BUY/SELL signal found to analyze.")
        return

    print("\n--- 2. Checking Price Action (AUDJPY=X) ---")
    try:
        # Fetch data
        ticker = yf.Ticker("AUDJPY=X")
        df = ticker.history(period="1d", interval="5m")
        
        if df.empty:
            print("No price data fetched.")
            return

        current_price = df['Close'].iloc[-1]
        max_high = df['High'].max()
        min_low = df['Low'].min()
        
        print(f"Data Points: {len(df)}")
        print(f"Last Price: {current_price:.5f}")
        print(f"Max High (24h): {max_high:.5f}")
        print(f"Min Low (24h): {min_low:.5f}")
        
        # Analysis
        signal_type = last_signal['signal']
        sl = last_signal['sl_price']
        tp = last_signal['tp_price']
        entry = last_signal['price_at_signal']
        
        print(f"\nAnalysis for {signal_type} @ {entry}:")
        print(f"SL: {sl} | TP: {tp}")
        
        if signal_type == 'SELL':
            if max_high >= sl:
                print(f"❌ STOP LOSS HIT! High {max_high:.3f} >= {sl:.3f}")
                print(f"Breach: {max_high - sl:.3f} pips")
            elif min_low <= tp:
                 print(f"🎯 TAKE PROFIT HIT! Low {min_low:.3f} <= {tp:.3f}")
            else:
                print("Trade status: Floating / Stagnant")
                
        elif signal_type == 'BUY':
            if min_low <= sl:
                print(f"❌ STOP LOSS HIT! Low {min_low:.3f} <= {sl:.3f}")
                print(f"Breach: {sl - min_low:.3f} pips")
            elif max_high >= tp:
                 print(f"🎯 TAKE PROFIT HIT! High {max_high:.3f} >= {tp:.3f}")
            else:
                print("Trade status: Floating / Stagnant")

    except Exception as e:
        print(f"Price Check Error: {e}")

if __name__ == "__main__":
    analyze_audjpy()
