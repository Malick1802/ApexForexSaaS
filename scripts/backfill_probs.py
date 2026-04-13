import sqlite3
import json

def backfill_active_signal():
    print("--- Backfilling Active Signal Probabilities ---")
    try:
        conn = sqlite3.connect('signals.db')
        c = conn.cursor()
        
        # Find ALL active signals
        c.execute("SELECT id, symbol, signal, confidence FROM signals WHERE outcome='ACTIVE'")
        rows = c.fetchall()
        
        if not rows:
            print("No active signals to backfill.")
            return

        for row in rows:
            sig_id, symbol, signal, conf = row
            # print(f"Found Active Signal: ID={sig_id}, {symbol} {signal} ({conf})")
        
            # Mock probabilities (Foundation format: [wait, buy, sell])
            b, s, w = 0.0, 0.0, 0.0
            if signal == 'BUY':
                b = conf
                s = (1.0 - conf) / 2
                w = (1.0 - conf) / 2
            elif signal == 'SELL':
                s = conf
                b = (1.0 - conf) / 2
                w = (1.0 - conf) / 2
            elif signal == 'WAIT':
                w = conf
                b = (1.0 - conf) / 2
                s = (1.0 - conf) / 2
            
            raw_probs = [w, b, s] # wait, buy, sell
            raw_json = json.dumps(raw_probs)
            
            c.execute("""
                UPDATE signals 
                SET buy_prob=?, sell_prob=?, wait_prob=?, raw_probabilities=? 
                WHERE id=?
            """, (b, s, w, raw_json, sig_id))
        
        conn.commit()
        print("✅ Backfill Complete.")
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    backfill_active_signal()
