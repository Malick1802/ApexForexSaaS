
import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def audit_signals():
    print(f"Connecting to DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM signals WHERE outcome='ACTIVE'"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No active signals to audit.")
            return

        print(f"Auditing {len(df)} Active Signals...")
        print("-" * 60)
        
        for _, row in df.iterrows():
            verdict = "PASS"
            reasons = []
            
            # 1. Confidence Check (Default 90%)
            conf = row.get('confidence', 0)
            if conf < 0.90:
                reasons.append(f"Low Confidence ({conf:.4f} < 0.90)")
                # verdict = "FAIL" # Strict?
            elif conf < 0.80:
                 verdict = "FAIL"
                 reasons.append(f"Very Low Confidence ({conf:.4f} < 0.80)")

            # 2. Time Check (e.g. > 24h)
            ts_str = row['timestamp']
            try:
                ts = datetime.fromisoformat(ts_str)
                age = (datetime.now() - ts).total_seconds() / 3600
                if age > 24:
                    reasons.append(f"Old Signal ({age:.1f}h > 24h)")
                    verdict = "FAIL"
                elif age > 12:
                    reasons.append(f"Stale Signal ({age:.1f}h > 12h)")
            except:
                reasons.append("Invalid Timestamp")

            # 3. Strategy Check
            strat = row.get('strategy', 'Unknown')
            if strat == 'Unknown' or strat == 'TEST':
                reasons.append(f"Invalid Strategy ({strat})")
                verdict = "FAIL"

            print(f"Signal ID {row['id']} ({row['symbol']} {row['signal']}): {verdict}")
            if reasons:
                print(f"  Issues: {', '.join(reasons)}")
                
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audit_signals()
