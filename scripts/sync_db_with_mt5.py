import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core.mt5_connector import get_mt5

DB_PATH = os.path.join(os.getcwd(), 'signals.db')

def sync_db_with_mt5():
    # 1. Connect to MT5
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ Failed to connect to MT5 Bridge")
        return

    acc_info = _mt5.account_info()
    if acc_info:
        print(f"✅ Connected to MT5: {acc_info.login}")
    else:
        print("⚠️ Connected to bridge, but no MT5 account active.")
        return
    
    # 2. Get Open Positions
    positions = mt5.positions_get()
    open_symbols = set()
    open_tickets = {}
    
    if positions:
        print(f"found {len(positions)} open positions")
        for pos in positions:
            open_symbols.add(pos.symbol)
            # Store ticket info if needed
            open_tickets[pos.symbol] = pos.ticket
            print(f"  OPEN: {pos.symbol} {pos.type} Vol:{pos.volume} PnL:{pos.profit}")
    else:
        print("No open positions found.")

    # 3. Get History (Deals) - Last 7 Days
    from_date = datetime.now() - timedelta(days=7)
    history = _mt5.history_deals_get(from_date, datetime.now())
    closed_deals = {}
    
    if history:
        for deal in history:
            if deal.symbol:
                # We care about EXIT deals (Entry=1 is In, Entry=2 is Out)
                # Or just any closed deal.
                # Actually, filtering by Entry 1 (In) vs 2 (Out).
                # Simplified: If it's in history with profit != 0, it's likely closed.
                # We map symbol -> outcome
                outcome = 'SUCCESS' if deal.profit > 0 else 'FAIL'
                closed_deals[deal.symbol] = outcome
                # print(f"  HIST: {deal.symbol} {outcome} PnL:{deal.profit}")

    # 4. Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 5. RECONCILIATION LOGIC
    
    # A. Resurrect False Closed (DB=Closed, MT5=Open)
    # Get all signals from this week that are NOT Active
    start_of_week = datetime.now() - timedelta(days=datetime.now().weekday() + 1) # Go back a bit more to be safe
    
    query = """
    SELECT id, symbol, outcome, signal FROM signals 
    WHERE outcome != 'ACTIVE' 
    AND timestamp > ?
    AND signal IN ('BUY', 'SELL')
    """
    df_closed = pd.read_sql_query(query, conn, params=(start_of_week.isoformat(),))
    
    resurrected = 0
    for _, row in df_closed.iterrows():
        sym = row['symbol']
        if sym in open_symbols:
            # IT IS OPEN IN MT5!
            print(f"🔄 RESURRECTING {sym} (DB: {row['outcome']} -> MT5: OPEN)")
            # IMPORTANT: Mark only ONE active signal per symbol (latest).
            # But here we mark all matching syms. We need dedup later.
            cursor.execute("UPDATE signals SET outcome = 'ACTIVE' WHERE id = ?", (row['id'],))
            resurrected += 1
            
    # B. Close Stale Active (DB=Active, MT5=Closed)
    # Get all Active signals
    query_active = "SELECT id, symbol FROM signals WHERE outcome = 'ACTIVE'"
    df_active = pd.read_sql_query(query_active, conn)
    
    closed_updates = 0
    for _, row in df_active.iterrows():
        sym = row['symbol']
        # If NOT in open_symbols AND IS in closed_deals
        if sym not in open_symbols and sym in closed_deals:
            new_outcome = closed_deals[sym]
            print(f"📉 CLOSING {sym} (DB: ACTIVE -> MT5: {new_outcome})")
            cursor.execute("UPDATE signals SET outcome = ? WHERE id = ?", (new_outcome, row['id']))
            closed_updates += 1
            
    conn.commit()
    conn.close()
    
    print("-" * 30)
    print(f"Sync Complete.")
    print(f"Resurrected (False Closed): {resurrected}")
    print(f"Closed (False Active): {closed_updates}")

if __name__ == "__main__":
    sync_db_with_mt5()
