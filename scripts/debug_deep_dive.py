
import yfinance as yf
import pandas as pd
import sqlite3

def check_yfinance_periods():
    print("--- Checking YFinance Periods ---")
    ticker = yf.Ticker("USDCHF=X")
    
    print("\n[Fetch 5d]")
    df5 = ticker.history(period="5d", interval="1h")
    print(f"Tail Date: {df5.index[-1]}")
    
    print("\n[Fetch 60d]")
    df60 = ticker.history(period="60d", interval="1h")
    print(f"Tail Date: {df60.index[-1]}")

def audit_db_signals():
    print("\n--- Auditing USDCHF Signals ---")
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT id, timestamp, signal, outcome, buy_prob, status FROM signals WHERE symbol='USDCHF' ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    
    for row in rows:
        d = dict(row)
        print(f"ID: {d['id']}, Sig: {d['signal']}, Out: {d['outcome']}, Status: {d['status']}, BuyProb: {d['buy_prob']}")
    
    conn.close()

if __name__ == "__main__":
    check_yfinance_periods()
    audit_db_signals()
