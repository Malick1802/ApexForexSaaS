import sqlite3
import json
from pathlib import Path

def report():
    db_path = Path("signals.db")
    wl_path = Path("config/trading_whitelist.json")
    
    # Load whitelist
    matrix = {}
    if wl_path.exists():
        data = json.load(open(wl_path))
        matrix = data.get("performance_matrix", {})

    # Load active signals from DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, signal, outcome, confidence, timestamp
        FROM signals
        WHERE outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')
        ORDER BY timestamp DESC
    """)
    active = {row['symbol']: dict(row) for row in cursor.fetchall()}
    conn.close()

    tiers = ["60", "70", "80", "90", "100"]
    print(f"{'Pair':<12} {'Active':^12} {'T60':^14} {'T70':^14} {'T80':^14} {'T90':^14} {'T100':^14}")
    print("-" * 98)

    for symbol in sorted(matrix.keys()):
        active_str = "-"
        if symbol in active:
            t = active[symbol]
            active_str = f"{t['signal']} {t['confidence']*100:.0f}%"

        row = f"{symbol:<12} {active_str:^12}"
        for t in tiers:
            td = matrix[symbol].get(t, {})
            acc = td.get("accuracy", 0.0)
            trades = td.get("trades", 0)
            status = td.get("status", "N/A")
            icon = "✅" if status == "APPROVED" else ("⏳" if trades > 0 else "⬜")
            row += f" {icon}{acc*100:>3.0f}% {trades}tr  "
        print(row)

    print()
    print("Active Trades in DB:")
    for sym, t in active.items():
        print(f"  {sym}: {t['signal']} @ {t['confidence']*100:.1f}% — since {t['timestamp'][:16]}")

if __name__ == "__main__":
    report()
