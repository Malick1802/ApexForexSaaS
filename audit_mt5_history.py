import MetaTrader5 as mt5
from datetime import datetime, timezone
import pandas as pd

p = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if mt5.initialize(path=p):
    from_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    to_date = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    deals = mt5.history_deals_get(from_date, to_date)
    mt5.shutdown()
    if deals:
        df = pd.DataFrame([d._asdict() for d in deals])
        df['time_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
        exits = df[df['entry'] == 1].copy()
        print(f"Total Closed Deals: {len(exits)}")
        print(f"Total Net Realized PnL: ${exits['profit'].sum():.2f}")
        wins = exits[exits['profit'] > 0]
        losses = exits[exits['profit'] < 0]
        print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(exits)*100:.1f}%")
        print("-" * 80)
        for _, r in exits.iterrows():
            t_str = str(r['time_utc'])[:19]
            print(f"{t_str} | {r['symbol']:<7} | Vol: {r['volume']:<4} | Profit: ${r['profit']:>7.2f} | {r['comment']}")
