import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

db_path = "c:/Users/artem/Downloads/ApexForexSaaS/signals.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT symbol, signal, outcome, confidence, timestamp FROM signals WHERE outcome IN ('SUCCESS', 'FAIL');", conn)

if len(df) > 0:
    stats = df.groupby('symbol')['outcome'].value_counts().unstack(fill_value=0)
    if 'SUCCESS' not in stats.columns: stats['SUCCESS'] = 0
    if 'FAIL' not in stats.columns: stats['FAIL'] = 0
    stats['total'] = stats['SUCCESS'] + stats['FAIL']
    stats['win_rate'] = (stats['SUCCESS'] / stats['total'] * 100).round(1)
    stats = stats.sort_values('total', ascending=False)
    print("=== SYMBOL WIN RATES ===")
    print(stats)
    
    total_wins = stats['SUCCESS'].sum()
    total_losses = stats['FAIL'].sum()
    total_trades = total_wins + total_losses
    print(f"\nOVERALL: {total_wins} Wins, {total_losses} Losses ({total_wins/total_trades*100:.1f}% Win Rate)")
else:
    print("No resolved signals found in database.")
conn.close()
