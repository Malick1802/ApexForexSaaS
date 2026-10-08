import sys, os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Connect directly to database
conn = sqlite3.connect('signals.db')
df = pd.read_sql_query('''
    SELECT id, timestamp, exit_time, duration_seconds, symbol, signal, confidence,
           is_hidden, outcome, exit_reason, price_at_signal, exit_price, mt5_ticket, model_version
    FROM signals
    WHERE signal IN ('BUY', 'SELL')
      AND (model_version = 'v1' OR model_version IS NULL)
      AND outcome IN ('SUCCESS', 'FAIL')
    ORDER BY timestamp ASC
''', conn)
conn.close()

df['t_utc'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
df['t_exit_utc'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True)
df['conf'] = df['confidence'].astype(float)
df['hour_utc'] = df['t_utc'].dt.hour
df['day_name_utc'] = df['t_utc'].dt.day_name()

from core.symbol_guard import is_commodity, is_direction_blocked, is_symbol_blocked

def is_qualified(r):
    sym = str(r['symbol']).strip()
    sig = str(r['signal']).strip()
    conf = float(r['conf'])
    if is_commodity(sym) or is_symbol_blocked(sym): return False
    if is_direction_blocked(sym, sig): return False
    if r['day_name_utc'] == 'Friday' and r['hour_utc'] >= 14: return False
    return conf >= 0.61

df_qualified = df[df.apply(is_qualified, axis=1)].sort_values('t_utc').reset_index(drop=True)

def simulate_from_balance(signals_df, start_balance=9157.25, target_balance=10000.0,
                           basket_cap=1, max_concurrent=3, risk_pct=0.005):
    signals = signals_df.copy().reset_index(drop=True)
    active_positions = []
    closed_trades = []
    balance = start_balance
    
    for idx, row in signals.iterrows():
        t_entry = row['t_utc']
        t_exit = row['t_exit_utc'] if pd.notnull(row['t_exit_utc']) else t_entry + pd.Timedelta(hours=4)
        sym = row['symbol']
        sig = row['signal']
        outcome = row['outcome']
        
        # Close positions that finished before this entry
        still_active = []
        for pos in active_positions:
            if pos['exit_time'] <= t_entry:
                balance += pos['pnl']
                pos['bal_after'] = balance
                closed_trades.append(pos)
            else:
                still_active.append(pos)
        active_positions = still_active
        
        # Check target reached
        if balance >= target_balance:
            break
            
        base = sym[:3]
        quote = sym[3:6] if len(sym) >= 6 else sym[3:]
        
        if max_concurrent and len(active_positions) >= max_concurrent:
            continue
            
        if basket_cap:
            curr_count = 0
            for pos in active_positions:
                if base in pos['currencies'] or quote in pos['currencies']:
                    curr_count += 1
            if curr_count >= basket_cap:
                continue
                
        # DYNAMIC 0.5% of current running balance:
        trade_risk = balance * risk_pct
        pnl = trade_risk * 1.44 if outcome == 'SUCCESS' else -trade_risk * 1.04
        
        active_positions.append({
            'id': row['id'],
            'symbol': sym,
            'signal': sig,
            'currencies': [base, quote],
            'entry_time': t_entry,
            'exit_time': t_exit,
            'outcome': outcome,
            'pnl': pnl
        })
        
    for pos in active_positions:
        balance += pos['pnl']
        pos['bal_after'] = balance
        closed_trades.append(pos)
        
    return pd.DataFrame(closed_trades), balance

# Run full baseline from $9,157.25
df_trades, final_b = simulate_from_balance(df_qualified, start_balance=9157.25, target_balance=999999)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['outcome'] == 'SUCCESS'])
losses = len(df_trades[df_trades['outcome'] == 'FAIL'])
all_time_wr = wins / total_trades if total_trades > 0 else 0

t_min = pd.to_datetime(df_trades['entry_time']).min()
t_max = pd.to_datetime(df_trades['entry_time']).max()
total_days = (t_max - t_min).days
total_weeks = total_days / 7.0
trades_per_week = total_trades / total_weeks
trades_per_day = trades_per_week / 5.0

print(f"=== RECOVERY PACE METRICS ===")
print(f"Qualified Trades Analyzed: {total_trades} over {total_weeks:.1f} weeks ({total_days} days)")
print(f"Average Execution Pace:   {trades_per_week:.1f} trades/week (~{trades_per_day:.1f} trades/trading day)")
print(f"All-Time Strategy Win Rate: {all_time_wr*100:.1f}% ({wins}W / {losses}L)")

# Deficit
deficit = 10000.0 - 9157.25
init_risk = 9157.25 * 0.005 # $45.79

# Rolling Recovery Windows:
# Test starting at every index in df_qualified with $9,157.25 and measure how many trades & days to reach $10,000!
windows = []
for i in range(len(df_qualified) - 25):
    sub = df_qualified.iloc[i:].reset_index(drop=True)
    res_trades, end_b = simulate_from_balance(sub, start_balance=9157.25, target_balance=10000.0)
    
    # Check if/when $10,000 was hit
    for idx, r in res_trades.iterrows():
        if r['bal_after'] >= 10000.0:
            t_s = pd.to_datetime(res_trades.iloc[0]['entry_time'])
            t_e = pd.to_datetime(r['exit_time'])
            cal_days = max(1, (t_e - t_s).days)
            windows.append({
                'start_date': t_s.strftime('%Y-%m-%d'),
                'trades': idx + 1,
                'cal_days': cal_days,
                'weeks': cal_days / 7.0,
                'trading_days': (idx + 1) / trades_per_day
            })
            break

df_w = pd.DataFrame(windows)
print(f"\n=== EMPIRICAL HISTORICAL RECOVERY WINDOWS ({len(df_w)} samples) ===")
if not df_w.empty:
    print(f"Fastest Recovery:    {df_w['trades'].min():>2} trades | {df_w['weeks'].min():>4.1f} weeks (~{df_w['cal_days'].min():>2} calendar days)")
    print(f"25th Percentile:     {df_w['trades'].quantile(0.25):>2.0f} trades | {df_w['weeks'].quantile(0.25):>4.1f} weeks (~{df_w['cal_days'].quantile(0.25):>2.0f} calendar days)")
    print(f"Median Recovery:     {df_w['trades'].median():>2.0f} trades | {df_w['weeks'].median():>4.1f} weeks (~{df_w['cal_days'].median():>2.0f} calendar days)")
    print(f"75th Percentile:     {df_w['trades'].quantile(0.75):>2.0f} trades | {df_w['weeks'].quantile(0.75):>4.1f} weeks (~{df_w['cal_days'].quantile(0.75):>2.0f} calendar days)")
    print(f"90th Percentile:     {df_w['trades'].quantile(0.90):>2.0f} trades | {df_w['weeks'].quantile(0.90):>4.1f} weeks (~{df_w['cal_days'].quantile(0.90):>2.0f} calendar days)")
    print(f"Slowest Window:      {df_w['trades'].max():>2} trades | {df_w['weeks'].max():>4.1f} weeks (~{df_w['cal_days'].max():>2} calendar days)")

# Regime Expected Values
# 1. Normal Trend Environment (63% Win Rate - May to August)
ev_norm = (0.63 * 1.44 - 0.37 * 1.04) * init_risk # +$23.91 / trade
trades_norm = deficit / ev_norm
weeks_norm = trades_norm / trades_per_week

# 2. All-Time Blended Environment (59.3% Win Rate)
ev_blend = (0.593 * 1.44 - 0.407 * 1.04) * init_risk # +$19.71 / trade
trades_blend = deficit / ev_blend
weeks_blend = trades_blend / trades_per_week

# 3. Challenging Environment (52% Win Rate)
ev_chop = (0.52 * 1.44 - 0.48 * 1.04) * init_risk # +$11.43 / trade
trades_chop = deficit / ev_chop
weeks_chop = trades_chop / trades_per_week

print("\n=== REGIME-BASED EXPECTATION BREAKDOWN ===")
print(f"1. Favorable Trend Regime (63.0% Win Rate):")
print(f"   • Expected Value: +${ev_norm:.2f} per trade")
print(f"   • Trades to Breakeven: ~{trades_norm:.0f} trades")
print(f"   • Projected Timeline:  ~{weeks_norm:.1f} weeks (~{weeks_norm*5:.0f} trading days)")

print(f"\n2. All-Time Blended Regime (59.3% Win Rate):")
print(f"   • Expected Value: +${ev_blend:.2f} per trade")
print(f"   • Trades to Breakeven: ~{trades_blend:.0f} trades")
print(f"   • Projected Timeline:  ~{weeks_blend:.1f} weeks (~{weeks_blend*5:.0f} trading days)")

print(f"\n3. Choppy / Lower-Edge Regime (52.0% Win Rate):")
print(f"   • Expected Value: +${ev_chop:.2f} per trade")
print(f"   • Trades to Breakeven: ~{trades_chop:.0f} trades")
print(f"   • Projected Timeline:  ~{weeks_chop:.1f} weeks (~{weeks_chop*5:.0f} trading days)")
