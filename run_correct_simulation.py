import sys, os
import sqlite3
import pandas as pd
import numpy as np
import zoneinfo
from datetime import datetime, timezone

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# Connect to database
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
df['t_ny'] = df['t_utc'].dt.tz_convert(NY_TZ)
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
print(f"Total Qualified Signals in DB (>=61% conf, Forex only, allowed directions): {len(df_qualified)}")

# Group by month
df_qualified['month'] = df_qualified['t_utc'].dt.to_period('M')
print("\n=== QUALIFIED SIGNALS SUMMARY BY MONTH ===")
for m, grp in df_qualified.groupby('month'):
    wins = (grp['outcome'] == 'SUCCESS').sum()
    losses = (grp['outcome'] == 'FAIL').sum()
    total = len(grp)
    wr = wins / total * 100 if total > 0 else 0
    print(f"{m} | Total: {total:>4} | Wins: {wins:>3} | Losses: {losses:>3} | Win Rate: {wr:>5.1f}%")

# Let's run a portfolio simulation that tracks open positions over time
# Scenario A: UNCAPPED (as configured, max_open_trades=0, no currency basket cap)
# Scenario B: BASKET CAP (Max 2 trades per currency leg)

def simulate_portfolio(signals_df, basket_cap=None, max_concurrent=None, 
                       slippage_loss_pct=1.05, win_payout=72.0, loss_cost=-52.0):
    """
    Simulates portfolio equity over time tracking actual concurrency.
    win_payout: Realistic win accounting for spread/commission ($72 instead of $75)
    loss_cost: Realistic loss accounting for spread/slippage/commission (-$52 instead of -$50)
    """
    signals = signals_df.copy().reset_index(drop=True)
    
    # Track open positions: list of dicts {symbol, signal, currencies, exit_time, pnl}
    active_positions = []
    closed_trades = []
    
    # Account metrics
    initial_balance = 10000.0
    equity = initial_balance
    balance = initial_balance
    peak_equity = initial_balance
    max_total_dd = 0.0
    
    daily_start_balance = initial_balance
    current_day = None
    max_daily_dd = 0.0
    daily_breaches = 0
    total_breaches = 0
    circuit_breaker_trips = 0
    
    daily_pnl = 0.0
    
    for idx, row in signals.iterrows():
        t_entry = row['t_utc']
        t_exit = row['t_exit_utc'] if pd.notnull(row['t_exit_utc']) else t_entry + pd.Timedelta(hours=4)
        sym = row['symbol']
        sig = row['signal']
        outcome = row['outcome']
        
        # Check day rollover (00:00 UTC / 22:00 UTC CEST)
        trade_day = t_entry.date()
        if current_day != trade_day:
            current_day = trade_day
            daily_start_balance = balance
            daily_pnl = 0.0
        
        # Close out any positions that ended before this entry
        still_active = []
        for pos in active_positions:
            if pos['exit_time'] <= t_entry:
                # Trade closed
                balance += pos['pnl']
                daily_pnl += pos['pnl']
                closed_trades.append(pos)
            else:
                still_active.append(pos)
        active_positions = still_active
        
        # Calculate currencies involved in symbol (e.g. GBPJPY -> GBP, JPY)
        base = sym[:3]
        quote = sym[3:6] if len(sym) >= 6 else sym[3:]
        
        # Check max concurrent positions
        if max_concurrent and len(active_positions) >= max_concurrent:
            continue
            
        # Check currency basket cap
        if basket_cap:
            curr_count = 0
            for pos in active_positions:
                if base in pos['currencies'] or quote in pos['currencies']:
                    curr_count += 1
            if curr_count >= basket_cap:
                # Blocked by basket cap
                continue
                
        # Check circuit breaker (-$450 daily loss)
        if daily_pnl <= -450.0:
            circuit_breaker_trips += 1
            continue
            
        # Assign realistic PnL
        pnl = win_payout if outcome == 'SUCCESS' else loss_cost
        
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
        
        # Check intraday drawdown
        total_dd = initial_balance - balance
        if total_dd > max_total_dd:
            max_total_dd = total_dd
        if total_dd >= 1000.0:
            total_breaches += 1
            
        cur_day_dd = daily_start_balance - (balance + sum(p['pnl'] for p in active_positions if p['pnl'] < 0))
        if cur_day_dd > max_daily_dd:
            max_daily_dd = cur_day_dd
        if cur_day_dd >= 500.0:
            daily_breaches += 1

    # Close remaining
    for pos in active_positions:
        balance += pos['pnl']
        closed_trades.append(pos)
        
    df_res = pd.DataFrame(closed_trades)
    total_trades = len(df_res)
    wins = len(df_res[df_res['outcome'] == 'SUCCESS']) if total_trades > 0 else 0
    losses = len(df_res[df_res['outcome'] == 'FAIL']) if total_trades > 0 else 0
    net_pnl = balance - initial_balance
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pf = abs(df_res[df_res['pnl'] > 0]['pnl'].sum() / df_res[df_res['pnl'] < 0]['pnl'].sum()) if losses > 0 else 0
    
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'max_total_dd': max_total_dd,
        'max_daily_dd': max_daily_dd,
        'daily_breaches': daily_breaches,
        'total_breaches': total_breaches,
        'circuit_breaker_trips': circuit_breaker_trips
    }

print("\n" + "="*70)
print("=== SIMULATION RESULTS (REALISTIC FRICTIONS & CONCURRENCY) ===")
print("="*70)

# 1. UNCAPPED (Current Production Configuration)
res_uncapped = simulate_portfolio(df_qualified, basket_cap=None, max_concurrent=None)
print("\n[SCENARIO 1: UNCAPPED (CURRENT PRODUCTION SETUP - No Basket Cap, Uncapped)]")
print(f"Total Executed Trades: {res_uncapped['total_trades']}")
print(f"Wins: {res_uncapped['wins']} | Losses: {res_uncapped['losses']} | Win Rate: {res_uncapped['win_rate']:.1f}%")
print(f"Net Profit/Loss: ${res_uncapped['net_pnl']:,.2f}")
print(f"Profit Factor: {res_uncapped['profit_factor']:.2f}")
print(f"Worst Intraday Daily Drop: -${res_uncapped['max_daily_dd']:,.2f}")
print(f"FTMO Daily Limit ($500) Breaches: {res_uncapped['daily_breaches']}")
print(f"Worst Total Account Drawdown: -${res_uncapped['max_total_dd']:,.2f}")
print(f"FTMO Total Limit ($1,000) Breaches: {res_uncapped['total_breaches']}")

# 2. BASKET CAP = 2 (Max 2 positions per currency)
res_cap2 = simulate_portfolio(df_qualified, basket_cap=2, max_concurrent=5)
print("\n[SCENARIO 2: BASKET CAP = 2 (Max 2 Positions Per Currency, Max 5 Total)]")
print(f"Total Executed Trades: {res_cap2['total_trades']}")
print(f"Wins: {res_cap2['wins']} | Losses: {res_cap2['losses']} | Win Rate: {res_cap2['win_rate']:.1f}%")
print(f"Net Profit/Loss: ${res_cap2['net_pnl']:,.2f}")
print(f"Profit Factor: {res_cap2['profit_factor']:.2f}")
print(f"Worst Intraday Daily Drop: -${res_cap2['max_daily_dd']:,.2f}")
print(f"FTMO Daily Limit ($500) Breaches: {res_cap2['daily_breaches']}")
print(f"Worst Total Account Drawdown: -${res_cap2['max_total_dd']:,.2f}")
print(f"FTMO Total Limit ($1,000) Breaches: {res_cap2['total_breaches']}")

# 3. BASKET CAP = 1 (Strict 1 Position Per Currency, Max 3 Total)
res_cap1 = simulate_portfolio(df_qualified, basket_cap=1, max_concurrent=3)
print("\n[SCENARIO 3: STRICT DIVERSIFICATION (Max 1 Per Currency, Max 3 Total)]")
print(f"Total Executed Trades: {res_cap1['total_trades']}")
print(f"Wins: {res_cap1['wins']} | Losses: {res_cap1['losses']} | Win Rate: {res_cap1['win_rate']:.1f}%")
print(f"Net Profit/Loss: ${res_cap1['net_pnl']:,.2f}")
print(f"Profit Factor: {res_cap1['profit_factor']:.2f}")
print(f"Worst Intraday Daily Drop: -${res_cap1['max_daily_dd']:,.2f}")
print(f"FTMO Daily Limit ($500) Breaches: {res_cap1['daily_breaches']}")
print(f"Worst Total Account Drawdown: -${res_cap1['max_total_dd']:,.2f}")
print(f"FTMO Total Limit ($1,000) Breaches: {res_cap1['total_breaches']}")
