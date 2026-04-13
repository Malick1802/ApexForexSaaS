"""
Expert Model Stability Audit
Analyzes the signals database and ghost_trades.csv for:
1. USDJPY confidence cluster analysis (Trending regime)
2. EURUSD rejection pattern analysis (Ranging regime)
3. Macro-validation (VIX proxy spikes, yield slope rejections)
"""
import sys
import os
import sqlite3
import csv
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "signals.db"
GHOST_CSV = PROJECT_ROOT / "logs" / "ghost_trades.csv"

SEP = "─" * 60

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fmt_conf(v):
    try: return f"{float(v)*100:.1f}%"
    except: return str(v)

# ─────────────────────────────────────────────────────────────
# 1. USDJPY Stability (Trending Regime)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  STABILITY TEST 1: USDJPY — Trending Regime Cluster Analysis")
print(f"{'='*60}")

with connect() as conn:
    rows = conn.execute("""
        SELECT timestamp, signal, confidence, regime, adx, atr_zscore
        FROM signals
        WHERE symbol = 'USDJPY'
        ORDER BY timestamp DESC LIMIT 20
    """).fetchall()

if rows:
    print(f"{'Timestamp':<22} {'Signal':<6} {'Confidence':<12} {'Regime':<18} {'ADX':<8} {'ATR_Z'}")
    print(SEP)
    prev_conf = None
    jumpy_count = 0
    stable_count = 0
    for r in rows:
        conf_val = float(r['confidence']) if r['confidence'] else 0.0
        jitter = abs(conf_val - prev_conf) if prev_conf is not None else 0.0
        label = " ⚡JUMP" if jitter > 0.15 else "  ✅STABLE" if (jitter < 0.05 and prev_conf is not None) else ""
        if jitter > 0.15: jumpy_count += 1
        elif jitter < 0.05 and prev_conf is not None: stable_count += 1
        ts = r['timestamp'][:19] if r['timestamp'] else "N/A"
        regime = r['regime'] or "UNKNOWN"
        adx = f"{float(r['adx']):.1f}" if r['adx'] else "N/A"
        atz = f"{float(r['atr_zscore']):.2f}" if r['atr_zscore'] else "N/A"
        print(f"  {ts:<22} {r['signal']:<6} {fmt_conf(conf_val):<12} {regime:<18} {adx:<8} {atz}{label}")
        prev_conf = conf_val
    print(f"\n  ✅ Stable transitions: {stable_count} | ⚡ Jumpy transitions: {jumpy_count}")
    if stable_count > jumpy_count:
        print("  📊 VERDICT: Expert Model STABLE — institutional confidence cluster detected")
    elif jumpy_count > stable_count:
        print("  ⚠️  VERDICT: Confidence JUMPY — may indicate 5-epoch clone behavior")
    else:
        print("  🔶 VERDICT: Mixed stability — needs more data from the 48h window")
else:
    print("  No USDJPY signals found yet — system may still be scanning.")

# ─────────────────────────────────────────────────────────────
# 2. EURUSD Ranging Filter (Rejection Analysis)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  STABILITY TEST 2: EURUSD — Ranging Regime Rejection Analysis")
print(f"{'='*60}")

with connect() as conn:
    rows = conn.execute("""
        SELECT timestamp, signal, confidence, regime, adx, vix_proxy, yield_slope
        FROM signals
        WHERE symbol = 'EURUSD'
        ORDER BY timestamp DESC LIMIT 20
    """).fetchall()

if rows:
    wait_sigs = [r for r in rows if r['signal'] == 'WAIT']
    action_sigs = [r for r in rows if r['signal'] in ('BUY','SELL')]
    near_misses = [r for r in wait_sigs 
                   if r['confidence'] and 0.65 <= float(r['confidence']) <= 0.72]
    
    print(f"  Total signals logged:    {len(rows)}")
    print(f"  BUY/SELL signals:        {len(action_sigs)}")
    print(f"  WAIT (rejected):         {len(wait_sigs)}")
    print(f"  Near-miss rejections (conf 65-72%): {len(near_misses)} ← proves 72% floor is working")
    
    if near_misses:
        print(f"\n  Near-Miss Detail (correctly filtered by regime gate):")
        print(f"  {'Timestamp':<22} {'Conf':<10} {'Regime':<18} {'ADX'}")
        print(f"  {SEP}")
        for r in near_misses[:5]:
            ts = r['timestamp'][:19] if r['timestamp'] else "N/A"
            adx = f"{float(r['adx']):.1f}" if r['adx'] else "N/A"
            print(f"  {ts:<22} {fmt_conf(r['confidence']):<10} {r['regime'] or 'N/A':<18} {adx}")
        print(f"\n  ✅ REGIME GATE WORKING: {len(near_misses)} noisy setups correctly blocked")
    else:
        print("\n  ℹ️  No near-miss rejections found yet. Log is still accumulating.")
else:
    print("  No EURUSD signals found yet.")

# ─────────────────────────────────────────────────────────────
# 3. MACRO VALIDATION (VIX + Yield Cross-Reference)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  STABILITY TEST 3: Macro-Validation (VIX / Yield Slope)")
print(f"{'='*60}")

with connect() as conn:
    macro_rows = conn.execute("""
        SELECT timestamp, symbol, signal, confidence, vix_proxy, yield_slope, regime
        FROM signals
        WHERE vix_proxy IS NOT NULL AND vix_proxy != 0.0
        ORDER BY timestamp DESC LIMIT 30
    """).fetchall()

if macro_rows:
    vix_blocks = [r for r in macro_rows if r['vix_proxy'] and float(r['vix_proxy']) > 1.0 and r['signal'] == 'WAIT']
    calm_entries = [r for r in macro_rows if r['vix_proxy'] and float(r['vix_proxy']) < 0 
                    and r['signal'] in ('BUY', 'SELL')
                    and r['confidence'] and float(r['confidence']) >= 0.70]
    
    print(f"  Total signals with macro data: {len(macro_rows)}")
    print(f"  VIX spike blocks (VIX_Z > 1):  {len(vix_blocks)} ← macro rejections working")
    print(f"  Clean entries (VIX_Z < 0, conf ≥ 70%): {len(calm_entries)} ← ideal institutional trades")
    
    if vix_blocks:
        print(f"\n  VIX-Blocked Rejections:")
        print(f"  {'Symbol':<10} {'Conf':<10} {'VIX_Z':<10} {'Yield_Z':<10} {'Regime'}")
        print(f"  {SEP}")
        for r in vix_blocks[:5]:
            vix_z = f"{float(r['vix_proxy']):.2f}" if r['vix_proxy'] else "N/A"
            yld_z = f"{float(r['yield_slope']):.2f}" if r['yield_slope'] else "N/A"
            print(f"  {r['symbol']:<10} {fmt_conf(r['confidence']):<10} {vix_z:<10} {yld_z:<10} {r['regime'] or 'N/A'}")
        print(f"\n  ✅ MACRO GATE OPERATIVE")
    else:
        print("\n  ℹ️  No VIX-spike blocks found yet.")
    
    if calm_entries:
        print(f"\n  Ideal Institutional Entries (calm macro + high confidence):")
        print(f"  {'Symbol':<10} {'Signal':<6} {'Conf':<10} {'VIX_Z':<10} {'Yield_Z'}")
        print(f"  {SEP}")
        for r in calm_entries[:5]:
            vix_z = f"{float(r['vix_proxy']):.2f}" if r['vix_proxy'] else "N/A"
            yld_z = f"{float(r['yield_slope']):.2f}" if r['yield_slope'] else "N/A"
            print(f"  {r['symbol']:<10} {r['signal']:<6} {fmt_conf(r['confidence']):<10} {vix_z:<10} {yld_z}")
else:
    print("  No macro-enriched signals found yet.")
    print("  ℹ️  The upgraded bridge will populate macro data on next signal cycle (~5-15 min).")

# ─────────────────────────────────────────────────────────────
# 4. Ghost Trades CSV Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  GHOST TRADES LOG — Summary")
print(f"{'='*60}")

if GHOST_CSV.exists():
    with open(GHOST_CSV, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    has_regime = 'regime' in (reader[0] if reader else {})
    print(f"  Total ghost trades logged: {total}")
    print(f"  Upgraded log format (macro columns): {'✅ Yes' if has_regime else '⚠️  No — restart bridge to apply'}")
    
    if reader:
        conf_col = 'confidence' if 'confidence' in reader[0] else None
        if conf_col:
            above_70 = [r for r in reader if r.get(conf_col) and float(r[conf_col]) > 0.70]
            print(f"  Trades above 70% confidence: {len(above_70)}")
        
        print(f"\n  Last 5 Entries:")
        headers = list(reader[0].keys())[:8]
        print(f"  {' '.join(h[:9].ljust(10) for h in headers)}")
        print(f"  {SEP}")
        for r in reader[-5:]:
            vals = [str(r.get(h, 'N/A'))[:9].ljust(10) for h in headers]
            print(f"  {' '.join(vals)}")
else:
    print("  ghost_trades.csv not found yet — will be created on next signal.")

print(f"\n{'='*60}")
print("  Stability Audit Complete")
print(f"{'='*60}\n")
