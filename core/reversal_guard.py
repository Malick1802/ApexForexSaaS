"""
core/reversal_guard.py
Dynamically calculates early exit reversal expectancy per Forex pair from all-time signals.db.
Maintains a dynamic whitelist of pairs where early exit on opposite signal is profitable.
"""
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("ReversalGuard")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "signals.db"

def get_reversal_expectancy() -> dict:
    """
    Calculate expectancy for all pairs using all-time data.
    Analyzes opposite signals generated within 48 hours of initial trade entries.
    Returns dict: {symbol: {'net_pips_saved': float, 'success_rate': float, 'total_events': int}}
    """
    expectancy = {}
    if not DB_PATH.exists():
        logger.warning(f"Database {DB_PATH} not found. Cannot calculate reversal expectancy.")
        return expectancy

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Load all completed trades
        signals = cur.execute("""
            SELECT timestamp, symbol, signal, is_hidden, outcome, price_at_signal, tp_price, sl_price, model_version, regime
            FROM signals
            WHERE signal IN ('BUY', 'SELL') AND outcome IN ('SUCCESS', 'FAIL')
            ORDER BY timestamp ASC
        """).fetchall()

        # Group signals by symbol for fast lookups
        symbol_signals = {}
        for sig in signals:
            sym = sig['symbol']
            if sym not in symbol_signals:
                symbol_signals[sym] = []
            symbol_signals[sym].append(sig)

        for sym, sym_sigs in symbol_signals.items():
            # Skip Gold and Crude Oil
            if 'XAU' in sym.upper() or 'OIL' in sym.upper():
                continue

            success_exits = 0
            total_reversals = 0
            total_pips_saved = 0.0

            for i, sig in enumerate(sym_sigs):
                # Analyze performance on v1 LIVE TRENDING trades
                if not (sig['model_version'] == 'v1' and 
                        sig['is_hidden'] == 0 and 
                        sig['regime'] is not None and 'TRENDING' in sig['regime']):
                    continue

                entry = sig['price_at_signal']
                tp = sig['tp_price']
                sl = sig['sl_price']
                direction = sig['signal']
                outcome = sig['outcome']

                if not entry or not tp or not sl:
                    continue

                try:
                    start_dt = datetime.fromisoformat(sig['timestamp'].replace('Z', '+00:00'))
                except Exception:
                    continue

                # Find if any opposite signal occurred within 48h
                for next_sig in sym_sigs[i+1:]:
                    try:
                        next_dt = datetime.fromisoformat(next_sig['timestamp'].replace('Z', '+00:00'))
                    except Exception:
                        continue

                    if next_dt - start_dt > timedelta(hours=48):
                        break

                    if next_sig['signal'] != direction:
                        # Found a reversal!
                        if direction == 'BUY':
                            orig_dist = (tp - entry) if outcome == 'SUCCESS' else (sl - entry)
                            exit_dist = (next_sig['price_at_signal'] - entry)
                        else:
                            orig_dist = (entry - tp) if outcome == 'SUCCESS' else (entry - sl)
                            exit_dist = (entry - next_sig['price_at_signal'])

                        benefit = exit_dist - orig_dist
                        is_jpy = 'JPY' in sym.upper()
                        multiplier = 100.0 if is_jpy else 10000.0
                        pips = benefit * multiplier

                        total_reversals += 1
                        total_pips_saved += pips
                        if pips > 0:
                            success_exits += 1
                        break

            if total_reversals > 0:
                expectancy[sym] = {
                    'net_pips_saved': total_pips_saved,
                    'success_rate': success_exits / total_reversals,
                    'total_events': total_reversals
                }

        conn.close()
    except Exception as e:
        logger.error(f"Error calculating reversal expectancy: {e}")

    return expectancy

def get_approved_reversal_pairs() -> set:
    """Return set of symbols where early exit on opposite signal has positive net pip expectancy and at least 3 historical events."""
    exp = get_reversal_expectancy()
    approved = {
        sym for sym, stats in exp.items() 
        if stats['net_pips_saved'] > 0 and stats['total_events'] >= 3
    }
    logger.info(f"Approved reversal pairs (expectancy > 0, events >= 3): {approved}")
    return approved
