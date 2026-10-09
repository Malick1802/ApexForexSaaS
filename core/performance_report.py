import sqlite3
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

PROFIT_REGEX = re.compile(r'(?:Profit:\s*)?\$([+-]?[\d,.]+)')

COMMODITY_SYMBOLS = {
    'XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'USOIL', 'USOIL.cash',
    'UKOIL', 'UKOIL.cash', 'BRENT', 'WTI', 'CrudeOIL', 'COPPER',
    'XPTUSD', 'XPDUSD', 'NGAS', 'NATGAS'
}

class PerformanceReporter:
    _cached_df: Optional[pd.DataFrame] = None
    _cached_time: float = 0.0

    @classmethod
    def invalidate_cache(cls):
        cls._cached_df = None
        cls._cached_time = 0.0

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = str(Path(__file__).resolve().parent.parent / "signals.db")
        else:
            self.db_path = db_path

    def _get_signals_df(self) -> pd.DataFrame:
        import time as _t
        now = _t.time()
        if PerformanceReporter._cached_df is not None and (now - PerformanceReporter._cached_time < 60.0):
            return PerformanceReporter._cached_df.copy()

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT id, timestamp, exit_time, duration_seconds, symbol, signal, confidence, confidence_tier,
                   is_hidden, outcome, exit_reason, price_at_signal, exit_price, sl_price, tp_price,
                   suggested_lots, sl_pips, tp_pips, mt5_ticket, model_version, is_manual, regime
            FROM signals
            WHERE signal IN ('BUY', 'SELL')
              AND outcome IN ('SUCCESS', 'FAIL')
            ORDER BY timestamp ASC
        ''', conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()

        df['t_utc'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
        df['conf'] = df['confidence'].astype(float)

        PerformanceReporter._cached_df = df
        PerformanceReporter._cached_time = now
        return df.copy()

    def _dedup(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Lifecycle-aware trade deduplication:
        A signal is only considered a duplicate if a trade on the same symbol and
        direction was ALREADY ACTIVE / RUNNING at that moment.
        Once the previous trade completes (exit_time < new trade timestamp), any new
        signal is a genuine, independent re-entry and is preserved.
        """
        if data.empty:
            return pd.DataFrame()

        data_sorted = data.sort_values('t_utc').copy()
        if 't_exit_utc' not in data_sorted.columns:
            data_sorted['t_exit_utc'] = pd.to_datetime(data_sorted['exit_time'], format='ISO8601', utc=True)

        active_until = {}
        seen_tickets = set()
        keep_indices = []

        for r in data_sorted.itertuples():
            ticket = getattr(r, 'mt5_ticket', None)
            if pd.notnull(ticket) and str(ticket).strip() not in ("", "0", "None"):
                t_str = str(ticket).replace(".0", "")
                if t_str in seen_tickets:
                    continue
                seen_tickets.add(t_str)
                keep_indices.append(r.Index)
                continue

            # Shadow / Paper trade intra-signal deduplication:

            key = (r.symbol, r.signal)
            entry_t = r.t_utc
            exit_t = getattr(r, 't_exit_utc', None)

            # If there's an active trade that hasn't closed yet at this entry time, it's an intra-trade duplicate
            if key in active_until:
                prev_exit = active_until[key]
                if pd.notnull(prev_exit) and entry_t < prev_exit:
                    # Previous trade still running — skip duplicate
                    continue
                elif pd.isnull(prev_exit) and (entry_t - active_until.get(f"{key}_entry", entry_t)).total_seconds() < 3600:
                    # Unresolved without exit time — 1 hour minimum cooldown
                    continue

            active_until[key] = exit_t
            active_until[f"{key}_entry"] = entry_t
            keep_indices.append(r.Index)

        return data_sorted.loc[keep_indices] if keep_indices else pd.DataFrame()

    def _get_mt5_deals_df(self) -> pd.DataFrame:
        try:
            from core.manual_model import get_mt5
            mt5 = get_mt5()
            if mt5:
                from_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                to_date = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                deals = mt5.history_deals_get(from_date, to_date)
                if deals:
                    deal_list = [d._asdict() for d in deals]
                    df_deals = pd.DataFrame(deal_list)
                    df_exits = df_deals[df_deals['entry'] == 1].copy() # 1 = DEAL_ENTRY_OUT
                    if not df_exits.empty and 'comment' in df_exits.columns:
                        # Exclude administrative cleanup / duplicate closes from strategy performance
                        df_exits = df_exits[~df_exits['comment'].astype(str).str.contains("Duplicate|Test|clean", case=False, na=False)].copy()
                    if not df_exits.empty:
                        # MetaTrader 5 deal.time is already stored in broker server time (Unix epoch seconds)
                        df_exits['t_trading'] = pd.to_datetime(df_exits['time'], unit='s')
                        return df_exits
        except Exception:
            pass
        return pd.DataFrame()

    def get_performance_matrix(
        self,
        period: str = "monthly", # "monthly" or "weekly"
        mode: str = "telegram_live", # "telegram_live" (sent to telegram), "production", "mt5_live", or "baseline"
        risk_per_trade: float = 50.0,
        reward_multiplier: float = 1.5,
        start_date: Optional[str] = "2026-08-01",
        end_date: Optional[str] = None,
        use_close_time: bool = True, # Base grouping on time of trade close
        account_size: Optional[float] = None
    ) -> pd.DataFrame:
        if account_size is None or account_size <= 0:
            account_size = 100000.0 if risk_per_trade >= 500.0 else 10000.0

        if mode == "mt5_live":
            # Master MT5 Account Executed Live Trades (Actual Real Broker Deals)
            df_deals = self._get_mt5_deals_df()
            if not df_deals.empty:
                if 'position_id' in df_deals.columns:
                    pos_df = df_deals.groupby('position_id').agg({
                        'symbol': 'first',
                        'time': 'last',
                        't_trading': 'last',
                        'profit': 'sum',
                        'volume': 'sum',
                        'comment': lambda x: ' | '.join(x)
                    }).reset_index()
                else:
                    pos_df = df_deals.copy()

                pos_df['time_metric'] = pos_df['t_trading']
                pos_df['t_naive'] = pos_df['t_trading']

                if start_date:
                    pos_df = pos_df[pos_df['t_naive'] >= pd.to_datetime(start_date)].copy()
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    if len(str(end_date).strip()) <= 10:
                        end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    pos_df = pos_df[pos_df['t_naive'] <= end_dt].copy()
                if pos_df.empty:
                    return pd.DataFrame()

                master_trade_risk = 500.0 if (account_size >= 50000.0 or risk_per_trade >= 250.0) else 50.0
                def calc_deal_r(r):
                    p = float(r.get('profit', 0.0))
                    if p > 1.5:
                        r_val = round(min(reward_multiplier, max(0.2, p / 450.0 if p >= 400.0 else p / master_trade_risk)), 2)
                        status = 'WIN'
                    elif p < -1.5:
                        r_val = round(max(-1.5, min(-0.2, p / master_trade_risk)), 2)
                        status = 'LOSS'
                    else:
                        r_val = 0.0
                        status = 'BREAKEVEN'
                    pnl_val = p if (account_size >= 50000.0 or risk_per_trade >= 250.0) else round(r_val * risk_per_trade, 2)
                    return pd.Series([r_val, pnl_val, status], index=['realized_r', 'pnl_amount', 'trade_status'])

                calc = pos_df.apply(calc_deal_r, axis=1)
                pos_df['realized_r'] = calc['realized_r']
                pos_df['pnl_amount'] = calc['pnl_amount']
                pos_df['trade_status'] = calc['trade_status']
                pos_df['outcome'] = np.where(pos_df['trade_status'] == 'WIN', 'SUCCESS', np.where(pos_df['trade_status'] == 'BREAKEVEN', 'BREAKEVEN', 'FAIL'))
                filtered = pos_df
            else:
                df = self._get_signals_df()
                if df.empty:
                    return pd.DataFrame()
                df['t_exit_utc'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True)
                df['time_metric'] = df['t_exit_utc'].fillna(df['t_utc']) if use_close_time else df['t_utc']
                filtered = df[df['mt5_ticket'].notnull()].copy()
                filtered = self._dedup(filtered)
        else:
            df = self._get_signals_df()
            if df.empty:
                return pd.DataFrame()

            df['t_exit_utc'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True)
            # Use exit time if available, otherwise entry time
            df['time_metric'] = df['t_exit_utc'].fillna(df['t_utc']) if use_close_time else df['t_utc']

            try:
                import zoneinfo
                trading_tz = zoneinfo.ZoneInfo("Europe/Athens")
                t_trading = df['time_metric'].dt.tz_convert(trading_tz)
            except Exception:
                from datetime import timedelta
                t_trading = df['time_metric'] + timedelta(hours=3)
            df['t_trading'] = t_trading
            df['t_naive'] = t_trading.dt.tz_localize(None)

            if start_date:
                df = df[df['t_naive'] >= pd.to_datetime(start_date)].copy()
            if end_date:
                end_dt = pd.to_datetime(end_date)
                if len(str(end_date).strip()) <= 10:
                    end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                df = df[df['t_naive'] <= end_dt].copy()
            if df.empty:
                return pd.DataFrame()

            from core.symbol_guard import is_symbol_blocked, is_direction_blocked

            if mode == "manual":
                # Manual M15 Wick Sniper Trades
                filtered = df[(df['is_manual'] == 1) | (df['model_version'] == 'manual_m15')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_ml_p60", "confluence_ml_p60_all"):
                # New Confluence ML Model (60% Partial TP + 2p BE)
                filtered = df[df['model_version'].isin(['confluence_ml_p60', 'confluence_ml_m15_p60']) & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_ml_p60_live",):
                filtered = df[df['model_version'].isin(['confluence_ml_p60', 'confluence_ml_m15_p60']) & df['mt5_ticket'].notnull() & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_std_p25", "confluence_std_p25_all"):
                # New Confluence Standard Model (25% Partial TP + 2p BE)
                filtered = df[df['model_version'].isin(['confluence_std_p25', 'confluence_m15_p25'])].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_std_p25_live",):
                filtered = df[df['model_version'].isin(['confluence_std_p25', 'confluence_m15_p25']) & df['mt5_ticket'].notnull()].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence", "confluence_live", "confluence_both_live"):
                # Confluence Models Live Broker Executions Only
                filtered = df[(df['model_version'].isin(['confluence_m15', 'confluence_ml_m15', 'confluence_ml_p60', 'confluence_std_p25']) | (df['regime'] == 'CONFLUENCE')) & df['mt5_ticket'].notnull()].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_standard", "confluence_standard_live", "confluence_std_live"):
                # Automated Confluence M15 Standard (Rule-Based Only, Live Executions)
                filtered = df[(df['model_version'] == 'confluence_m15') & df['mt5_ticket'].notnull()].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_ml", "confluence_ml_live", "confluence_ai_live"):
                # Confluence M15 + AI Quality Gate (Live Executions Only)
                filtered = df[(df['model_version'] == 'confluence_ml_m15') & df['mt5_ticket'].notnull() & (df['confidence'] >= 0.48) & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_all", "confluence_all_models"):
                # All Confluence Trades Across All 4 Models (Live Fills + Background Shadow Paper Trades)
                filtered = df[df['model_version'].isin(['confluence_m15', 'confluence_ml_m15', 'confluence_ml_p60', 'confluence_std_p25']) | (df['regime'] == 'CONFLUENCE')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_standard_all", "confluence_std_all"):
                # Confluence Standard Pure Rule-Based (Live + Background Shadow Paper Trades)
                filtered = df[(df['model_version'] == 'confluence_m15')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_ml_all", "confluence_ai_all"):
                # Confluence Deep Learning AI Gate (Live + Background Shadow Paper Trades)
                filtered = df[(df['model_version'] == 'confluence_ml_m15') & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("confluence_ml_suppressed", "confluence_suppressed"):
                # Suppressed Setups Only (Shadow evaluation of setups rejected by the AI Quality Gate)
                filtered = df[(df['model_version'].isin(['confluence_ml_m15', 'confluence_ml_p60'])) & (df['exit_reason'] == 'ML_SUPPRESSED')].copy()
                filtered = self._dedup(filtered)
            elif mode in ("foundation", "foundation_v1", "foundation_all"):
                # Foundation V1 Macro AI (All: Live + Background Shadow Paper Trades)
                filtered = df[df['model_version'].isin(['v1', 'foundation_tft', 'foundation'])].copy()
                filtered = self._dedup(filtered)
            elif mode == "foundation_live":
                # Foundation V1 Macro AI (Live MT5 Broker Executions Only)
                filtered = df[df['model_version'].isin(['v1', 'foundation_tft', 'foundation']) & df['mt5_ticket'].notnull()].copy()
                filtered = self._dedup(filtered)
            elif mode in ("dynamic_ytd", "dynamic_ytd_model", "dynamic_ytd_all", "dynamic_ytd_live"):
                # Dynamic YTD Model (Daily Winning Assets Strategy):
                # Restricts trades strictly to assets with Year-To-Date Net R >= 0.0 for that model
                from core.dynamic_model_whitelist import get_dynamic_whitelist_manager, normalize_model_key, normalize_symbol
                if "live" in mode:
                    filtered = df[df['mt5_ticket'].notnull() & df['outcome'].isin(['SUCCESS', 'FAIL'])].copy()
                else:
                    cand = df[df['outcome'].isin(['SUCCESS', 'FAIL']) & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                    if not cand.empty:
                        approved_set = get_dynamic_whitelist_manager().get_approved_set()
                        m_keys = [normalize_model_key(m) for m in cand['model_version']]
                        sym_keys = [normalize_symbol(s) for s in cand['symbol']]
                        cand['is_whitelisted'] = [(m, s) in approved_set for m, s in zip(m_keys, sym_keys)]
                        filtered = cand[cand['is_whitelisted']].copy()
                    else:
                        filtered = cand
                filtered = self._dedup(filtered)
            elif mode in ("aggregate_all", "all_models"):
                # All Strategy Models Aggregated (Live + Shadow Paper Trades Combined)
                filtered = df[df['outcome'].isin(['SUCCESS', 'FAIL'])].copy()
                filtered = self._dedup(filtered)
            elif mode == "aggregate_live":
                # Selected Live Models Aggregated (Executed Real Broker Trades Only)
                filtered = df[df['mt5_ticket'].notnull() & df['outcome'].isin(['SUCCESS', 'FAIL'])].copy()
                filtered = self._dedup(filtered)
            elif mode == "telegram_live":
                # Live Alerts Sent to Telegram: non-hidden signals, active symbols, shielded (61%+ Forex / 55%+ Commodities)
                from core.symbol_guard import is_commodity
                def is_valid_tg(r):
                    if r.get('is_manual') == 1 or r.get('model_version') in ('manual_m15', 'confluence_m15', 'confluence_ml_m15'):
                        return True
                    sym = r['symbol']
                    sig = r['signal']
                    conf = float(r.get('conf', 0))
                    min_conf = 0.55 if is_commodity(sym) else 0.61
                    if conf < min_conf:
                        return False
                    if is_symbol_blocked(sym):
                        return False
                    if is_direction_blocked(sym, sig):
                        return False
                    return bool(r.get('is_hidden', 0) == 0)

                filtered = df[df.apply(is_valid_tg, axis=1)].copy()
                filtered = self._dedup(filtered)
            elif mode == "production":
                # Live Production: 61%+ for Forex, 55%+ for Commodities
                from core.symbol_guard import is_commodity
                def is_valid_prod(r):
                    if r.get('is_manual') == 1 or r.get('model_version') in ('manual_m15', 'confluence_m15'):
                        return True
                    sym = r['symbol']
                    sig = r['signal']
                    conf = float(r.get('conf', 0))
                    min_conf = 0.55 if is_commodity(sym) else 0.61
                    if conf < min_conf:
                        return False
                    if is_symbol_blocked(sym):
                        return False
                    if is_direction_blocked(sym, sig):
                        return False
                    return True

                filtered = df[df.apply(is_valid_prod, axis=1)].copy()
                filtered = self._dedup(filtered)
            else: # "baseline"
                # 50.0%+ floor, active traded instruments
                filtered = df[((df['conf'] >= 0.50) | (df['is_manual'] == 1) | (df['model_version'] in ('manual_m15', 'confluence_m15'))) & (~df['symbol'].apply(is_symbol_blocked))].copy()
                filtered = self._dedup(filtered)

        if filtered.empty:
            return pd.DataFrame()

        def robust_calc_trade(row):
            reason = str(row.get('exit_reason') or '')
            outcome = row.get('outcome')
            model_ver = str(row.get('model_version') or '')
            
            # Check raw dollar profit from exit_reason
            m = PROFIT_REGEX.search(reason)
            raw_profit = float(m.group(1).replace(',', '')) if m else None
            
            # Check price distance
            entry = row.get('price_at_signal')
            sl = row.get('sl_price')
            exit_p = row.get('exit_price')
            sig = row.get('signal')
            price_r = None
            if entry and sl and exit_p and entry != sl and not pd.isna(entry) and not pd.isna(sl) and not pd.isna(exit_p):
                risk_dist = abs(entry - sl)
                gain_dist = (exit_p - entry) if sig == 'BUY' else (entry - exit_p)
                price_r = gain_dist / risk_dist
                
            is_be = False
            if raw_profit is not None and abs(raw_profit) < 2.0:
                is_be = True
            elif price_r is not None and abs(price_r) < 0.12 and outcome == 'FAIL':
                is_be = True
            elif "BE hit" in reason or "Breakeven" in reason or "SL hit ($0.00)" in reason or "BE Profit: $0.00" in reason:
                is_be = True
                
            if is_be:
                return pd.Series([0.0, 0.0, 'BREAKEVEN'], index=['realized_r', 'pnl_amount', 'trade_status'])
                
            if outcome == 'FAIL':
                # Genuine SL loss is always -1.0R (clamped to realistic bounds if price_r available)
                r_val = -1.0
                if price_r is not None and -1.2 <= price_r <= -0.5:
                    r_val = round(price_r, 2)
                return pd.Series([r_val, r_val * risk_per_trade, 'LOSS'], index=['realized_r', 'pnl_amount', 'trade_status'])
                
            if outcome == 'SUCCESS':
                is_p60 = ("p60" in model_ver) or ("p60" in mode)
                is_p25 = ("p25" in model_ver) or ("p25" in mode)
                
                # Dynamic Partial + BE Runner detection across any account size ($10k or $100k)
                is_partial_be = False
                if "Partial" in reason or "BE" in reason:
                    is_partial_be = True
                elif price_r is not None and 0.05 <= price_r < 0.6:
                    is_partial_be = True
                elif raw_profit is not None:
                    ref_risk = 500.0 if (account_size >= 50000.0 or risk_per_trade >= 250.0) else 50.0
                    r_est = raw_profit / ref_risk
                    if 0.08 <= r_est <= 0.65:
                        is_partial_be = True

                if is_p60:
                    if is_partial_be:
                        r_val = 0.90  # TP1 (60% @ 1.5R) + BE runner
                    elif price_r is not None and price_r >= 1.4:
                        r_val = 1.50  # Full TP reached on runner
                    else:
                        r_val = 1.15
                elif is_p25:
                    if is_partial_be:
                        r_val = 0.38  # TP1 (25% @ 1.5R) + BE runner
                    elif price_r is not None and price_r >= 1.4:
                        r_val = 1.50  # Full TP reached
                    else:
                        r_val = 0.94
                else:
                    if "Friday" in reason and price_r is not None and price_r > 0:
                        r_val = round(min(reward_multiplier, max(0.2, price_r)), 2)
                    else:
                        r_val = reward_multiplier
                        
                return pd.Series([r_val, r_val * risk_per_trade, 'WIN'], index=['realized_r', 'pnl_amount', 'trade_status'])
                
            return pd.Series([0.0, 0.0, 'OTHER'], index=['realized_r', 'pnl_amount', 'trade_status'])

        if 'trade_status' not in filtered.columns:
            calc_df = filtered.apply(robust_calc_trade, axis=1)
            filtered['realized_r'] = calc_df['realized_r']
            filtered['pnl_amount'] = calc_df['pnl_amount']
            filtered['trade_status'] = calc_df['trade_status']

        if 't_naive' in filtered.columns:
            t_naive = filtered['t_naive']
        else:
            try:
                import zoneinfo
                trading_tz = zoneinfo.ZoneInfo("Europe/Athens")
                t_trading = filtered['time_metric'].dt.tz_convert(trading_tz)
            except Exception:
                from datetime import timedelta
                t_trading = filtered['time_metric'] + timedelta(hours=3)
            t_naive = t_trading.dt.tz_localize(None)
        if period == "monthly":
            filtered['period_obj'] = t_naive.dt.to_period('M')
            format_fn = lambda p: str(p)
        elif period == "daily":
            filtered['period_obj'] = t_naive.dt.to_period('D')
            format_fn = lambda p: p.start_time.strftime('%Y-%m-%d (%a)')
        else: # "weekly"
            filtered['period_obj'] = t_naive.dt.to_period('W-SUN')
            format_fn = lambda p: f"{p.start_time.strftime('%b %d')} - {p.end_time.strftime('%b %d')} (W{p.week:02d})"

        periods = sorted(filtered['period_obj'].unique(), reverse=True)
        rows = []

        for p_obj in periods:
            sub = filtered[filtered['period_obj'] == p_obj]
            tot = len(sub)
            if tot == 0:
                continue

            pnl = float(sub['pnl_amount'].sum())
            net_r = float(sub['realized_r'].sum())
            gross_win_r = float(sub[sub['realized_r'] > 0]['realized_r'].sum())
            gross_loss_r = abs(float(sub[sub['realized_r'] < 0]['realized_r'].sum()))
            pf = round(gross_win_r / gross_loss_r, 2) if gross_loss_r > 0 else (999.0 if gross_win_r > 0 else 0.0)
            w = len(sub[sub['trade_status'] == 'WIN'])
            l = len(sub[sub['trade_status'] == 'LOSS'])
            be = len(sub[sub['trade_status'] == 'BREAKEVEN'])
            wr = (w / tot * 100.0) if tot > 0 else 0.0

            rec_str = f"{w}W - {l}L" + (f" - {be}BE" if be > 0 else "")
            rows.append({
                'Period': format_fn(p_obj),
                'Trades': tot,
                'Record': rec_str,
                'Wins': w,
                'Losses': l,
                'Breakeven': be,
                'Win Rate (%)': round(wr, 1),
                'Net R': round(net_r, 2),
                'Profit Factor': pf,
                'Net PnL ($)': round(pnl, 2),
                'Return (%)': round((pnl / account_size) * 100.0, 2)
            })

        return pd.DataFrame(rows)

    def get_trades_for_day(
        self,
        date_str: str,
        mode: str = "dynamic_ytd_live",
        risk_per_trade: float = 50.0,
        account_size: Optional[float] = None
    ) -> pd.DataFrame:
        """Return all individual trades that closed on a specific calendar trading day."""
        clean_date = date_str.split(' ')[0].strip()
        if account_size is None or account_size <= 0:
            account_size = 100000.0 if risk_per_trade >= 500.0 else 10000.0

        try:
            import zoneinfo
            trading_tz = zoneinfo.ZoneInfo("Europe/Athens")
        except Exception:
            trading_tz = None

        if mode == "mt5_live":
            # For MT5 live executed trades: extract directly from real broker history deals!
            df_deals = self._get_mt5_deals_df()
            if df_deals.empty:
                return pd.DataFrame()
            if 'position_id' in df_deals.columns:
                pos_df = df_deals.groupby('position_id').agg({
                    'symbol': 'first',
                    'time': 'last',
                    't_trading': 'last',
                    'profit': 'sum',
                    'volume': 'sum',
                    'comment': lambda x: ' | '.join(x)
                }).reset_index()
            else:
                pos_df = df_deals.copy()

            pos_df['date_key'] = pos_df['t_trading'].dt.strftime('%Y-%m-%d')
            day_deals = pos_df[pos_df['date_key'] == clean_date].copy()
            if day_deals.empty:
                return pd.DataFrame()

            master_risk = 500.0 if (account_size >= 50000.0 or risk_per_trade >= 250.0) else 50.0
            def calc_deal_drill(r):
                p = float(r.get('profit', 0.0))
                if p > 1.5:
                    r_val = round(min(1.5, max(0.2, p / 450.0 if p >= 400.0 else p / master_risk)), 2)
                    st = 'WIN'
                elif p < -1.5:
                    r_val = round(max(-1.5, min(-0.2, p / master_risk)), 2)
                    st = 'LOSS'
                else:
                    r_val = 0.0
                    st = 'BREAKEVEN'
                pnl_val = p if (account_size >= 50000.0 or risk_per_trade >= 250.0) else round(r_val * risk_per_trade, 2)
                return pd.Series([r_val, pnl_val, st], index=['realized_r', 'pnl_amount', 'status'])

            c = day_deals.apply(calc_deal_drill, axis=1)
            day_deals['realized_r'] = c['realized_r']
            day_deals['pnl_amount'] = c['pnl_amount']
            day_deals['status'] = c['status']

            res = pd.DataFrame({
                'Ticket': day_deals['position_id'].astype(str),
                'Time (Broker)': day_deals['t_trading'].dt.strftime('%H:%M:%S'),
                'Symbol': day_deals['symbol'],
                'Direction': day_deals['comment'].apply(lambda x: 'BUY' if 'BUY' in str(x) else ('SELL' if 'SELL' in str(x) else '-')),
                'Model': day_deals['comment'].apply(lambda x: str(x).split(' ')[0] if ' ' in str(x) else str(x)),
                'Status': day_deals['status'],
                'Edge (R)': day_deals['realized_r'].apply(lambda x: f"{x:+.2f}R"),
                'PnL ($)': day_deals['pnl_amount'].apply(lambda x: f"${x:+,.2f}"),
                'Exit Reason': day_deals['comment'].fillna('-')
            })
            return res.sort_values('Time (Broker)', ascending=False)

        df = self._get_signals_df()
        if df.empty:
            return pd.DataFrame()
        df['t_exit_utc'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True)
        df['time_metric'] = df['t_exit_utc'].fillna(df['t_utc'])
        if trading_tz is not None:
            t_trading = df['time_metric'].dt.tz_convert(trading_tz)
        else:
            from datetime import timedelta
            t_trading = df['time_metric'] + timedelta(hours=3)
        df['date_key'] = t_trading.dt.strftime('%Y-%m-%d')
        df['time_trading'] = t_trading

        if mode in ("dynamic_ytd", "dynamic_ytd_model", "dynamic_ytd_all", "dynamic_ytd_live"):
            from core.dynamic_model_whitelist import get_dynamic_whitelist_manager, normalize_model_key, normalize_symbol
            if "live" in mode:
                filtered = df[df['mt5_ticket'].notnull() & df['outcome'].isin(['SUCCESS', 'FAIL'])].copy()
            else:
                cand = df[df['outcome'].isin(['SUCCESS', 'FAIL']) & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
                if not cand.empty:
                    approved_set = get_dynamic_whitelist_manager().get_approved_set()
                    m_keys = [normalize_model_key(m) for m in cand['model_version']]
                    sym_keys = [normalize_symbol(s) for s in cand['symbol']]
                    cand['is_whitelisted'] = [(m, s) in approved_set for m, s in zip(m_keys, sym_keys)]
                    filtered = cand[cand['is_whitelisted']].copy()
                else:
                    filtered = cand
            filtered = self._dedup(filtered)
        elif mode in ("confluence_ml_p60", "confluence_ml_p60_all"):
            filtered = df[df['model_version'].isin(['confluence_ml_p60', 'confluence_ml_m15_p60']) & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
            filtered = self._dedup(filtered)
        elif mode in ("confluence_ml_p60_live",):
            filtered = df[df['model_version'].isin(['confluence_ml_p60', 'confluence_ml_m15_p60']) & df['mt5_ticket'].notnull() & (df['exit_reason'] != 'ML_SUPPRESSED')].copy()
            filtered = self._dedup(filtered)
        elif mode in ("confluence_std_p25", "confluence_std_p25_all"):
            filtered = df[df['model_version'].isin(['confluence_std_p25', 'confluence_m15_p25'])].copy()
            filtered = self._dedup(filtered)
        elif mode in ("confluence_std_p25_live",):
            filtered = df[df['model_version'].isin(['confluence_std_p25', 'confluence_m15_p25']) & df['mt5_ticket'].notnull()].copy()
            filtered = self._dedup(filtered)
        elif mode in ("confluence", "confluence_live", "confluence_both_live"):
            filtered = df[(df['model_version'].isin(['confluence_m15', 'confluence_ml_m15', 'confluence_ml_p60', 'confluence_std_p25']) | (df['regime'] == 'CONFLUENCE')) & df['mt5_ticket'].notnull()].copy()
            filtered = self._dedup(filtered)
        else:
            filtered = df[df['outcome'].isin(['SUCCESS', 'FAIL'])].copy()
            filtered = self._dedup(filtered)

        day_trades = filtered[filtered['date_key'] == clean_date].copy()
        if day_trades.empty:
            return pd.DataFrame()

        def calc_row(r):
            reason = str(r.get('exit_reason') or '')
            outcome = r.get('outcome')
            m = PROFIT_REGEX.search(reason)
            raw_p = float(m.group(1).replace(',', '')) if m else None
            model_ver = str(r.get('model_version') or '')
            if "BE hit" in reason or "Breakeven" in reason or "BE Profit" in reason or (raw_p is not None and abs(raw_p) < 2.0):
                return pd.Series([0.0, 0.0, 'BREAKEVEN'], index=['realized_r', 'pnl_amount', 'status'])
            if outcome == 'FAIL':
                return pd.Series([-1.0, -risk_per_trade, 'LOSS'], index=['realized_r', 'pnl_amount', 'status'])
            if outcome == 'SUCCESS':
                is_p60 = "p60" in model_ver or "p60" in mode
                is_p25 = "p25" in model_ver or "p25" in mode
                is_partial_be = ("Partial" in reason) or ("BE" in reason)
                if is_p60:
                    r_val = 0.90 if is_partial_be else 1.50
                elif is_p25:
                    r_val = 0.38 if is_partial_be else 1.50
                else:
                    r_val = 1.50
                return pd.Series([r_val, r_val * risk_per_trade, 'WIN'], index=['realized_r', 'pnl_amount', 'status'])
            return pd.Series([0.0, 0.0, 'OTHER'], index=['realized_r', 'pnl_amount', 'status'])

        c = day_trades.apply(calc_row, axis=1)
        day_trades['realized_r'] = c['realized_r']
        day_trades['pnl_amount'] = c['pnl_amount']
        day_trades['status'] = c['status']

        res = pd.DataFrame({
            'Ticket': day_trades['mt5_ticket'].fillna('-').astype(str).str.replace(r'\.0$', '', regex=True),
            'Time (Broker)': day_trades['time_trading'].dt.strftime('%H:%M:%S'),
            'Symbol': day_trades['symbol'],
            'Direction': day_trades['signal'],
            'Model': day_trades['model_version'].fillna('v1'),
            'Status': day_trades['status'],
            'Edge (R)': day_trades['realized_r'].apply(lambda x: f"{x:+.2f}R"),
            'PnL ($)': day_trades['pnl_amount'].apply(lambda x: f"${x:+,.2f}"),
            'Exit Reason': day_trades['exit_reason'].fillna('-')
        })
        return res.sort_values('Time (Broker)', ascending=False)

    def generate_telegram_scorecard(
        self,
        period: str = "all", # "daily", "weekly", "monthly", "both", or "all"
        risk_per_trade: float = 50.0,
        mode: str = "production",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        account_size: Optional[float] = None
    ) -> str:
        if account_size is None or account_size <= 0:
            account_size = 100000.0 if risk_per_trade >= 500.0 else 10000.0

        policy_label = "🎯 Manual M15 Wick Sniper Trades" if mode == "manual" else ("Master MT5 Executed Trades" if mode == "mt5_live" else ("Live Telegram Signals" if mode == "telegram_live" else "61.0%+ Live Production (Shielded)"))
        msg_parts = []
        msg_parts.append("📊 *ForexAlert AI · PERFORMANCE SCORECARD*")
        msg_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        msg_parts.append(f"⏱️ *Basis:* Forex Trading Day (Broker Server Time)")
        msg_parts.append(f"🛡️ *Policy:* {policy_label}")
        msg_parts.append(f"💰 *Base Risk:* ${risk_per_trade:,.0f} / trade (Account: ${account_size:,.0f})\n")

        if period in ("daily", "all"):
            df_d = self.get_performance_matrix(period="daily", mode=mode, risk_per_trade=risk_per_trade, start_date=start_date, end_date=end_date, account_size=account_size, use_close_time=True)
            if not df_d.empty:
                msg_parts.append("☀️ *RECENT DAILY BREAKDOWN (Last 5 Days)*")
                msg_parts.append("```")
                msg_parts.append("Day          W-L   Win%   Net R   PnL($)")
                msg_parts.append("---------------------------------------")
                for _, r in df_d.head(5).iterrows():
                    p_str = str(r['Period']).split(' ')[0]
                    wl = f"{int(r['Wins'])}-{int(r['Losses'])}"
                    wr = f"{r['Win Rate (%)']:.0f}%"
                    nr = f"{r['Net R']:+.1f}R"
                    pnl = f"${r['Net PnL ($)']:+,.0f}"
                    msg_parts.append(f"{p_str:<10} {wl:>5}  {wr:>4} {nr:>7} {pnl:>7}")
                msg_parts.append("```\n")

        if period in ("monthly", "both", "all"):
            df_m = self.get_performance_matrix(period="monthly", mode=mode, risk_per_trade=risk_per_trade, start_date=start_date, end_date=end_date, account_size=account_size, use_close_time=True)
            msg_parts.append("🗓️ *MONTHLY BREAKDOWN*")
            msg_parts.append("```")
            msg_parts.append("Period    W-L   Win%   Net R   PnL($)")
            msg_parts.append("-------------------------------------")
            for _, r in df_m.iterrows():
                p = str(r['Period'])
                wl = f"{int(r['Wins'])}-{int(r['Losses'])}"
                wr = f"{r['Win Rate (%)']:.0f}%"
                nr = f"{r['Net R']:+.1f}R"
                pnl = f"${r['Net PnL ($)']:+,.0f}"
                msg_parts.append(f"{p:<7} {wl:>5}  {wr:>4} {nr:>7} {pnl:>7}")
            msg_parts.append("```\n")

        if period in ("weekly", "both", "all"):
            # Sort weeks by start date descending
            df_w = self.get_performance_matrix(period="weekly", mode=mode, risk_per_trade=risk_per_trade, start_date=start_date, end_date=end_date, account_size=account_size, use_close_time=True)
            # Take exactly the last 6 weeks
            msg_parts.append("📅 *RECENT WEEKS BREAKDOWN (Last 6 Weeks)*")
            msg_parts.append("```")
            msg_parts.append("Week      W-L   Win%   Net R   PnL($)")
            msg_parts.append("-------------------------------------")
            for _, r in df_w.head(6).iterrows():
                w_str = str(r['Period']).split(' - ')[0] if ' - ' in str(r['Period']) else str(r['Period'])
                wl = f"{int(r['Wins'])}-{int(r['Losses'])}"
                wr = f"{r['Win Rate (%)']:.0f}%"
                nr = f"{r['Net R']:+.1f}R"
                pnl = f"${r['Net PnL ($)']:+,.0f}"
                msg_parts.append(f"{w_str:<7} {wl:>5}  {wr:>4} {nr:>7} {pnl:>7}")
            msg_parts.append("```\n")

        # Totals
        df_all = self.get_performance_matrix(period="monthly", mode=mode, risk_per_trade=risk_per_trade, start_date=start_date, end_date=end_date, account_size=account_size, use_close_time=True)
        if not df_all.empty:
            tot_t = df_all['Trades'].sum()
            tot_w = df_all['Wins'].sum()
            tot_l = df_all['Losses'].sum()
            tot_wr = (tot_w / tot_t * 100.0) if tot_t > 0 else 0.0
            tot_r = df_all['Net R'].sum()
            tot_pnl = df_all['Net PnL ($)'].sum()
            ret_pct = (tot_pnl / account_size) * 100.0
            msg_parts.append("🏆 *TOTAL PERIOD METRICS*")
            msg_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            msg_parts.append(f"• *Closed Setups:* {int(tot_t)} Trades")
            msg_parts.append(f"• *Overall Record:* *{int(tot_w)}W – {int(tot_l)}L* (*{tot_wr:.1f}% Win Rate*)")
            msg_parts.append(f"• *Cumulative Edge:* *{tot_r:+.2f}R*")
            msg_parts.append(f"• *Total Net Profit:* *${tot_pnl:+,.2f}* (*{ret_pct:+.1f}% on ${account_size:,.0f}*)")

        msg_parts.append(f"\n_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
        return "\n".join(msg_parts)

    def get_model_comparison_breakdown(
        self,
        risk_per_trade: float = 50.0,
        start_date: Optional[str] = "2026-08-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare all active, shadow, and baseline models side-by-side, plus aggregate performance:
        - 🧠 Confluence M15 + Deep Learning (AI Gate)
        - ⚡ Confluence M15 Standard (Rule-Based)
        - 🛡️ Confluence M15 Suppressed Setups (ML-Blocked)
        - 🌐 Foundation V1 Macro AI
        - 🎯 Manual M15 Wick Sniper
        - 🏦 Master MT5 Executed Deals
        - 🌟 Aggregate: All Models Combined (Live + Shadow)
        - 🏆 Aggregate: Selected Live Models
        """
        try:
            from core.model_gatekeeper import load_gatekeeper_config
            gate_cfg = load_gatekeeper_config()
        except Exception:
            gate_cfg = {}

        active_counts = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT COALESCE(model_version, 'v1'), COUNT(*)
                    FROM signals
                    WHERE outcome = 'ACTIVE' AND signal IN ('BUY', 'SELL')
                    GROUP BY COALESCE(model_version, 'v1')
                """)
                for mv, cnt in cur.fetchall():
                    active_counts[mv] = cnt
        except Exception:
            pass

        try:
            from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
            ytd_sub_models = get_dynamic_whitelist_manager().get_sub_models_config()
        except Exception:
            ytd_sub_models = {}

        model_active_map = {
            "dynamic_ytd_model": sum(v for k, v in active_counts.items() if gate_cfg.get(k, False) and ytd_sub_models.get(k, True)),
            "confluence_ml_p60": active_counts.get("confluence_ml_p60", 0),
            "confluence_std_p25": active_counts.get("confluence_std_p25", 0),
            "confluence_ml_all": active_counts.get("confluence_ml_m15", 0),
            "confluence_standard_all": active_counts.get("confluence_m15", 0),
            "confluence_ml_suppressed": 0,
            "foundation_all": active_counts.get("v1", 0),
            "manual": active_counts.get("manual_m15", 0),
            "mt5_live": sum(v for k, v in active_counts.items() if "confluence" in k or k == "manual_m15"),
            "aggregate_all": sum(active_counts.values()),
            "aggregate_live": sum(v for k, v in active_counts.items() if gate_cfg.get(k, False)),
        }

        active_sub_cnt = sum(1 for v in ytd_sub_models.values() if v)
        models_to_compare = [
            (f"🌟 Dynamic YTD Model ({active_sub_cnt} Active Models)", "dynamic_ytd_model", "🌟 LIVE ACTIVE" if gate_cfg.get("dynamic_ytd_model", True) else "👻 SHADOW MODE"),
            ("🧠 Confluence ML M15 (60% TP + 2p BE)", "confluence_ml_p60", "🟢 LIVE ACTIVE" if gate_cfg.get("confluence_ml_p60") else "👻 SHADOW MODE"),
            ("⚡ Confluence Standard M15 (25% TP + 2p BE)", "confluence_std_p25", "🟢 LIVE ACTIVE" if gate_cfg.get("confluence_std_p25") else "👻 SHADOW MODE"),
            ("🧠 Confluence AI Quality Gate (Fixed 1.5R)", "confluence_ml_all", "🟢 LIVE ACTIVE" if gate_cfg.get("confluence_ml_m15") else "👻 SHADOW MODE"),
            ("⚡ Confluence Standard Rule-Based (Fixed 1.5R)", "confluence_standard_all", "🟢 LIVE ACTIVE" if gate_cfg.get("confluence_m15") else "👻 SHADOW MODE"),
            ("🛡️ Confluence Suppressed Trades", "confluence_ml_suppressed", "🛡️ AI FILTERED"),
            ("🌐 Foundation V1 Macro AI", "foundation_all", "🟢 LIVE ACTIVE" if gate_cfg.get("foundation_v1") else "👻 SHADOW MODE"),
            ("🎯 Manual M15 Sniper", "manual", "🟢 LIVE ACTIVE" if gate_cfg.get("manual_m15", True) else "👻 SHADOW MODE"),
            ("🏦 Master MT5 Live Fills", "mt5_live", "🏦 BROKER FILLS"),
            ("🌟 Aggregate: All Models Combined", "aggregate_all", "🌟 AGGREGATE (ALL)"),
            ("🏆 Aggregate: Selected Live Models", "aggregate_live", "🏆 AGGREGATE (LIVE)"),
        ]
        rows = []
        for label, m_key, status_label in models_to_compare:
            try:
                df = self.get_performance_matrix(
                    period="monthly",
                    mode=m_key,
                    risk_per_trade=risk_per_trade,
                    start_date=start_date,
                    end_date=end_date,
                    use_close_time=True
                )
                if not df.empty:
                    tot_t = int(df['Trades'].sum())
                    tot_w = int(df['Wins'].sum())
                    tot_l = int(df['Losses'].sum())
                    tot_be = int(df['Breakeven'].sum()) if 'Breakeven' in df.columns else (tot_t - tot_w - tot_l)
                    wr = (tot_w / tot_t * 100.0) if tot_t > 0 else 0.0
                    net_r = float(df['Net R'].sum())
                    pnl = float(df['Net PnL ($)'].sum())
                    gross_win_r = float(df['Wins'].sum()) * (1.15 if ('p60' in m_key or 'dynamic' in m_key) else (0.94 if 'p25' in m_key else 1.5))
                    gross_loss_r = float(df['Losses'].sum()) * 1.0
                    pf = round(gross_win_r / gross_loss_r, 2) if gross_loss_r > 0 else (999.0 if gross_win_r > 0 else 0.0)
                else:
                    tot_t, tot_w, tot_l, tot_be, wr, net_r, pnl, pf = 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

                record_str = f"{tot_w}W – {tot_l}L" + (f" – {tot_be}BE" if tot_be > 0 else "")
                rows.append({
                    "Model Engine": label,
                    "Execution Mode": status_label,
                    "Active Setups": model_active_map.get(m_key, 0),
                    "Total Trades": tot_t,
                    "Record (W-L)": record_str,
                    "Win Rate (%)": f"{wr:.1f}%",
                    "Net Edge (R)": f"{net_r:+.1f}R",
                    "Net PnL ($)": f"${pnl:+,.0f}",
                    "Profit Factor": f"{pf:.2f}",
                })
            except Exception as e:
                pass
        return pd.DataFrame(rows)


