import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

COMMODITY_SYMBOLS = {
    'XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'USOIL', 'USOIL.cash',
    'UKOIL', 'UKOIL.cash', 'BRENT', 'WTI', 'CrudeOIL', 'COPPER',
    'XPTUSD', 'XPDUSD', 'NGAS', 'NATGAS'
}

class PerformanceReporter:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = str(Path(__file__).resolve().parent.parent / "signals.db")
        else:
            self.db_path = db_path

    def _get_signals_df(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT id, timestamp, exit_time, duration_seconds, symbol, signal, confidence, confidence_tier,
                   is_hidden, outcome, exit_reason, price_at_signal, exit_price, mt5_ticket, model_version
            FROM signals
            WHERE signal IN ('BUY', 'SELL')
              AND (model_version = 'v1' OR model_version IS NULL)
              AND outcome IN ('SUCCESS', 'FAIL')
            ORDER BY timestamp ASC
        ''', conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()

        df['t_utc'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
        df['conf'] = df['confidence'].astype(float)
        return df

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

        trades = []
        active_until = {}

        for _, r in data_sorted.iterrows():
            key = (r['symbol'], r['signal'])
            entry_t = r['t_utc']
            exit_t = r.get('t_exit_utc')

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
            trades.append(r)

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def _get_mt5_deals_df(self) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5
            from pathlib import Path
            ftmo_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
            init_kwargs = {"timeout": 5000}
            if Path(ftmo_path).exists():
                init_kwargs["path"] = ftmo_path
            if mt5.initialize(**init_kwargs):
                from_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                to_date = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                deals = mt5.history_deals_get(from_date, to_date)
                mt5.shutdown()
                if deals:
                    deal_list = [d._asdict() for d in deals]
                    df_deals = pd.DataFrame(deal_list)
                    df_exits = df_deals[df_deals['entry'] == 1].copy() # 1 = DEAL_ENTRY_OUT
                    if not df_exits.empty and 'comment' in df_exits.columns:
                        # Exclude administrative cleanup / duplicate closes from strategy performance
                        df_exits = df_exits[~df_exits['comment'].astype(str).str.contains("Duplicate|Test|clean", case=False, na=False)].copy()
                    if not df_exits.empty:
                        df_exits['t_utc'] = pd.to_datetime(df_exits['time'], unit='s', utc=True)
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

        df = self._get_signals_df()
        if df.empty:
            return pd.DataFrame()

        df['t_exit_utc'] = pd.to_datetime(df['exit_time'], format='ISO8601', utc=True)
        # Use exit time if available, otherwise entry time
        df['time_metric'] = df['t_exit_utc'].fillna(df['t_utc']) if use_close_time else df['t_utc']

        if start_date:
            df = df[df['time_metric'] >= pd.to_datetime(start_date, utc=True)].copy()
        if end_date:
            df = df[df['time_metric'] <= pd.to_datetime(end_date, utc=True)].copy()

        from core.symbol_guard import is_symbol_blocked, is_direction_blocked

        if mode == "mt5_live":
            # Master MT5 Account Executed Live Trades (Live Broker Fills)
            filtered = df[df['mt5_ticket'].notnull() & (~df['symbol'].apply(is_symbol_blocked))].copy()
            filtered = self._dedup(filtered)
        elif mode == "telegram_live":
            # Live Alerts Sent to Telegram: non-hidden signals, active symbols, shielded (61%+ Forex / 55%+ Commodities)
            from core.symbol_guard import is_commodity
            def is_valid_tg(r):
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
            # Live Production: 61%+ for Forex, 55%+ for Commodities, EURUSD BUY, EURCAD BUY & XAUUSD BUY shielded
            from core.symbol_guard import is_commodity
            def is_valid_prod(r):
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
            filtered = df[(df['conf'] >= 0.50) & (~df['symbol'].apply(is_symbol_blocked))].copy()
            filtered = self._dedup(filtered)

        if filtered.empty:
            return pd.DataFrame()

        if mode == "mt5_live":
            def calc_mt5_trade(row):
                reason = str(row.get('exit_reason') or '')
                import re
                m = re.search(r'Profit:\s*\$([+-]?[\d,.]+)', reason)
                if m:
                    raw_profit = float(m.group(1).replace(',', ''))
                    # Master MT5 account executed live trades at base risk = $50.0 (0.5% on $10k)
                    r_val = raw_profit / 50.0
                    pnl_val = r_val * risk_per_trade
                    return pd.Series([r_val, pnl_val], index=['realized_r', 'pnl_amount'])
                if row.get('outcome') == 'SUCCESS':
                    return pd.Series([reward_multiplier, risk_per_trade * reward_multiplier], index=['realized_r', 'pnl_amount'])
                elif row.get('outcome') == 'FAIL':
                    return pd.Series([-1.0, -risk_per_trade], index=['realized_r', 'pnl_amount'])
                return pd.Series([0.0, 0.0], index=['realized_r', 'pnl_amount'])

            calc_df = filtered.apply(calc_mt5_trade, axis=1)
            filtered['realized_r'] = calc_df['realized_r']
            filtered['pnl_amount'] = calc_df['pnl_amount']

        t_naive = filtered['time_metric'].dt.tz_localize(None)
        if period == "monthly":
            filtered['period_obj'] = t_naive.dt.to_period('M')
            format_fn = lambda p: str(p)
        else: # "weekly"
            filtered['period_obj'] = t_naive.dt.to_period('W-SUN')
            format_fn = lambda p: f"{p.start_time.strftime('%b %d')} - {p.end_time.strftime('%b %d')} (W{p.week:02d})"

        periods = sorted(filtered['period_obj'].unique(), reverse=True)
        rows = []
        reward_per_trade = risk_per_trade * reward_multiplier

        for p_obj in periods:
            sub = filtered[filtered['period_obj'] == p_obj]
            tot = len(sub)
            if tot == 0:
                continue

            if mode == "mt5_live" and 'pnl_amount' in sub.columns:
                pnl = float(sub['pnl_amount'].sum())
                net_r = float(sub['realized_r'].sum())
                gross_win_r = float(sub[sub['realized_r'] > 0]['realized_r'].sum())
                gross_loss_r = abs(float(sub[sub['realized_r'] < 0]['realized_r'].sum()))
                pf = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else 999.0
                w = len(sub[sub['realized_r'] > 0])
                l = len(sub[sub['realized_r'] < 0])
                wr = (w / tot * 100.0) if tot > 0 else 0.0
            else:
                w = len(sub[sub['outcome'] == 'SUCCESS'])
                l = tot - w
                wr = (w / tot * 100.0) if tot > 0 else 0.0
                net_r = (w * reward_multiplier) - (l * 1.0)
                pnl = net_r * risk_per_trade
                gross_win = w * reward_multiplier
                gross_loss = l * 1.0
                pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0

            rows.append({
                'Period': format_fn(p_obj),
                'Trades': tot,
                'Wins': w,
                'Losses': l,
                'Win Rate (%)': round(wr, 1),
                'Net R': round(net_r, 2),
                'Profit Factor': round(pf, 2),
                'Net PnL ($)': round(pnl, 2),
                'Return (%)': round((pnl / account_size) * 100.0, 2)
            })

        return pd.DataFrame(rows)

    def generate_telegram_scorecard(
        self,
        period: str = "both", # "weekly", "monthly", or "both"
        risk_per_trade: float = 50.0,
        mode: str = "production",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        account_size: Optional[float] = None
    ) -> str:
        if account_size is None or account_size <= 0:
            account_size = 100000.0 if risk_per_trade >= 500.0 else 10000.0

        policy_label = "Master MT5 Executed Trades" if mode == "mt5_live" else ("Live Telegram Signals" if mode == "telegram_live" else "61.0%+ Live Production (Shielded)")
        msg_parts = []
        msg_parts.append("📊 *ForexAlert AI · PERFORMANCE SCORECARD*")
        msg_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        msg_parts.append(f"⏱️ *Basis:* Trade Close Date (UTC)")
        msg_parts.append(f"🛡️ *Policy:* {policy_label}")
        msg_parts.append(f"💰 *Base Risk:* ${risk_per_trade:,.0f} / trade (Account: ${account_size:,.0f})\n")

        if period in ("monthly", "both"):
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

        if period in ("weekly", "both"):
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


