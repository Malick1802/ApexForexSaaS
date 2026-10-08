# ── SYSTEM PATH INJECTION (CRITICAL FOR AZURE/WINDOWS) ──
import os
import sys
from pathlib import Path

# Get the absolute path of the directory containing this file (dashboard/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Get the absolute path of the project root
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

# Shared design system
from theme import (
    inject_css, get_db, get_engine, get_inference,
    kpi_card, hero_banner, sidebar_logo, sidebar_footer, section_header,
    PROJECT_ROOT, render_system_monitor
)

logger = logging.getLogger(__name__)

# Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Page Config (Main Entry) ────────────────────────────────
st.set_page_config(
    page_title="ApexForex · AI Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

# ── Robust Indicator Fallbacks ──────────────────────────────
def calculate_rsi_manual(prices, period=14):
    """Manual RSI calculation if pandas-ta extension fails."""
    if len(prices) < period:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs.iloc[-1]))


# ── Cached Loaders ─────────────────────────────────────────
def load_engine():
    engine = get_engine()
    return engine

def get_all_active_instruments(engine=None):
    """Return all unblocked trading instruments across Foundation, Confluence, Commodities, and Crypto."""
    from core.confluence_model import load_confluence_config
    from core.symbol_guard import is_symbol_blocked
    base_pairs = []
    if engine:
        try:
            base_pairs = engine.get_all_pairs()
        except Exception:
            pass
    conf_syms = load_confluence_config().get("symbols", [])
    combined = list(dict.fromkeys(base_pairs + conf_syms))
    return [s for s in combined if not is_symbol_blocked(s)]

@st.cache_resource
def load_inference_v2():
    engine = get_inference()
    return engine

def get_training_status():
    import os
    import re
    log_path = os.path.join(PROJECT_ROOT, "logs", "foundation_v2_training.log")
    if not os.path.exists(log_path):
        return None
    try:
        # Only check if log was modified recently (last 6 hours)
        if time.time() - os.path.getmtime(log_path) > 21600:
            return None
            
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-100:]
        
        for line in reversed(lines):
            if "TRAINING COMPLETE" in line:
                return None
            elif "Epoch" in line and "/" in line:
                match = re.search(r'Epoch (\d+)/(\d+)', line)
                if match:
                    return f"Training Epoch {match.group(1)}/{match.group(2)}"
            elif "OOS HOLDOUT EVALUATION" in line:
                return "Evaluating OOS..."
            elif "Building training corpus" in line or "Total sequences across all pairs" in line:
                return "Building Data Corpus..."
            elif "Fetching" in line:
                return "Fetching MT5 Data..."
        return "Training in Background"
    except Exception:
        return None

# ── Chart Renderer (TradingView Lightweight Charts – blink-free) ───
def render_chart(df, symbol, key=None, levels=None):
    if df.empty:
        st.warning("Chart unavailable.")
        return

    import streamlit.components.v1 as components
    import json as _json

    # Prepare candlestick data for Lightweight Charts
    candles = []
    volumes = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp())
        candles.append({
            "time": t,
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
        })
        vol = float(row["volume"]) if "volume" in df.columns else 0
        color = "rgba(0,255,136,0.35)" if row["close"] >= row["open"] else "rgba(255,68,102,0.35)"
        volumes.append({"time": t, "value": vol, "color": color})

    candles_json = _json.dumps(candles)
    volumes_json = _json.dumps(volumes)
    levels_json = _json.dumps(levels) if levels else "null"

    # Determine price precision from symbol
    precision = 3 if "JPY" in symbol else 5
    chart_id = f"tv-chart-{key}" if key else f"tv-chart-{symbol.lower()}"

    html = f"""
    <div id="{chart_id}" style="width:100%;height:460px;border-radius:12px;overflow:hidden;"></div>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const container = document.getElementById('{chart_id}');
        if (!container) return;
        const chart = LightweightCharts.createChart(container, {{
            width: container.offsetWidth,
            height: 460,
            layout: {{
                background: {{ type: 'solid', color: '#0a0e1a' }},
                textColor: '#8b95a8',
                fontFamily: "'Inter', sans-serif",
                fontSize: 11,
            }},
            grid: {{
                vertLines: {{ color: 'rgba(255,255,255,0.03)' }},
                horzLines: {{ color: 'rgba(255,255,255,0.03)' }},
            }},
            rightPriceScale: {{
                borderColor: 'rgba(255,255,255,0.06)',
                scaleMargins: {{ top: 0.05, bottom: 0.25 }},
            }},
            timeScale: {{
                borderColor: 'rgba(255,255,255,0.06)',
                timeVisible: true,
                secondsVisible: false,
                barSpacing: 6,
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{ color: 'rgba(0,229,255,0.25)', width: 1, style: 2, labelBackgroundColor: '#0f1629' }},
                horzLine: {{ color: 'rgba(0,229,255,0.25)', width: 1, style: 2, labelBackgroundColor: '#0f1629' }},
            }},
        }});

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#00FF88',
            downColor: '#FF4466',
            borderUpColor: '#00FF88',
            borderDownColor: '#FF4466',
            wickUpColor: '#00FF88',
            wickDownColor: '#FF4466',
            priceFormat: {{ type: 'price', precision: {precision}, minMove: {10**(-precision)} }},
        }});
        candleSeries.setData({candles_json});

        const volumeSeries = chart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: 'vol',
        }});
        chart.priceScale('vol').applyOptions({{
            scaleMargins: {{ top: 0.82, bottom: 0 }},
        }});
        volumeSeries.setData({volumes_json});

        // Optional price lines for Entry, TP, SL, Day Lines & Swing Lines
        const lvls = {levels_json};
        if (lvls) {{
            if (lvls.upper_day) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.upper_day),
                    color: '#FFA726',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'DAY HIGH ' + parseFloat(lvls.upper_day).toFixed({precision}),
                }});
            }}
            if (lvls.lower_day) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.lower_day),
                    color: '#FFA726',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'DAY LOW ' + parseFloat(lvls.lower_day).toFixed({precision}),
                }});
            }}
            if (lvls.upper_swing) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.upper_swing),
                    color: '#AB47BC',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: true,
                    title: 'SWING HIGH ' + parseFloat(lvls.upper_swing).toFixed({precision}),
                }});
            }}
            if (lvls.lower_swing) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.lower_swing),
                    color: '#AB47BC',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: true,
                    title: 'SWING LOW ' + parseFloat(lvls.lower_swing).toFixed({precision}),
                }});
            }}
            if (lvls.entry) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.entry),
                    color: '#00E5FF',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    axisLabelVisible: true,
                    title: 'ENTRY ' + parseFloat(lvls.entry).toFixed({precision}),
                }});
            }}
            if (lvls.tp) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.tp),
                    color: '#00FF88',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'TP ' + parseFloat(lvls.tp).toFixed({precision}),
                }});
            }}
            if (lvls.sl) {{
                candleSeries.createPriceLine({{
                    price: parseFloat(lvls.sl),
                    color: '#FF4466',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'SL ' + parseFloat(lvls.sl).toFixed({precision}),
                }});
            }}
        }}

        chart.timeScale().fitContent();

        // Responsive resize
        const ro = new ResizeObserver(() => {{
            chart.applyOptions({{ width: container.offsetWidth }});
        }});
        ro.observe(container);
    }})();
    </script>
    """
    components.html(html, height=470, scrolling=False)


# =============================================================================
# VIEW 1: Command Center (Home)
# =============================================================================
def show_command_center():
    engine = load_engine()
    db = get_db()

    hero_banner("Command Center",
                "Real-time AI surveillance across 31 assets (Forex & Commodities) · Institutional-level precision targeting",
                show_status=True)

    # ── Active AI Engine Model Strip ─────────────────────────────
    _cfg_path = PROJECT_ROOT / "config.yaml"
    _cur_ver = "v3"
    _cur_mode = "truth"
    if _cfg_path.exists():
        try:
            with open(_cfg_path, 'r', encoding='utf-8') as _f:
                _ycfg = yaml.safe_load(_f) or {}
            _cur_ver = _ycfg.get("foundation", {}).get("active_version", "v3")
            _cur_mode = _ycfg.get("fleet", {}).get("routing_mode", "truth")
        except Exception:
            pass
    _ver_titles = {
        "v3": "Foundation Brain v3 (57-Feature TFT · 30 Pairs + Macro)",
        "v1": "Foundation Brain v1 (34-Feature TFT · 833k Samples)",
        "v2": "Foundation Brain v2 (Extended 29-Pair TFT)",
    }
    _curr_title = _ver_titles.get(_cur_ver, f"Foundation Brain {_cur_ver.upper()}")
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;padding:10px 18px;margin-bottom:18px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:10px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.15rem;">🧠</span>
            <div>
                <span style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-tertiary);font-weight:600;">Active AI Engine:</span>
                <span style="font-size:0.88rem;font-weight:700;color:var(--text-primary);margin-left:6px;">{_curr_title}</span>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:0.75rem;padding:3px 9px;border-radius:12px;background:rgba(0,230,118,0.1);color:#00e676;border:1px solid rgba(0,230,118,0.25);font-weight:600;">Routing: {_cur_mode.upper()}</span>
            <span style="font-size:0.75rem;padding:3px 9px;border-radius:12px;background:rgba(41,182,246,0.1);color:#29b6f6;border:1px solid rgba(41,182,246,0.25);font-weight:600;">Conviction: 61%+</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Onboarding Banner for Subscribers ───────────────────────
    _u_email = st.session_state.get("user_email", "")
    _u_role = st.session_state.get("user_role", "subscriber")
    if _u_email and _u_role != "admin":
        try:
            from core.user_accounts import get_user_by_email
            _u_acc = get_user_by_email(_u_email)
            if not _u_acc or not _u_acc.get("mt5_login"):
                with st.container(border=True):
                    ob_c1, ob_c2 = st.columns([3.5, 1.2])
                    with ob_c1:
                        st.markdown("""
                        <div style="display:flex;align-items:center;gap:14px;">
                            <span style="font-size:2rem;">🚀</span>
                            <div>
                                <div style="font-weight:700;font-size:1.05rem;color:var(--text-primary);">
                                    Welcome to ApexForex! Connect your broker to start automated copy trading.
                                </div>
                                <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:3px;">
                                    Your 14-day free trial is active. Enter your MT5 credentials in the Copy Trading Hub to mirror institutional signals automatically.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with ob_c2:
                        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
                        if st.button("👉 Set Up Copy Trading", type="primary", use_container_width=True, key="ob_btn_setup"):
                            st.session_state["nav_target"] = "copy_trading"
                            st.rerun()
                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        except Exception:
            pass

    all_pairs = get_all_active_instruments(engine)
    
    # 1. Active Signals Pool
    # We fetch ALL signals marked as ACTIVE (Success/Fail/Wait intent)
    # including hidden/shadow signals so we can filter/show them correctly.
    raw_active = db.get_active_signals(include_hidden=True)
    
    from core.symbol_guard import is_commodity
    # Filter: Show real live trade signals (BUY/SELL >= 61% for Forex, >= 55% for Commodities, or has active MT5 ticket)
    active_signals = [
        s for s in raw_active 
        if s.get('signal') in ['BUY', 'SELL'] and not bool(s.get('is_hidden', 0)) and (
            float(s.get('confidence') or 0) >= (0.55 if is_commodity(s.get('symbol', '')) else 0.61)
            or s.get('mt5_ticket') is not None
        )
    ]
    active_count = len(active_signals)

    # 2. Expired/Closed Signals (Current Week Window - UTC aligned)
    recent = db.get_recent_signals(limit=5000, include_hidden=True)
    expired_signals = []
    success_rate = 0.0
    completed_count = 0
    w_count = 0
    l_count = 0

    if recent:
        # Time Window: Current Week (Monday 00:00 UTC to Now)
        now_utc = datetime.now(timezone.utc)
        start_of_week = (now_utc - timedelta(days=now_utc.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = start_of_week
        
        # Filter recent signals by time (using UTC-aware timestamps to avoid offset-naive/aware TypeError)
        recent_window = []
        for s in recent:
            if s.get('signal') in ['WAIT', 'HEARTBEAT']:
                continue
            try:
                s_time = pd.to_datetime(s['timestamp'], utc=True)
                if s_time >= cutoff_time:
                    recent_window.append(s)
            except Exception:
                continue
        
        df_sig = pd.DataFrame(recent_window)
        if not df_sig.empty:
            if 'outcome' not in df_sig.columns:
                df_sig['outcome'] = 'ACTIVE'
            
            # Expired/Closed for display: Real live non-hidden signals that closed
            expired_signals = [
                s for s in recent_window 
                if s.get('outcome') != 'ACTIVE' 
                and not bool(s.get('is_hidden', 0))
                and s.get('signal') in ['BUY', 'SELL']
            ]
            
            # Closed for KPI = Real live trades that hit TP or SL (SUCCESS or FAIL)
            completed = df_sig[
                (df_sig['outcome'].isin(['SUCCESS', 'FAIL'])) &
                (df_sig['is_hidden'] == 0)
            ]
            completed_count = len(completed)
            w_count = len(completed[completed['outcome'] == 'SUCCESS'])
            l_count = len(completed[completed['outcome'] == 'FAIL'])
            
            if not completed.empty:
                success_rate = (w_count / completed_count) * 100

    # Fetch Real Live Win Rate (ALL non-shadow trades, not just is_proven)
    live_stats = db.get_live_win_rate()
    live_win_rate = live_stats.get('win_rate', 0.0)
    live_total = live_stats.get('total', 0)
    live_wins = live_stats.get('wins', 0)
    live_losses = live_stats.get('losses', 0)

    # Also fetch certified (is_proven) rate for the secondary label
    val_stats = db.get_validated_win_rate()
    val_win_rate = val_stats.get('win_rate', 0.0)
    val_total = val_stats.get('total', 0)

    _tok = st.session_state.get("_session_token", "") or st.query_params.get("t", "")
    _tok_suffix = f"&t={_tok}" if _tok else ""
    _tok_prefix = f"?t={_tok}" if _tok else ""

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Monitored Pairs", len(all_pairs), "Majors · Minors · Crosses · Commodities", "accent-cyan", link_url=f"/market{_tok_prefix}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Active Signals", active_count, "Running trades", "accent-gold", link_url=f"/analytics?filter=active{_tok_suffix}"), unsafe_allow_html=True)
    with c3:
        # AI Model Live Win Rate
        wr_color = "accent-green" if live_win_rate >= 60 else "accent-gold" if live_win_rate >= 50 else "accent-red"
        st.markdown(kpi_card("AI Win Rate", f"{live_win_rate:.1f}%", f"{live_wins}W · {live_losses}L · {live_total} certified", wr_color, link_url=f"/analytics?filter=all{_tok_suffix}"), unsafe_allow_html=True)
    with c4:
        closed_delta = f"{w_count}W · {l_count}L (TP/SL Hit)" if completed_count > 0 else "Hit TP or SL"
        st.markdown(kpi_card("Closed Trades (Week)", completed_count, closed_delta, "accent-cyan", link_url=f"/analytics?filter=closed{_tok_suffix}"), unsafe_allow_html=True)
    with c5:
        training_status = get_training_status()
        if training_status:
            st.markdown(kpi_card("System Health", "Training v2", training_status, "accent-gold", link_url=f"/audit{_tok_prefix}"), unsafe_allow_html=True)
        else:
            st.markdown(kpi_card("System Health", "Online", "Watchdog · Sentinel · API", "accent-cyan", link_url=f"/audit{_tok_prefix}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🎯", "High Confidence Opportunities")

    if active_signals:
        # Sort by confidence descending
        # Convert list of dicts to DF for display
        df_active = pd.DataFrame(active_signals)
        # Filter for display (optional, depending on what we want to show)
        # Assuming all active signals are "opportunities"
        
        if not df_active.empty:
            cols_to_show = ['symbol', 'signal', 'confidence', 'price_at_signal', 'timestamp']
            if 'mt5_ticket' in df_active.columns:
                df_active['ticket_disp'] = df_active['mt5_ticket'].apply(lambda x: f"#{int(x)}" if pd.notnull(x) and x else "—")
                cols_to_show = ['symbol', 'signal', 'confidence', 'price_at_signal', 'ticket_disp', 'timestamp']
                df_display = df_active[cols_to_show].copy()
                df_display.columns = ['Pair', 'Direction', 'Confidence', 'Entry', 'MT5 Ticket', 'Detected']
            else:
                df_display = df_active[cols_to_show].copy()
                df_display.columns = ['Pair', 'Direction', 'Confidence', 'Entry', 'Detected']
            
            # Formatting
            df_display['Confidence'] = df_display['Confidence'].apply(lambda x: float(x) * 100)
            
            # Render styled table
            event = st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="signal_table_active",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            if event.selection.rows:
                try:
                    selected_idx = event.selection.rows[0]
                    symbol = df_display.iloc[selected_idx]['Pair']
                    st.session_state['pair_selector'] = symbol
                    # Navigate to Trading Terminal via router
                    st.session_state['nav_target'] = 'terminal'
                    st.rerun()
                except Exception as e:
                    st.error(f"Navigation failed: {e}")
            
    # Fallback to empty if no signals
    if not active_signals:
        st.markdown("""
        <div class="glass-card" style="padding: 26px; text-align: center; border: 1px dashed rgba(255,255,255,0.14); margin-top: 10px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📡</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: var(--text-primary); margin-bottom: 4px;">Market Surveillance Active</div>
            <div style="color: var(--text-secondary); font-size: 0.85rem; max-width: 540px; margin: 0 auto; line-height: 1.6;">
                Apex Neural Engines are continuously monitoring 29 FX pairs and commodities. High-conviction trade opportunities will populate here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Closed Trades View
    st.markdown("<br>", unsafe_allow_html=True)
    expander_title = f"📜 Closed Trade History ({completed_count} Closed This Week)" if completed_count > 0 else "📜 Closed Trade History (Click to View)"
    with st.expander(expander_title, expanded=False):
        if expired_signals:
            df_hist = pd.DataFrame(expired_signals)
            # Format MT5 ticket if present
            if 'mt5_ticket' in df_hist.columns:
                df_hist['ticket_disp'] = df_hist['mt5_ticket'].apply(lambda x: f"#{int(x)}" if pd.notnull(x) and x else "—")
                cols = ['symbol', 'signal', 'outcome', 'confidence', 'price_at_signal', 'ticket_disp', 'timestamp']
            else:
                cols = ['symbol', 'signal', 'outcome', 'confidence', 'price_at_signal', 'timestamp']
            if 'confidence' in df_hist.columns:
                df_hist['confidence'] = df_hist['confidence'].apply(lambda x: float(x or 0) * 100 if float(x or 0) <= 1.0 else float(x or 0))
            show_cols = [c for c in cols if c in df_hist.columns]
            
            col_cfg = {
                "symbol": "Pair",
                "signal": "Direction",
                "outcome": "Result",
                "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
                "price_at_signal": st.column_config.NumberColumn("Entry", format="%.5f"),
                "timestamp": "Detected"
            }
            if 'ticket_disp' in df_hist.columns:
                col_cfg["ticket_disp"] = "MT5 Ticket"

            st.dataframe(
                df_hist[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg
            )
        else:
            st.info("No expired or closed trades found in recent history.")


# =============================================================================
# VIEW 2: Market Overview
# =============================================================================
def show_market_overview():
    import yaml

    hero_banner("Market Overview", "Real-time AI signal grid across 31 global currency pairs")

    # RANGING-approved whitelist (must match core/inference.py)
    RANGING_APPROVED = {'EURAUD', 'AUDNZD', 'GBPUSD', 'XAUUSD', 'USOIL.cash', 'USDJPY', 'EURNZD', 'USDSGD'}

    db = get_db()

    try:
        config_path = PROJECT_ROOT / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config_pairs = config.get('currency_pairs', {})
    except:
        config_pairs = {}

    @st.fragment(run_every=timedelta(seconds=30))
    def _market_overview_pulse():
        # Build signal groups (Symbol -> List) to handle Tier overlapping
        active_signals = db.get_active_signals(include_hidden=True)
        sig_map = {}
        has_secondary_tier = {}

        # Resolve certified / validated symbols from the performance matrix
        from core.performance_gate import get_performance_gate
        gate = get_performance_gate()
        try:
            gate.recompute_from_db(lookback_days=14)
        except:
            pass
            
        certified_symbols = set()
        if gate and gate.performance_matrix:
            for sym, contents in gate.performance_matrix.items():
                if isinstance(contents, dict):
                    for d, tiers in contents.items():
                        if isinstance(tiers, dict):
                            for t_str, data in tiers.items():
                                if isinstance(data, dict) and data.get('status') == 'APPROVED':
                                    certified_symbols.add(sym)

        # Group by symbol
        groups = {}
        for s in active_signals:
            sym = s['symbol']
            if sym not in groups: groups[sym] = []
            groups[sym].append(s)

        # Select the 'Best' signal for tile display (Priority: Live > Highest Tier > Newest)
        # NOTE: This loop must be OUTSIDE the group-building loop above
        for sym, sigs in groups.items():
            # Filter for real trades (Exclude WAIT/0% from being counted as 'secondary')
            real_sigs = [s for s in sigs if s.get('signal') in ('BUY', 'SELL') and int(s.get('confidence_tier') or 0) > 0]
            
            sorted_sigs = sorted(
                sigs, 
                key=lambda x: (not bool(x.get('is_hidden', 0)), x.get('confidence_tier', 0), x['timestamp']),
                reverse=True
            )
            sig_map[sym] = sorted_sigs[0]
            # Only show indicator if there are multiple REAL trade setups
            has_secondary_tier[sym] = len(real_sigs) > 1

        # Fallback for symbols with only historical signals (no active ones)
        # Use a large limit and exclude SYSTEM heartbeats to ensure all pairs are covered
        recent = db.get_recent_signals(limit=500, include_hidden=True)
        for s in recent:
            sym = s['symbol']
            if sym == 'SYSTEM':
                continue  # Skip heartbeat rows — they pollute the lookup table
            if sym not in sig_map:
                sig_map[sym] = s

        # Signal grid categories
        from core.symbol_guard import is_symbol_blocked, is_commodity, is_crypto
        active_commodities = [p for p in config_pairs.get('commodities', []) if not is_symbol_blocked(p.get('symbol', ''))]
        active_crypto = [p for p in config_pairs.get('crypto', []) if not is_symbol_blocked(p.get('symbol', ''))]
        minors_forex = [p for p in config_pairs.get('minors', []) if not is_commodity(p.get('symbol', '')) and not is_crypto(p.get('symbol', ''))]

        categories = {
            "⚡ Majors": config_pairs.get('majors', []),
            "🔷 Minors": minors_forex,
            "🔶 Crosses": config_pairs.get('crosses', []),
        }
        if active_commodities:
            categories["🏆 Commodities & Metals"] = active_commodities
        if active_crypto:
            categories["🪙 Crypto (24/7)"] = active_crypto

        for cat_name, pair_list in categories.items():
            if not pair_list: continue
            st.markdown(f'<div class="section-header"><span class="section-header-text">{cat_name}</span></div>', unsafe_allow_html=True)
            cols = st.columns(3)
            symbols = [p['symbol'] for p in pair_list]
            for i, symbol in enumerate(symbols):
                sig_data = sig_map.get(symbol)
                with cols[i % 3]:
                    _tok = st.session_state.get("_session_token", "") or st.query_params.get("t", "")
                    _tok_param = f"&t={_tok}" if _tok else ""
                    link = f"/terminal?symbol={symbol}{_tok_param}"
                    
                    is_ranging_regime = symbol in RANGING_APPROVED
                    pair_regime = "RANGING" if is_ranging_regime else "TRENDING"
                    is_validated = symbol in certified_symbols

                    if is_validated:
                        badge_bg = "rgba(0, 229, 255, 0.12)"
                        badge_border = "rgba(0, 229, 255, 0.3)"
                        badge_color = "#00E5FF"
                        badge_text = "🛡️ CERTIFIED"
                    else:
                        badge_bg = "rgba(255, 255, 255, 0.04)"
                        badge_border = "rgba(255, 255, 255, 0.1)"
                        badge_color = "var(--text-muted)"
                        badge_text = "⚠️ SHADOW"

                    validation_badge_html = f'''
                    <div style="
                        margin-top: 8px;
                        font-size: 0.65rem;
                        font-weight: 700;
                        letter-spacing: 0.05em;
                        font-family: var(--font-mono);
                        color: {badge_color};
                        background: {badge_bg};
                        border: 1px solid {badge_border};
                        padding: 3px 8px;
                        border-radius: 6px;
                        display: inline-block;
                    ">
                        {badge_text}
                    </div>
                    '''

                    if not sig_data:
                        tile_html = (
                            f'<div class="signal-tile tile-wait" style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 155px; margin-bottom: 0px; padding: 16px 14px 12px 14px;">'
                            f'<div class="tile-symbol" style="margin-top: 6px;">{symbol} <span class="tile-arrow" style="font-size: 0.75rem; opacity: 0.35; transition: all 0.2s ease;">&#x2197;</span></div>'
                            f'<div class="tile-signal tile-signal-wait" style="margin: 4px 0;">—</div>'
                            f'<div class="tile-conf">Awaiting Data</div>'
                            f'{validation_badge_html}'
                            f'</div>'
                        )
                    else:
                        sig = sig_data.get('signal', 'WAIT')
                        conf = sig_data.get('confidence', 0)
                        outcome = sig_data.get('outcome', 'ACTIVE')
                        regime = sig_data.get('regime') or ''

                        regime_badge = ""
                        r_upper = str(regime).upper()
                        is_crisis = "CRISIS" in r_upper or "VOLATILE" in r_upper
                        
                        sig_is_hidden = bool(sig_data.get('is_hidden', False))
                        is_cert = symbol in certified_symbols
                        # A signal state is LIVE if certified AND either we are not in an active trade (WAIT) OR the trade is visible
                        is_live_badge = is_cert and (sig == 'WAIT' or not sig_is_hidden)

                        if regime:
                            if is_crisis:
                                regime_badge = '<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: #FF4466; background: rgba(255,68,102,0.15); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em; border: 1px solid rgba(255,68,102,0.3); box-shadow: 0 0 10px rgba(255,68,102,0.2);">⚡ CRISIS</div>'
                            elif "TRENDING" in r_upper:
                                _badge_title = "TRENDING ⭐ LIVE" if is_live_badge else "TRENDING · SHADOW"
                                _color = "#00FF88" if is_live_badge else "#aaa"
                                _bg = "rgba(0,255,136,0.1)" if is_live_badge else "rgba(255,255,255,0.05)"
                                _border = "rgba(0,255,136,0.2)" if is_live_badge else "rgba(255,255,255,0.1)"
                                regime_badge = f'<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: {_color}; background: {_bg}; padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em; border: 1px solid {_border};">{_badge_title}</div>'
                            elif "RANGING" in r_upper:
                                _badge_title = "RANGING ⭐ LIVE" if is_live_badge else "RANGING · SHADOW"
                                _color = "#00E5FF" if is_live_badge else "#aaa"
                                _bg = "rgba(0,229,255,0.1)" if is_live_badge else "rgba(255,255,255,0.05)"
                                _border = "rgba(0,229,255,0.2)" if is_live_badge else "rgba(255,255,255,0.1)"
                                regime_badge = f'<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: {_color}; background: {_bg}; padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em; border: 1px solid {_border};">{_badge_title}</div>'

                        display_sig = sig
                        css_tile = "tile-wait"
                        css_signal = "tile-signal-wait"
                        conf_display = "Monitoring..."
                        conf_bar = ""
                        
                        is_hidden = bool(sig_data.get('is_hidden', False))

                        extra_styles = ""
                        if is_crisis:
                            display_sig = "SAFE"
                            css_tile = "tile-wait"
                            css_signal = "tile-signal-wait"
                            # Use wait_prob or calibrated conf
                            f_conf = sig_data.get('wait_prob', conf)
                            conf_display = f"{(f_conf or 0.0):.0%}" if (f_conf or 0.0) > 0 else "Blocked"
                            # Force red border for crisis tiles
                            extra_styles = "border: 1px solid rgba(255,68,102,0.4); background: rgba(255,68,102,0.03); box-shadow: inset 0 0 20px rgba(255,68,102,0.05);"
                        elif outcome == 'ACTIVE' and not is_hidden and float(conf or 0) >= 0.61:
                            if sig == "BUY":
                                css_tile = "tile-buy"
                                css_signal = "tile-signal-buy"
                                conf_display = f"{conf:.0%}"
                                conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar conf-bar-buy" style="width: {conf:.1%}"></div></div>'
                            elif sig == "SELL":
                                css_tile = "tile-sell"
                                css_signal = "tile-signal-sell"
                                conf_display = f"{conf:.0%}"
                                conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar conf-bar-sell" style="width: {conf:.1%}"></div></div>'
                        else:
                            display_sig = "WAIT"
                            conf_display = f"{(sig_data.get('wait_prob', conf) or 0.0):.0%}" if (sig_data.get('wait_prob', conf) or 0.0) > 0 else "Monitoring..."

                        ghost_html = '<div class="ghost-indicator" title="Secondary Tier Active"></div>' if has_secondary_tier.get(symbol) else ""
                        tile_html = (
                            f'<div class="signal-tile {css_tile}" '
                            f'style="position: relative; {"opacity: 0.85;" if is_hidden else ""} {extra_styles} display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 155px; margin-bottom: 0px; padding: 16px 14px 12px 14px;">'
                            f'{regime_badge}'
                            f'{ghost_html}'
                            f'<div class="tile-symbol" style="margin-top: 10px;">{symbol} <span class="tile-arrow" style="font-size: 0.75rem; opacity: 0.35; transition: all 0.2s ease;">&#x2197;</span></div>'
                            f'<div class="tile-signal {css_signal}" style="margin: 3px 0;">{display_sig}</div>'
                            f'<div class="tile-conf">{conf_display}</div>'
                            f'{conf_bar}'
                            f'{validation_badge_html}'
                            f'</div>'
                        )

                    # Render the directly clickable glassmorphic card
                    card_link_html = f'<a href="{link}" target="_self" class="signal-tile-link" title="Open {symbol} in Trading Terminal">{tile_html}</a>'
                    st.markdown(card_link_html, unsafe_allow_html=True)

    # Initial Pulse Trigger
    _market_overview_pulse()

    # Sidebar filters
    with st.sidebar:
        section_header("🎛️", "Filters")

        # Use session state to persist filters across pages
        if 'accuracy_target' not in st.session_state:
            st.session_state['accuracy_target'] = '90%'
        
        accuracy_target = st.select_slider('Desired Accuracy',
            options=['60%', '70%', '80%', '90%', 'Apex'],
            key='accuracy_target')

        if 'confidence_thresh' not in st.session_state:
            st.session_state['confidence_thresh'] = 70
            
        confidence_thresh = st.slider("Confidence Filter", 50, 95, 
            key='confidence_thresh')

        st.caption(f"**{accuracy_target}** accuracy · **{confidence_thresh}%** min confidence")

    # --- REMOVED REDUNDANT OUTER LOOP ---


# =============================================================================
# VIEW 3: Trading Terminal
# =============================================================================
def show_trading_terminal():
    engine = load_engine()
    inf_engine = load_inference_v2()
    db = get_db()

    # Get trading status from config for UI labels
    is_actively_trading = inf_engine.config.get('trading', {}).get('execute_trades', False)

    with st.sidebar:
        section_header("🎛️", "Analysis Controls")
        all_pairs = get_all_active_instruments(engine)

        # Check for navigation from Market Overview
        qp = st.query_params
        if "symbol" in qp:
            target_sym = qp["symbol"]
            if target_sym in all_pairs:
                st.session_state['pair_selector'] = target_sym

        if 'pair_selector' not in st.session_state:
            st.session_state['pair_selector'] = "EURUSD" if "EURUSD" in all_pairs else all_pairs[0]

        symbol = st.selectbox("Select Pair", all_pairs, key='pair_selector')
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
        st.divider()

        if 'accuracy_target' not in st.session_state:
            st.session_state['accuracy_target'] = '70%'
        if 'confidence_thresh' not in st.session_state:
            st.session_state['confidence_thresh'] = 70

        st.select_slider('Desired Accuracy',
            options=['60%', '70%', '80%', '90%', 'Apex'],
            key='accuracy_target')

        accuracy_target = st.session_state['accuracy_target']
        tier_labels = {'60%': '⚡ Aggressive', '70%': '🚀 Growth',
                       '80%': '💎 Precision', '90%': '🏆 Expert', 'Apex': '👑 Institutional'}
        st.caption(f"**{tier_labels.get(accuracy_target, '')}**")
        st.divider()

        st.slider("Confidence Filter", 50, 95, key='confidence_thresh')
        confidence_thresh = st.session_state['confidence_thresh']

    # ── Trading Engine Selector ───────────────────────────────────────────────
    if "terminal_active_strategy" not in st.session_state:
        st.session_state["terminal_active_strategy"] = "🎯 M15 Wick Sniper (Discretionary)"

    c_strat1, c_strat2 = st.columns([3.4, 1.1])
    with c_strat1:
        strategy_options = [
            "🌟 Dynamic YTD Model (Daily Winning Asset Strategy)",
            "🧠 Confluence ML M15 (P60 · 60% Partial + BE+2p)",
            "⚡ Confluence Standard M15 (P25 · 25% Partial + BE+2p)",
            "🧠 Confluence ML M15 (Original AI Gate · Fixed 1.5R)",
            "⚡ Confluence Standard M15 (Original Rule-Based · Fixed 1.5R)",
            "🎯 M15 Wick Sniper (Discretionary)",
            "🤖 AI Automated Scanner (TFT Neural Network)",
        ]
        curr_strat = st.session_state.get("terminal_active_strategy", strategy_options[0])
        # Backwards compatibility check
        if curr_strat == "🧠 Confluence M15 + Deep Learning (AI Gate)":
            curr_strat = strategy_options[1]
        elif curr_strat == "⚡ Confluence M15 Standard (Rule-Based)":
            curr_strat = strategy_options[2]

        idx = strategy_options.index(curr_strat) if curr_strat in strategy_options else 0
        active_strategy = st.radio(
            "Select Trading Strategy Engine",
            strategy_options,
            index=idx,
            horizontal=True,
            key="terminal_active_strategy",
            label_visibility="collapsed"
        )
    with c_strat2:
        if "Dynamic" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(255,214,0,0.15);border:1px solid #ffd60066;color:#ffd600;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">🌟 DYNAMIC YTD · WINNING ASSETS</span></div>',
                unsafe_allow_html=True
            )
        elif "P60" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(168,85,247,0.15);border:1px solid #a855f755;color:#c084fc;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">🧠 AI GATE · P60 & BE+2p</span></div>',
                unsafe_allow_html=True
            )
        elif "P25" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(0,230,118,0.15);border:1px solid #00e67655;color:#00e676;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">⚡ STANDARD · P25 & BE+2p</span></div>',
                unsafe_allow_html=True
            )
        elif "Original AI Gate" in active_strategy or "Deep Learning" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(168,85,247,0.15);border:1px solid #a855f755;color:#c084fc;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">🧠 ORIGINAL AI GATE · FIXED 1.5R</span></div>',
                unsafe_allow_html=True
            )
        elif "Confluence" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(255,214,0,0.12);border:1px solid #ffd60044;color:#ffd600;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">⚡ ORIGINAL STANDARD · FIXED 1.5R</span></div>',
                unsafe_allow_html=True
            )
        elif "M15" in active_strategy:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(0,230,118,0.12);border:1px solid #00e67644;color:#00e676;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">🟢 PENDING STOP ENGINE</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="text-align:right;padding-top:4px;"><span style="background:rgba(41,182,246,0.12);border:1px solid #29b6f644;color:#29b6f6;font-size:0.75rem;padding:4px 10px;border-radius:12px;font-weight:700;">🧠 TFT NEURAL SCANNER</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    if "Dynamic" in active_strategy:
        _show_dynamic_ytd_cockpit(symbol, all_pairs)
        return

    if "Confluence" in active_strategy:
        if "P60" in active_strategy:
            m_key = "confluence_ml_p60"
        elif "P25" in active_strategy:
            m_key = "confluence_std_p25"
        elif "Original AI Gate" in active_strategy:
            m_key = "confluence_ml_m15"
        else:
            m_key = "confluence_m15"
        _show_confluence_cockpit(symbol, all_pairs, active_model_key=m_key)
        return

    if "M15" in active_strategy:
        _show_manual_m15_cockpit(symbol, all_pairs)
        return

    # ── Live Data Fragment (reruns every 10s WITHOUT full page blink) ──
    @st.fragment(run_every=timedelta(seconds=10))
    def _live_terminal_data():
        result = None
        pred = "WAIT"
        conf = 0.0
        df = pd.DataFrame()
        is_on_cooldown = False
        cooldown_remaining_min = 0.0
        locked_trade = None
        is_market_closed = False

        col_main, col_side = st.columns([3, 1])

        with col_main:
            try:
                # Add a pulsing heartbeat to indicate scanning is active
                st.markdown(f"""
                <div style="background: rgba(0,255,136,0.05); padding: 5px 15px; border-radius: 20px; border: 1px solid rgba(0,255,136,0.1); display: inline-flex; align-items: center; gap: 8px; margin-bottom: 20px;">
                    <div style="width: 8px; height: 8px; background: #00FF88; border-radius: 50%; box-shadow: 0 0 10px #00FF88;"></div>
                    <span style="font-size: 0.7rem; font-family: var(--font-mono); color: #00FF88; letter-spacing: 0.1em;">LIVE ANALYSIS PULSE: {datetime.now().strftime('%H:%M:%S')}</span>
                </div>
                """, unsafe_allow_html=True)

                # 0. Fetch Data (REAL TIME SYNC)
                df = inf_engine.data_engine.fetch(symbol, interval=timeframe, days=7, use_cache=False)
                if df.empty:
                    raise Exception(f"No candlestick data received for {symbol}")

                # 1. Check for EXISTING ACTIVE SIGNAL (to manage PnL and Locked state)
                # IMPORTANT: include_hidden=False ensures 50% shadow trades NEVER lock a terminal
                active_signals = db.get_active_signals(symbol=symbol, include_hidden=False)
                locked_trade = active_signals[0] if active_signals else None
                
                # 2. ALWAYS Run FRESH INFERENCE for Live Pulse (Background stats)
                # For the UI pulse, we are more lenient with 'allow_stale' to keep decimals moving
                # but we FORCE use_cache=False to ensure we aren't stuck on a stale parquet file.
                live_result = inf_engine.predict_symbol(
                    symbol, save_to_db=False, 
                    win_rate=st.session_state['accuracy_target'], 
                    allow_stale=True,
                    use_cache=False
                )

                # ── UI READ-ONLY SYNC (Execution handled exclusively by background Executive daemon) ──
                pass

                # 3. DIRECTIONAL / COMMODITY BLACKLIST & LOCKING LOGIC
                from core.symbol_guard import is_symbol_blocked, is_direction_blocked
                is_sym_blocked = is_symbol_blocked(symbol)
                is_buy_blocked = is_direction_blocked(symbol, 'BUY')
                is_sell_blocked = is_direction_blocked(symbol, 'SELL')

                if locked_trade:
                    result = locked_trade
                    st.caption(f"🔒 TERMINAL LOCKED TO ACTIVE POSITION (ID #{locked_trade['id']})")
                else:
                    result = live_result
                    if is_sym_blocked:
                        st.caption(f"🛑 COMMODITY SHIELD · {symbol} is blacklisted from live execution")
                    elif is_buy_blocked and is_sell_blocked:
                        st.caption(f"🚫 DIRECTIONAL BLACKLIST · {symbol} is blacklisted from live execution")
                    elif is_buy_blocked:
                        st.caption(f"🚫 DIRECTIONAL BLACKLIST · {symbol} BUY is permanently blacklisted (SELL enabled)")
                    elif is_sell_blocked:
                        st.caption(f"🚫 DIRECTIONAL BLACKLIST · {symbol} SELL is permanently blacklisted (BUY enabled)")
                    elif live_result:
                        st.caption("📡 LIVE AI PULSE (Real-Time Monitoring)")
                    else:
                        st.caption("⚠️ AI PULSE OFFLINE (Awaiting Market Data)")

                # 4. FALLBACK: If inference failed and no locked trade, show most recent DB signal
                if not result:
                    # Fallback: show last non-shadow signal for this symbol
                    sym_signals = db.get_recent_signals(symbol=symbol, limit=1, include_hidden=False)
                    if sym_signals:
                        result = sym_signals[0]
                        st.caption(f"📋 Showing last recorded signal · {result.get('outcome', 'UNKNOWN')}")

                if result:
                    pred = result.get('signal', 'WAIT')
                    conf = result.get('confidence', 0)
                
                # Removed raw JSON dump to clean UI
                pass
            except Exception as e:
                logger.error(f"Inference error: {e}")
                st.error(f"⚠️ API Rate Limit or Network Error: {e}")

            is_market_closed = False
            if not df.empty:
                # Check for "Market Closed" (Stale Data > 4h)
                try:
                    last_ts = df.index[-1]
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.tz_localize('UTC')
                    else:
                        last_ts = last_ts.tz_convert('UTC')
                    
                    diff_hours = (pd.Timestamp.now(tz='UTC') - last_ts).total_seconds() / 3600.0
                    # Use 55h threshold so weekend gaps (Fri 22:00 -> Sun 22:00 = ~48h) don't false-trigger
                    if diff_hours > 55.0:
                        is_market_closed = True
                        st.warning(f"⛔ MARKET CLOSED · Displaying analysis from last close ({last_ts.strftime('%d %b %H:%M UTC')})")
                    elif diff_hours > 4.0:
                        # Weekend gap — market just reopened or about to open
                        st.info(f"⏳ Weekend · Market reopening · Last candle: {last_ts.strftime('%d %b %H:%M UTC')} · First new candle forming soon")
                except:
                    pass

                if len(df) >= 2:
                    try:
                        last_price = df['close'].iloc[-1]
                        prev_price = df['close'].iloc[-2]
                        change = (last_price - prev_price) / prev_price

                        # RSI — always use manual calculation as safe baseline (no pandas-ta hard dependency)
                        current_rsi = calculate_rsi_manual(df['close'])
                        try:
                            import pandas_ta as ta
                            if hasattr(df, 'ta') and hasattr(df.ta, 'rsi'):
                                rsi_series = df.ta.rsi(length=14)
                                if rsi_series is not None and not rsi_series.empty:
                                    rsi_val = float(rsi_series.iloc[-1])
                                    if not np.isnan(rsi_val):
                                        current_rsi = rsi_val
                        except Exception:
                            pass  # Keep manual RSI fallback from above

                        volatility = df['close'].pct_change().std() * 100
                        volatility = float(volatility) if not np.isnan(float(volatility)) else 0.0

                        # Calculate real-time PnL if active trade
                        pnl_html = ""
                        if locked_trade and locked_trade.get('signal') in ('BUY', 'SELL'):
                            try:
                                entry_price = float(locked_trade['price_at_signal'])
                                direction = 1 if locked_trade.get('signal') == 'BUY' else -1
                                # ── Pip Size Resolution ─────────────────────────────────────
                                is_commodity = any(x in symbol.upper() for x in ['XAU', 'GOLD', 'XAG', 'SILVER', 'OIL', 'WTI', 'BRENT'])
                                pip_size = 0.01 if (is_commodity or 'JPY' in symbol.upper()) else 0.0001
                                pnl_pips = (last_price - entry_price) / pip_size * direction
                                
                                pnl_color = "#00FF88" if pnl_pips >= 0 else "#FF4466"
                                pnl_html = f'<span style="font-family: monospace; font-size: 1.1rem; font-weight: 700; color: {pnl_color}; margin-left: 16px; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.05);">{pnl_pips:+.1f} pips</span>'
                            except:
                                pass

                        # High-Visibility LOCKED Badge
                        lock_html = ""
                        if locked_trade:
                            lock_html = f"""
<span style="background: #00E5FF; color: #000; padding: 4px 12px; border-radius: 6px; font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 900; letter-spacing: 0.05em; vertical-align: middle; margin-right: 15px; box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);">LOCKED</span>
"""

                        st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
{lock_html}
<div>
<span style="font-size: 1.6rem; font-weight: 800; color: #ffffff; line-height: 1;">{symbol}</span>
<span style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #00E5FF; margin-left: 12px;">{last_price:.5f}</span>
<span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: {'#00FF88' if change >= 0 else '#FF4466'}; margin-left: 10px;">
{'▲' if change >= 0 else '▼'} {abs(change):.2%}
</span>
{pnl_html}
</div>
</div>
""", unsafe_allow_html=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Price", f"{last_price:.5f}", f"{change:+.2%}")
                        m2.metric("Volatility", f"{volatility:.3f}%")
                        m3.metric("RSI (14)", f"{current_rsi:.1f}",
                                  "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
                    except Exception as e:
                        logger.error(f"Header rendering failed for {symbol}: {e}")
                        st.caption("Header metrics currently unavailable.")
                else:
                    st.warning("Insufficient data for header calculation.")

                render_chart(df, symbol)

                # ── Trading Levels (Aligned Under Graph & Matched to Right Box Bottom) ──
                if result and result.get('tp_price') and (result.get('signal') in ('BUY', 'SELL') or pred in ('BUY', 'SELL')):
                    is_comm = any(x in symbol.upper() for x in ['XAU', 'GOLD', 'XAG', 'SILVER', 'OIL', 'WTI', 'BRENT'])
                    p_size = 0.01 if (is_comm or 'JPY' in symbol.upper()) else 0.0001
                    tp_pips = result.get('tp_pips') or 0
                    sl_pips = result.get('sl_pips') or 0
                    if not tp_pips and result.get('tp_price') and result.get('price_at_signal'):
                        tp_pips = int(round(abs(float(result['tp_price']) - float(result['price_at_signal'])) / p_size))
                    if not sl_pips and result.get('sl_price') and result.get('price_at_signal'):
                        sl_pips = int(round(abs(float(result['price_at_signal']) - float(result['sl_price'])) / p_size))
                    rr = (tp_pips / max(sl_pips, 1)) if sl_pips > 0 else 1.5

                    st.markdown(f"""
                    <div style="margin-top: 10px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border-glass);">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.05rem;">📍</span>
                                <span style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); letter-spacing: 0.02em;">Trading Levels</span>
                            </div>
                            <div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em;">
                                Automated Bracket Execution
                            </div>
                        </div>
                        <div class="trading-levels-grid">
                            <div class="trading-level-card tl-entry">
                                <div class="tl-header">Entry Price</div>
                                <div class="tl-value">{float(result['price_at_signal']):.5f}</div>
                                <div class="tl-footer">Execution Baseline</div>
                            </div>
                            <div class="trading-level-card tl-tp">
                                <div class="tl-header">Take Profit (TP)</div>
                                <div class="tl-value">{float(result['tp_price']):.5f}</div>
                                <div class="tl-footer">+{tp_pips} pips target</div>
                            </div>
                            <div class="trading-level-card tl-sl">
                                <div class="tl-header">Stop Loss (SL)</div>
                                <div class="tl-value">{float(result['sl_price']):.5f}</div>
                                <div class="tl-footer">-{sl_pips} pips risk</div>
                            </div>
                            <div class="trading-level-card tl-rr">
                                <div class="tl-header">Risk / Reward</div>
                                <div class="tl-value">1:{rr:.1f}</div>
                                <div class="tl-footer">Dynamic Bracket</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="margin-top: 10px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border-glass);">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.05rem;">📍</span>
                                <span style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); letter-spacing: 0.02em;">Trading Levels</span>
                            </div>
                        </div>
                        <div class="glass-card" style="height: 84px; padding: 0 18px; display: flex; align-items: center; justify-content: space-between; border-radius: 8px; box-sizing: border-box;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <span style="font-size: 1.2rem;">📡</span>
                                <div>
                                    <div style="font-weight: 700; font-size: 0.85rem; color: var(--text-primary);">Awaiting Validated Setup</div>
                                    <div style="font-size: 0.72rem; color: var(--text-secondary);">Execution brackets (Entry, TP, SL) activate when AI detects a high-conviction setup.</div>
                                </div>
                            </div>
                            <div style="font-family: var(--font-mono); font-size: 0.70rem; color: var(--text-muted); background: rgba(255,255,255,0.04); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border-glass);">
                                STATUS: MONITORING
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No chart data available. Check API connection.")

        with col_side:
            section_header("🤖", "AI Verdict")

            if result:
                p_buy = result.get('buy_prob') or 0.0
                p_sell = result.get('sell_prob') or 0.0
                p_wait = result.get('wait_prob') or 0.0

                # --- 3. Multi-Tier Conviction Stack (NEW) ---
                st.markdown('<div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 8px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Stacked Conviction View</div>', unsafe_allow_html=True)
                
                # Fetch all active tiers for this symbol
                all_active_tiers = db.get_active_signals(symbol=symbol, include_hidden=True)
                if all_active_tiers:
                    # Sort: Live first, then tier descending
                    all_active_tiers.sort(key=lambda x: (not bool(x.get('is_hidden', 0)), x.get('confidence_tier', 0)), reverse=True)
                    
                    seen_tiers = set()
                    for tier_data in all_active_tiers:
                        t_val = tier_data.get('confidence_tier', 0)
                        t_sig = tier_data.get('signal', 'WAIT')
                        
                        # FILTER: Skip neutral 'WAIT' signals or junk '0% Tier' data
                        if t_sig == 'WAIT' or int(t_val or 0) == 0:
                            continue
                        
                        t_live = not bool(tier_data.get('is_hidden', 0))
                        
                        # Create a unique key for the stack (Live/Shadow + Tier)
                        tier_key = f"{'LIVE' if t_live else 'SHADOW'}-{t_val}"
                        if tier_key in seen_tiers:
                            continue
                        seen_tiers.add(tier_key)
                        item_class = "tier-stack-live" if t_live else "tier-stack-shadow"
                        badge_label = "LIVE" if t_live else "SHADOW"
                        
                        st.markdown(f"""
                        <div class="tier-stack-item {item_class}">
                            <div style="font-weight: 600; font-family: var(--font-mono); color: {'var(--signal-buy)' if t_sig == 'BUY' else 'var(--signal-sell)' if t_sig == 'SELL' else 'var(--text-muted)'};">
                                {t_val}% {t_sig}
                            </div>
                            <div class="tier-stack-badge" style="background: { 'rgba(0,255,136,0.1)' if t_live else 'rgba(255,255,255,0.05)' }; color: { 'var(--signal-buy)' if t_live else 'var(--text-muted)' }; border: 1px solid { 'var(--signal-buy-border)' if t_live else 'var(--border-glass)' };">
                                {badge_label}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No secondary convictions detected.")

                st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)

                try:
                    # Clean the tier string (handles cases like '70%70%' or None)
                    _raw_tier = result.get('winning_tier', st.session_state.get('accuracy_target', '60%'))
                    import re as _re
                    _tier_match = _re.search(r'(\d+)', str(_raw_tier))
                    _tier_num = int(_tier_match.group(1)) if _tier_match else 60
                    # Clamp to nearest valid tier
                    _valid_tiers = [60, 70, 80, 90, 100]
                    _clamped_tier = str(min(_valid_tiers, key=lambda t: abs(t - _tier_num)))
                    winning_tier = _clamped_tier
                    
                    from core.performance_gate import get_performance_gate
                    perf_gate = get_performance_gate()
                    is_approved = perf_gate.is_tier_approved(symbol, pred, float(winning_tier) / 100.0)
                    
                    regime = str(result.get('regime') or 'RANGING').upper()
                    is_crisis = 'CRISIS' in regime
                    
                    status_text = "PASSED" if (conf > 0 and pred != "WAIT") else "FILTERED (Caution)" if (pred == "WAIT" and conf > 0.1) else "FILTERED"
                    status_color = "var(--text-muted)" # Initial fallback
                    
                    # Dynamic override for Crisis/Safety/Blacklist blocks
                    if is_crisis:
                        status_text = "⚠️ CRISIS BLOCK (Safety)"
                        status_color = "#FF4466" # Bright Red
                        pred = "WAIT"
                    elif is_sym_blocked or (pred == "BUY" and is_buy_blocked) or (pred == "SELL" and is_sell_blocked):
                        if is_sym_blocked:
                            status_text = "🛑 COMMODITY SHIELD (Live Blocked)"
                        elif pred == "BUY" and is_buy_blocked:
                            status_text = "🚫 DIRECTIONAL BLACKLIST (BUY Blocked)"
                        elif pred == "SELL" and is_sell_blocked:
                            status_text = "🚫 DIRECTIONAL BLACKLIST (SELL Blocked)"
                        else:
                            status_text = "🚫 BLACKLISTED (Live Blocked)"
                        status_color = "#FF4466"
                        pred = "WAIT"
                    elif is_market_closed:
                        status_text = "HISTORICAL ANALYSIS"
                        status_color = "var(--text-muted)"
                        if pred in ('BUY', 'SELL'): pred = "WAIT"
                    elif pred in ('BUY', 'SELL'):
                        is_hidden = bool(result.get('is_hidden', 0))
                        # If terminal is locked to an active trade, ignore global 'is_actively_trading' resting filter 
                        # so that shadow/live active positions keep their locked state and show correct indicators.
                        is_locked_trade = locked_trade is not None and result.get('id') == locked_trade.get('id')
                        if not is_actively_trading and not is_locked_trade:
                            status_text = f"RESTING (AI Conviction: {pred})"
                            status_color = "var(--text-muted)"
                            pred = "WAIT"
                        elif is_hidden or not is_approved:
                            status_text = "CERTIFICATION PHASE (Shadow)"
                            status_color = "var(--accent-gold)"
                        else:
                            status_color = "var(--signal-buy)" if pred == "BUY" else "var(--signal-sell)"
                    else:
                        status_color = "var(--accent-gold)" if (pred == "WAIT" and conf > 0.1) else "var(--text-muted)"
                except Exception as e:
                    logger.warning(f"Status calculation failed: {e}")
                    status_text = "INITIALIZING..."
                    status_color = "var(--text-muted)"
                    winning_tier = "60"

                # --- 4. Main Verdict Display (Legacy Refactored) ---
                st.markdown(f"""
                <div style="padding: 16px; background: {status_color if '#000' in status_color else 'rgba(255,255,255,0.02)'}; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 4px;">SYSTEM STATUS</div>
                    <div style="font-weight: 700; color: {status_color}; font-size: 0.9rem;">{status_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- 3. Render AI Verdict Card ---
                try:
                    ts_display = "Just Now"
                    ts_full = ""
                    try:
                        ts_obj = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
                        ts_display = ts_obj.strftime("%d %b %H:%M")
                        ts_full = ts_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
                    except: pass

                    display_conf = conf or 0.0
                    vol_trades = result.get('model_trades', 0) or 0
                    
                    css = f"signal-{pred.lower()}"
                    
                    st.markdown(f"""
<div class="glass-card" style="padding: 24px; text-align: center; border-top: 3px solid {status_color}; min-height: 575px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
<div>
<!-- 1. DECISION LAYER -->
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
    <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.15em; color: var(--text-muted); text-transform: uppercase;">
    Target: {winning_tier}%
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);">
    🕒 {ts_display}
    </div>
</div>
<div style="text-align: right; font-family: var(--font-mono); font-size: 0.58rem; color: rgba(255,255,255,0.25); margin-bottom: 16px; letter-spacing: 0.05em;">
Signal Generated: {ts_full}
</div>
<div class="signal-badge {css}" style="margin-bottom: 20px;">{pred}</div>
{f'<div style="font-family: var(--font-mono); font-size: 0.6rem; color: #00FF88; margin-top: -15px; margin-bottom: 15px;">AI INTENT: {result.get("expert_intent")}</div>' if (pred == "WAIT" and result.get("expert_intent") and result.get("expert_intent") != "WAIT") else ''}
<!-- 2. EXPERT CONVICTION vs HURDLE -->
<div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--border-glass); text-align: left;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Expert Conviction</span>
<span style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan);">{display_conf:.1%}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Precision Hurdle</span>
<span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);">{f"{result.get('regime_threshold', 0):.0%}" if result.get('regime_threshold') else f"{winning_tier}%"}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Expertise Volume</span>
<span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-cyan);">{vol_trades} Trades</span>
</div>
<div style="margin-top: 10px; font-family: var(--font-mono); font-size: 0.6rem; color: {status_color}; font-weight: 700; letter-spacing: 0.1em;">
STATUS: {status_text}
</div>
</div>
<!-- 3. MARKET HEATMAP -->
<div style="padding-top: 10px; border-top: 1px solid var(--border-glass);">
<div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.05em;">Market Sentiment Heatmap</div>
<div style="display: flex; height: 6px; border-radius: 3px; overflow: hidden; background: rgba(255,255,255,0.05); margin-bottom: 10px;">
<div style="width: {(p_buy or 0.0):.1%}; background: var(--signal-buy);"></div>
<div style="width: {(p_wait or 0.0):.1%}; background: var(--signal-wait);"></div>
<div style="width: {(p_sell or 0.0):.1%}; background: var(--signal-sell);"></div>
</div>
<div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.65rem;">
<div style="color: var(--signal-buy);">B {(p_buy or 0.0):.0%}</div>
<div style="color: var(--signal-wait);">W {(p_wait or 0.0):.0%}</div>
<div style="color: var(--signal-sell);">S {(p_sell or 0.0):.0%}</div>
</div>
</div>
</div>
<!-- 4. SAFETY EXPLAINER (PINNED TO BOTTOM) -->
<div style="margin-top: 15px; padding: 12px; background: rgba(0, 229, 255, 0.03); border-radius: 8px; border: 1px dashed rgba(0, 229, 255, 0.1);">
<div style="font-size: 0.65rem; color: var(--accent-cyan); font-weight: 700; margin-bottom: 5px; text-transform: uppercase;">Safety Intelligence Audit</div>
<p style="font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4; margin: 0;">
                            {"Setup validated by core logic, but filtered by the 15% Heatmap Caution gate to protect against fake breakouts." if (pred == "WAIT" and conf > 0.5) else 
                             "AI is currently monitoring institutional flows. High-conviction entry pending institutional surge." if (pred == "WAIT" and conf <= 0.5) else
                             "High-conviction institutional entry detected. Precision targeting active."}
</p>
</div>
</div>
""", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Card rendering failed: {e}")
                    st.error("AI Verdict Card: Initialization in Progress...")


            else:
                # No result yet — show an informative monitoring card
                # (either model not trained for this pair, or inference is still loading)
                last_price_display = ""
                try:
                    if not df.empty:
                        last_price_display = f"{df['close'].iloc[-1]:.5f}"
                except: pass

                st.markdown(f"""
<div class="glass-card" style="padding: 24px; text-align: center; border-top: 3px solid var(--text-muted); min-height: 575px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
  <div>
    <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.15em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 16px;">
      {symbol} · Monitoring
    </div>
    <div class="signal-badge signal-wait" style="margin-bottom: 20px;">WAIT</div>
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; margin-bottom: 16px; border: 1px solid var(--border-glass);">
      <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px;">Last Price</div>
      <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan);">{last_price_display or "—"}</div>
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted); font-weight: 700; letter-spacing: 0.1em; margin-bottom: 16px;">
      STATUS: SCANNING...
    </div>
  </div>
  <div style="padding: 12px; background: rgba(0, 229, 255, 0.03); border-radius: 8px; border: 1px dashed rgba(0, 229, 255, 0.1);">
    <div style="font-size: 0.65rem; color: var(--accent-cyan); font-weight: 700; margin-bottom: 5px; text-transform: uppercase;">AI Status</div>
    <p style="font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4; margin: 0;">
      No specialist model is currently certified for {symbol}. The engine is monitoring for high-conviction setups.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    # Invoke the fragment — first call renders, subsequent calls auto-rerun every 15s
    _live_terminal_data()

def _show_manual_m15_cockpit(symbol: str, all_pairs: list):
    """
    Manual M15 Wick Sniper Terminal.
    Full technical analysis suite with interactive candlestick charting, AI Foundation
    intelligence, live trade locking, precision M15 wick execution, and complete audit recording.
    """
    try:
        from core.manual_model import (
            get_forming_candle, get_last_15m_candle, calculate_manual_order,
            submit_manual_order, get_active_manual_orders,
            cancel_manual_order, close_manual_position, get_pip_size,
            arm_m15_order, disarm_order, get_armed_orders,
            start_armed_sniper_watcher
        )
        start_armed_sniper_watcher()
    except ImportError as e:
        st.error(f"Manual Model module not available: {e}")
        return

    engine = load_engine()
    inf_engine = load_inference_v2()
    db = get_db()

    # ── Top Bar: Pair & Timeframe Selector + Live Quote ───────────────────────────
    col_sym, col_tf, col_quote, col_ref = st.columns([1.8, 1.2, 3.2, 0.8])
    with col_sym:
        sniper_sym = st.selectbox(
            "Instrument", all_pairs,
            index=all_pairs.index(symbol) if symbol in all_pairs else 0,
            key="sniper_symbol",
            label_visibility="collapsed"
        )
    with col_tf:
        m15_tf = st.selectbox(
            "Timeframe", ["15m", "1h", "4h", "1d"],
            index=0,
            key="sniper_tf_selector",
            label_visibility="collapsed"
        )
    with col_quote:
        try:
            from core.mt5_connector import MT5Connector
            _mt5_q = MT5Connector().get_connection()
            if _mt5_q is None:
                import MetaTrader5 as _mt5_q
                _mt5_q.initialize()
            _si  = _mt5_q.symbol_info(sniper_sym)
            _acc = _mt5_q.account_info()
            if _si and _acc:
                _sp_pips = _si.spread * _si.point / get_pip_size(sniper_sym)
                _login_str = f"#{_acc.login}" if hasattr(_acc, 'login') else ""
                st.markdown(
                    f'<div style="padding:6px 14px;background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.15);border-radius:8px;display:flex;gap:18px;align-items:center">'
                    f'<span style="font-family:monospace;font-size:0.95rem;font-weight:700;color:#29b6f6">Bid: {_si.bid:.5f}</span>'
                    f'<span style="font-family:monospace;font-size:0.95rem;font-weight:700;color:#00e676">Ask: {_si.ask:.5f}</span>'
                    f'<span style="font-size:0.75rem;color:var(--text-secondary)">Spread: {_sp_pips:.1f}p</span>'
                    f'<span style="font-size:0.75rem;color:var(--text-secondary)">Master {_login_str}: <b>${_acc.balance:,.2f}</b></span>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.caption("Awaiting MT5 quotes...")
        except Exception:
            st.caption("Connect MT5 to see live quotes.")
    with col_ref:
        if st.button("🔄", key="sniper_top_refresh", help="Refresh M15 candle & terminal data"):
            st.rerun()

    # ── Trade Locking & Active Position Detection ─────────────────────────────
    active_manual_orders = []
    armed_setups = []
    try:
        active_manual_orders = get_active_manual_orders()
    except Exception:
        pass
    try:
        armed_setups = get_armed_orders()
    except Exception:
        pass

    sym_open = [o for o in active_manual_orders if o.get("symbol") == sniper_sym and o.get("type") == "OPEN"]
    sym_pending = [o for o in active_manual_orders if o.get("symbol") == sniper_sym and o.get("type") == "PENDING"]
    sym_armed = [a for a in armed_setups if a.get("symbol") == sniper_sym and a.get("status") == "ARMED_WAITING_CLOSE"]

    locked_trade = None
    lock_mode = None
    if sym_open:
        locked_trade = sym_open[0]
        lock_mode = "OPEN"
    elif sym_pending:
        locked_trade = sym_pending[0]
        lock_mode = "PENDING"
    elif sym_armed:
        locked_trade = sym_armed[0]
        lock_mode = "ARMED"

    # ── Fetch Candlestick Chart Data & Technical Metrics ───────────────────────
    df = pd.DataFrame()
    try:
        df = inf_engine.data_engine.fetch(sniper_sym, interval=m15_tf, days=4, use_cache=False)
    except Exception as e:
        logger.warning(f"Could not fetch {sniper_sym} data for chart: {e}")

    # Technical Indicators (Price, RSI, Volatility, Spread)
    last_price = 0.0
    change = 0.0
    current_rsi = 50.0
    volatility = 0.0
    if not df.empty and len(df) >= 2:
        try:
            last_price = float(df['close'].iloc[-1])
            prev_price = float(df['close'].iloc[-2])
            change = (last_price - prev_price) / prev_price if prev_price > 0 else 0.0
            current_rsi = calculate_rsi_manual(df['close'])
            volatility = float(df['close'].pct_change().std() * 100)
            if np.isnan(volatility): volatility = 0.0
        except Exception:
            pass

    # ── Fetch AI Foundation Intelligence Context ──────────────────────────────
    ai_result = None
    try:
        ai_result = inf_engine.predict_symbol(
            sniper_sym, save_to_db=False,
            win_rate=st.session_state.get('accuracy_target', '70%'),
            allow_stale=True, use_cache=False
        )
    except Exception as e:
        logger.debug(f"AI pulse fetch in manual cockpit: {e}")

    ai_pred = ai_result.get('signal', 'WAIT') if ai_result else 'WAIT'
    ai_conf = ai_result.get('confidence', 0.0) if ai_result else 0.0
    ai_regime = str(ai_result.get('regime') or 'RANGING').upper() if ai_result else 'RANGING'
    p_buy = float(ai_result.get('buy_prob') or 0.0) if ai_result else 0.0
    p_wait = float(ai_result.get('wait_prob') or 0.0) if ai_result else 0.0
    p_sell = float(ai_result.get('sell_prob') or 0.0) if ai_result else 0.0

    # ── High-Visibility Locked Position / Armed Banner ────────────────────────
    pip_sz = get_pip_size(sniper_sym)
    if lock_mode == "OPEN" and locked_trade:
        open_dir = locked_trade.get("direction", "BUY")
        open_entry = float(locked_trade.get("entry", 0.0))
        open_pnl = float(locked_trade.get("pnl") or 0.0)
        open_tkt = locked_trade.get("ticket", "")
        dir_mult = 1 if open_dir == "BUY" else -1
        pnl_pips = ((last_price - open_entry) / pip_sz * dir_mult) if last_price > 0 and open_entry > 0 else 0.0
        pnl_color = "#00FF88" if open_pnl >= 0 else "#FF4466"

        st.markdown(f"""
        <div style="background:rgba(0,229,255,0.08);border:1px solid #00e5ff;padding:10px 18px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;margin:10px 0 14px 0;box-shadow:0 0 18px rgba(0,229,255,0.15)">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
                <span style="background:#00E5FF;color:#000;padding:4px 12px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.75rem;font-weight:900;letter-spacing:0.05em">LOCKED</span>
                <span style="font-weight:800;color:#ffffff;font-size:0.95rem">ACTIVE POSITION #{open_tkt}</span>
                <span style="color:{'#00FF88' if open_dir == 'BUY' else '#FF4466'};font-weight:700;font-size:0.9rem">{open_dir} {locked_trade.get('lots', '')} Lots @ {open_entry:.5f}</span>
                <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:{pnl_color};background:rgba(255,255,255,0.06);padding:2px 10px;border-radius:4px">{pnl_pips:+.1f} pips ({open_pnl:+.2f} USD)</span>
                <span style="font-size:0.75rem;color:var(--text-secondary)">SL: {locked_trade.get('sl', 0):.5f} · TP: {locked_trade.get('tp', 0):.5f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lock_mode == "PENDING" and locked_trade:
        pen_tkt = locked_trade.get("ticket", "")
        pen_dir = locked_trade.get("direction", "BUY")
        pen_entry = float(locked_trade.get("entry", 0.0))
        pen_dir_color = "#00FF88" if pen_dir == "BUY" else "#FF4466"
        st.markdown(f"""
        <div style="background:rgba(255,214,0,0.08);border:1px solid #ffd600;padding:10px 18px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;margin:10px 0 14px 0">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
                <span style="background:#FFD600;color:#000;padding:4px 12px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.75rem;font-weight:900;letter-spacing:0.05em">LOCKED</span>
                <span style="font-weight:800;color:#ffffff;font-size:0.95rem">PENDING STOP ORDER #{pen_tkt}</span>
                <span style="color:{pen_dir_color};font-weight:700;font-size:0.9rem">{pen_dir} {locked_trade.get('lots', '')} Lots @ {pen_entry:.5f}</span>
                <span style="font-size:0.75rem;color:var(--text-secondary)">Awaiting next candle wick touch · SL: {locked_trade.get('sl', 0):.5f} · TP: {locked_trade.get('tp', 0):.5f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lock_mode == "ARMED" and locked_trade:
        arm_dir = locked_trade.get("direction", "BUY")
        arm_close = locked_trade.get("target_close_time", "")
        arm_id = locked_trade.get("arm_id", "")
        arm_dir_color = "#00FF88" if arm_dir == "BUY" else "#FF4466"
        try:
            arm_dt = datetime.fromisoformat(arm_close)
            sec_left = max(0, int((arm_dt - datetime.now(timezone.utc)).total_seconds()))
            countdown_label = f"{sec_left//60:02d}:{sec_left%60:02d} left"
            close_label = arm_dt.strftime("%H:%M UTC")
        except Exception:
            countdown_label = "awaiting close"
            close_label = arm_close[:16]

        st.markdown(f"""
        <div style="background:rgba(0,230,118,0.08);border:1px solid #00e676;padding:10px 18px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;margin:10px 0 14px 0">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
                <span style="background:#00E676;color:#000;padding:4px 12px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.75rem;font-weight:900;letter-spacing:0.05em">ARMED</span>
                <span style="font-weight:800;color:#ffffff;font-size:0.95rem">M15 WICK SNIPER PRE-ARMED</span>
                <span style="color:{arm_dir_color};font-weight:700;font-size:0.9rem">{arm_dir} (Next M15 Candle)</span>
                <span style="font-family:monospace;font-size:0.95rem;color:#00FF88">Locks in {countdown_label} (at {close_label})</span>
                <span style="font-size:0.75rem;color:var(--text-secondary)">RRR 1:{locked_trade.get('rrr', 1.5)} · Risk {locked_trade.get('risk_value', 0.5)}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Fetch Forming M15 Candle (for sniper calculations) ───────────────────
    candle = get_forming_candle(sniper_sym)

    # ── Two-Column Main Layout (Left: Chart & Analysis, Right: AI & Cockpit) ──
    col_main, col_side = st.columns([3, 1])

    with col_side:
        section_header("🤖", "AI Market Intelligence")
        
        # 1. AI Regime & Verdict Card
        status_color = "#00FF88" if ai_pred == "BUY" else "#FF4466" if ai_pred == "SELL" else "var(--text-muted)"
        if "CRISIS" in ai_regime:
            status_text = "⚠️ CRISIS REGIME"
            status_color = "#FF4466"
        else:
            status_text = f"{ai_regime} ({ai_pred})"

        st.markdown(f"""
        <div style="padding:12px;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid rgba(255,255,255,0.08);margin-bottom:14px">
            <div style="font-size:0.7rem;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px">Foundation Brain Bias</div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:1.1rem;font-weight:800;color:{status_color}">{ai_pred}</span>
                <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:var(--accent-cyan)">{ai_conf:.1%}</span>
            </div>
            <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">Regime: <b>{ai_regime}</b></div>
            <!-- Sentiment Heatmap -->
            <div style="display:flex;height:5px;border-radius:3px;overflow:hidden;background:rgba(255,255,255,0.05);margin-top:10px;margin-bottom:6px">
                <div style="width:{p_buy:.1%};background:var(--signal-buy)"></div>
                <div style="width:{p_wait:.1%};background:var(--signal-wait)"></div>
                <div style="width:{p_sell:.1%};background:var(--signal-sell)"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:0.62rem;color:var(--text-muted)">
                <span>B {p_buy:.0%}</span>
                <span>W {p_wait:.0%}</span>
                <span>S {p_sell:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("⚙️", "Order Configuration")
        direction = st.radio(
            "Direction", ["BUY", "SELL"], horizontal=True, key="sniper_direction",
            format_func=lambda x: f"🟢 {x}" if x == "BUY" else f"🔴 {x}"
        )

        can_arm_prev = candle.get("can_arm_previous", False) if candle else False
        grace_left = candle.get("grace_seconds_left", 0) if candle else 0
        target_candle_sel = "forming"

        if can_arm_prev:
            m_left = grace_left // 60
            s_left = grace_left % 60
            st.markdown(f"""
            <div style="background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.3);border-radius:8px;padding:8px 10px;margin-bottom:8px">
                <div style="font-weight:700;font-size:0.78rem;color:#00E676;display:flex;justify-content:space-between">
                    <span>🟢 5-MIN GRACE WINDOW</span>
                    <span style="font-family:monospace">{m_left:02d}:{s_left:02d} left</span>
                </div>
                <div style="font-size:0.7rem;color:var(--text-secondary);margin-top:2px">
                    You can trade the previous completed candle's wicks.
                </div>
            </div>
            """, unsafe_allow_html=True)
            mode_choice = st.radio(
                "Wick Target Reference",
                [
                    f"🎯 Previous Candle (5-Min Grace: {m_left:02d}:{s_left:02d})",
                    "⏳ Current Forming Candle (Pre-Arm for Close)"
                ],
                index=0,
                key="sniper_target_choice",
                help="Select whether to trade using the previous candle's locked wicks (immediate submission) or pre-arm the currently forming candle."
            )
            target_candle_sel = "previous" if "Previous Candle" in mode_choice else "forming"

        sl_buffer = st.number_input(
            "SL Buffer (pips beyond wick)",
            min_value=0.5, max_value=20.0, value=5.0, step=0.5,
            format="%.1f", key="sniper_sl_buffer",
            help="BUY: SL = Low wick − buffer  |  SELL: SL = High wick + buffer"
        )

        rrr_map = {"1:1.5 (Default)": 1.5, "1:2.0": 2.0, "1:1.0": 1.0}
        rrr_sel = st.radio("Risk:Reward Ratio", list(rrr_map.keys()), key="sniper_rrr")
        rrr = rrr_map[rrr_sel]

        risk_type_sel = st.radio("Risk Mode", ["Account %", "Fixed $"], horizontal=True, key="sniper_risk_type")
        risk_type = "percent" if risk_type_sel == "Account %" else "amount"

        if risk_type == "percent":
            risk_value = st.number_input("Risk %", min_value=0.01, max_value=10.0, value=0.5, step=0.1, format="%.2f", key="sniper_risk_pct")
        else:
            risk_value = st.number_input("Risk Amount ($)", min_value=1.0, max_value=10000.0, value=50.0, step=5.0, format="%.2f", key="sniper_risk_amt")

        broadcast_toggle = st.toggle("📡 Copy to Subscribers", value=True, key="sniper_broadcast", help="Execute order on all connected broker accounts.")
        telegram_toggle = st.toggle("📲 Telegram Alert", value=True, key="sniper_telegram")

        # Finish Color Safety Notice
        st.caption(
            "🛡️ **Finish Color Rule**: SELL setup cancels if candle closes Bullish. BUY setup cancels if candle closes Bearish."
        )

        # Compute Pre-Flight Spec
        spec = {}
        if candle:
            spec = calculate_manual_order(
                symbol=sniper_sym, direction=direction, rrr=rrr,
                risk_type=risk_type, risk_value=risk_value,
                sl_buffer_pips=sl_buffer,
                target_candle=target_candle_sel,
            )

        if spec and not spec.get("valid") and spec.get("error"):
            st.error(f"⚠️ {spec.get('error')}")

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        
        # Primary Action Button
        if lock_mode == "ARMED" and sym_armed:
            disarm_id = sym_armed[0].get("arm_id")
            if st.button("❌ DISARM SETUP", type="secondary", use_container_width=True, key="btn_disarm_main"):
                disarm_order(disarm_id)
                st.toast(f"Setup {sniper_sym} disarmed.", icon="ℹ️")
                st.rerun()
        elif lock_mode == "OPEN" and sym_open:
            close_tkt = sym_open[0].get("ticket")
            if st.button("🚨 CLOSE POSITION", type="secondary", use_container_width=True, key="btn_close_main"):
                cr = close_manual_position(close_tkt)
                if cr.get("success"):
                    st.toast(f"Position #{close_tkt} closed.", icon="✅")
                    st.rerun()
                else:
                    st.error(f"Close failed: {cr.get('error')}")
        else:
            btn_title = (
                f"🚀 SUBMIT M15 SNIPER — {direction} {sniper_sym} (Prev Wicks)"
                if target_candle_sel == "previous" else
                f"🚀 ARM M15 SNIPER — {direction} {sniper_sym}"
            )
            arm_btn = st.button(
                btn_title,
                type="primary", use_container_width=True, key="sniper_arm_btn",
                disabled=not (spec and spec.get("valid"))
            )
            if arm_btn and spec and spec.get("valid"):
                with st.spinner(f"Processing M15 Wick Sniper for {direction} {sniper_sym}..."):
                    arm_result = arm_m15_order(
                        symbol=sniper_sym,
                        direction=direction,
                        rrr=rrr,
                        risk_type=risk_type,
                        risk_value=risk_value,
                        sl_buffer_pips=sl_buffer,
                        broadcast_to_subscribers=broadcast_toggle,
                        send_telegram=telegram_toggle,
                        target_candle=target_candle_sel,
                    )
                if arm_result.get("success"):
                    if arm_result.get("placed_immediately"):
                        st.success(
                            f"🎯 **Order #{arm_result.get('ticket')} PLACED on MT5!**\n\n"
                            f"• Executed on Previous Candle wicks under 5-minute grace window.\n"
                            f"• Expiry set to end of current candle (**{arm_result.get('target_close_str')}**).\n"
                            f"• Broadcast to connected subscriber accounts."
                        )
                    else:
                        st.success(
                            f"🎯 **Setup PRE-ARMED for {direction} {sniper_sym}!**\n\n"
                            f"• Watching current M15 candle.\n"
                            f"• Locks wick at **{arm_result.get('target_close_str')}**.\n"
                            f"• 🛡️ Evaluates finish color (auto-cancels if finishes opposite color).\n"
                            f"• Places **{direction}_STOP** pending order on MT5 & copy accounts."
                        )
                    st.rerun()
                else:
                    st.error(f"Arming failed: {arm_result.get('error')}")

    with col_main:
        # Determine chart levels: if locked position/pending exists, plot them; otherwise plot pre-flight spec
        chart_levels = None
        if lock_mode in ("OPEN", "PENDING") and locked_trade:
            chart_levels = {
                "entry": locked_trade.get("entry"),
                "tp": locked_trade.get("tp"),
                "sl": locked_trade.get("sl"),
            }
        elif spec and spec.get("valid"):
            chart_levels = {
                "entry": spec.get("entry"),
                "tp": spec.get("tp"),
                "sl": spec.get("sl"),
            }

        # 1. Header Metrics Row
        if not df.empty and len(df) >= 2:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Live Price", f"{last_price:.5f}", f"{change:+.2%}")
            m2.metric("Timeframe", f"{m15_tf.upper()}")
            m3.metric("Volatility", f"{volatility:.3f}%")
            m4.metric("RSI (14)", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
            if lock_mode == "OPEN" and locked_trade:
                pnl_val = float(locked_trade.get("pnl") or 0.0)
                m5.metric("Floating P/L", f"${pnl_val:+.2f}", f"{pnl_pips:+.1f} pips")
            else:
                m5.metric("Engine", "M15 SNIPER", "Discretionary")

        # 2. Interactive Candlestick Chart (with Entry, TP, SL price lines)
        render_chart(df, sniper_sym, key=f"m15_{m15_tf}_{sniper_sym.lower()}", levels=chart_levels)

        # 3. M15 Reference Candle Card
        if candle:
            now_utc = datetime.now(timezone.utc)
            candle_close = candle.get("close_time")
            next_close   = candle.get("next_candle_close")
            time_to_close = int((candle_close - now_utc).total_seconds()) if candle_close else 0
            time_to_close = max(0, time_to_close)
            mins_left  = time_to_close // 60
            secs_left  = time_to_close % 60
            candle_open_str  = candle["time"].strftime("%H:%M")
            candle_close_str = candle_close.strftime("%H:%M UTC") if candle_close else "—"
            next_close_str   = next_close.strftime("%H:%M UTC") if next_close else "—"

            if target_candle_sel == "previous" and candle.get("previous_candle"):
                ref = candle["previous_candle"]
                candle_range_pips = (ref["high"] - ref["low"]) / pip_sz
                bull = ref["is_bullish"]
                body_color = "#00e676" if bull else "#ff4466"
                status_color = "#00e676" if bull else "#ff4466"
                finish_text = "🟢 CLOSED BULLISH" if bull else "🔴 CLOSED BEARISH"
                ref_open_str = ref["open_time"].strftime("%H:%M")
                ref_close_str = ref["close_time"].strftime("%H:%M UTC")

                st.markdown(f"""
                <div style="padding:12px 16px;background:rgba(255,255,255,0.03);border:1px solid rgba(0,230,118,0.3);border-radius:10px;margin-top:10px;margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px">
                        <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;color:#00E676;font-weight:700">
                            🎯 Reference Target: Previous Closed Candle &middot; {ref_open_str}–{ref_close_str}
                        </span>
                        <div style="display:flex;gap:8px;align-items:center">
                            <span style="padding:2px 10px;border-radius:10px;background:{status_color}22;color:{status_color};border:1px solid {status_color}44;font-size:0.7rem;font-weight:700">{finish_text}</span>
                            <span style="font-size:0.7rem;color:var(--text-secondary)">Current candle ends: <b>{candle_close_str}</b> ({mins_left:02d}:{secs_left:02d} left)</span>
                        </div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center">
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">PREV HIGH (BUY)</span><br>
                            <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#00e676">HIGH {ref["high"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">OPEN</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;color:var(--text-primary)">{ref["open"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">CLOSE</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;font-weight:700;color:{body_color}">{ref["close"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">PREV LOW (SELL)</span><br>
                            <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#ff4466">LOW {ref["low"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">RANGE</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;color:var(--accent-cyan)">{candle_range_pips:.1f} pips</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                candle_range_pips = (candle["high"] - candle["low"]) / pip_sz
                bull = candle["close"] >= candle["open"]
                body_color = "#00e676" if bull else "#ff4466"
                is_forming = candle.get("is_forming", False)

                status_badge = (
                    f'<span style="padding:2px 10px;border-radius:10px;background:#00e67622;color:#00e676;'
                    f'border:1px solid #00e67644;font-size:0.7rem;font-weight:700">⏱ FORMING · {mins_left:02d}:{secs_left:02d} left</span>'
                    if is_forming else
                    f'<span style="padding:2px 10px;border-radius:10px;background:#ff446622;color:#ff4466;'
                    f'border:1px solid #ff446644;font-size:0.7rem;font-weight:700">🔴 MARKET CLOSED</span>'
                )

                st.markdown(f"""
                <div style="padding:12px 16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;margin-top:10px;margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px">
                        <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-tertiary)">
                            Current M15 Candle &middot; {candle_open_str}–{candle_close_str}
                        </span>
                        <div style="display:flex;gap:8px;align-items:center">
                            {status_badge}
                            <span style="font-size:0.7rem;color:var(--text-secondary)">Next candle closes: <b>{next_close_str}</b></span>
                        </div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center">
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">BUY TRIGGER</span><br>
                            <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#00e676">HIGH {candle["high"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">OPEN</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;color:var(--text-primary)">{candle["open"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">CURRENT</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;font-weight:700;color:{body_color}">{candle["close"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">SELL TRIGGER</span><br>
                            <span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#ff4466">LOW {candle["low"]:.5f}</span>
                        </div>
                        <div>
                            <span style="font-size:0.68rem;color:var(--text-secondary)">CANDLE RANGE</span><br>
                            <span style="font-family:monospace;font-size:0.95rem;color:var(--accent-cyan)">{candle_range_pips:.1f} pips</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 4. Trading Levels Execution Bracket
        disp_entry = 0.0
        disp_tp = 0.0
        disp_sl = 0.0
        disp_tp_pips = 0.0
        disp_sl_pips = 0.0
        disp_rr = rrr
        disp_label = "Pre-Flight Calculation"

        if lock_mode in ("OPEN", "PENDING") and locked_trade:
            disp_entry = float(locked_trade.get("entry", 0.0))
            disp_tp = float(locked_trade.get("tp", 0.0))
            disp_sl = float(locked_trade.get("sl", 0.0))
            disp_tp_pips = abs(disp_tp - disp_entry) / pip_sz if pip_sz > 0 else 0
            disp_sl_pips = abs(disp_entry - disp_sl) / pip_sz if pip_sz > 0 else 0
            disp_rr = (disp_tp_pips / max(disp_sl_pips, 1.0)) if disp_sl_pips > 0 else rrr
            disp_label = f"Active Position #{locked_trade.get('ticket')}" if lock_mode == "OPEN" else f"Pending Stop #{locked_trade.get('ticket')}"
        elif spec and spec.get("valid"):
            disp_entry = spec.get("entry", 0.0)
            disp_tp = spec.get("tp", 0.0)
            disp_sl = spec.get("sl", 0.0)
            disp_tp_pips = spec.get("tp_pips", 0.0)
            disp_sl_pips = spec.get("sl_pips", 0.0)
            disp_rr = spec.get("rrr", rrr)
            disp_label = f"M15 Wick Auto-Brackets ({spec.get('lots', '')} Lots · Risk ${spec.get('risk_usd', 0):.2f})"

        disp_entry_footer = "Low Wick Target" if direction == "SELL" else "High Wick Target"

        if disp_entry > 0 and disp_tp > 0 and disp_sl > 0:
            st.markdown(f"""
            <div style="margin-top: 6px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border-glass);">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1rem;">📍</span>
                        <span style="font-size: 0.9rem; font-weight: 700; color: var(--text-primary);">Trading Levels & Execution Brackets</span>
                    </div>
                    <div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase;">
                        {disp_label}
                    </div>
                </div>
                <div class="trading-levels-grid">
                    <div class="trading-level-card tl-entry">
                        <div class="tl-header">Entry Price</div>
                        <div class="tl-value">{disp_entry:.5f}</div>
                        <div class="tl-footer">{disp_entry_footer}</div>
                    </div>
                    <div class="trading-level-card tl-tp">
                        <div class="tl-header">Take Profit (TP)</div>
                        <div class="tl-value">{disp_tp:.5f}</div>
                        <div class="tl-footer">+{disp_tp_pips:.1f} pips target</div>
                    </div>
                    <div class="trading-level-card tl-sl">
                        <div class="tl-header">Stop Loss (SL)</div>
                        <div class="tl-value">{disp_sl:.5f}</div>
                        <div class="tl-footer">-{disp_sl_pips:.1f} pips ({sl_buffer:.1f}p buffer)</div>
                    </div>
                    <div class="trading-level-card tl-rr">
                        <div class="tl-header">Risk / Reward</div>
                        <div class="tl-value">1:{disp_rr:.1f}</div>
                        <div class="tl-footer">Selectable Bracket</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Active Orders & Armed Queue Monitor (Full Width) ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        section_header("📊", "Active M15 Sniper Orders & Armed Queue")
        st.caption("All pre-armed setups awaiting candle close + live pending and open MT5 positions (Magic #202425)")

        # 1. Armed Queue
        if armed_setups:
            st.markdown("##### ⏱️ Pre-Armed Setups (Waiting for Candle Close)")
            for a in armed_setups:
                a_sym = a.get("symbol", "?")
                a_dir = a.get("direction", "?")
                a_close = a.get("target_close_time", "")
                a_id = a.get("arm_id", "")
                a_dir_color = "#00e676" if a_dir == "BUY" else "#ff4466"
                try:
                    c_dt = datetime.fromisoformat(a_close)
                    c_str = c_dt.strftime("%H:%M UTC")
                    sec_left = max(0, int((c_dt - datetime.now(timezone.utc)).total_seconds()))
                    countdown_str = f"locks in {sec_left//60:02d}:{sec_left%60:02d}"
                except Exception:
                    c_str = a_close[:16]
                    countdown_str = "pending close"

                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(
                            f'<span style="padding:2px 8px;border-radius:6px;background:#ffd60022;color:#ffd600;border:1px solid #ffd60044;font-size:0.72rem;font-weight:700">⏱ ARMED</span> '
                            f'<span style="color:{a_dir_color};font-weight:700">{a_dir}</span> '
                            f'<b>{a_sym}</b> · Target Close: <code>{c_str}</code> ({countdown_str}) · '
                            f'RRR: <code>1:{a.get("rrr", 1.5)}</code> · Risk: <code>{a.get("risk_value", 0.5)}%</code> · '
                            f'Buffer: <code>{a.get("sl_buffer_pips", 2.5)}p</code>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"Armed at: {a.get('created_at', '')[:19]} UTC · Pending stop order will be submitted right as candle closes.")
                    with c2:
                        if st.button("❌ Disarm", key=f"disarm_q_{a_id}", use_container_width=True):
                            disarm_order(a_id)
                            st.toast(f"Setup {a_sym} disarmed.", icon="ℹ️")
                            st.rerun()

        # 2. MT5 Active Pending & Open Orders
        if active_manual_orders:
            st.markdown("##### 📌 MT5 Active Orders & Positions")
            for o in active_manual_orders:
                badge_color = "#ffd600" if o["type"] == "PENDING" else "#00e676"
                dir_color   = "#00e676" if o["direction"] == "BUY" else "#ff4466"
                pnl_str = ""
                if o.get("pnl") is not None:
                    pnl_color = "#00e676" if o["pnl"] >= 0 else "#ff4466"
                    pnl_str = f'<span style="color:{pnl_color};font-weight:700"> · Floating P/L: ${o["pnl"]:.2f}</span>'
                with st.container(border=True):
                    row1, row2 = st.columns([5, 1])
                    with row1:
                        st.markdown(
                            f'<span style="padding:2px 8px;border-radius:6px;background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44;font-size:0.72rem;font-weight:700">{o["type"]}</span> '
                            f'<span style="color:{dir_color};font-weight:700"> {o["direction"]}</span> '
                            f'<b>{o["symbol"]}</b> · <code>{o["order_type_str"]}</code> · '
                            f'Entry: <code>{o["entry"]:.5f}</code> · SL: <code>{o["sl"]:.5f}</code> · TP: <code>{o["tp"]:.5f}</code> · '
                            f'Lots: <code>{o["lots"]}</code> · Ticket: <code>#{o["ticket"]}</code>{pnl_str}',
                            unsafe_allow_html=True)
                        st.caption(f"Placed: {o['placed_at'][:16]} UTC  ·  {o.get('comment', '')}")
                    with row2:
                        if o["type"] == "PENDING":
                            if st.button("❌ Cancel", key=f"cancel_{o['ticket']}", use_container_width=True):
                                r = cancel_manual_order(o["ticket"])
                                if r["success"]:
                                    st.toast(f"Order #{o['ticket']} cancelled.", icon="✅")
                                    st.rerun()
                                else:
                                    st.error(f"Cancel failed: {r.get('error')}")
                        elif o["type"] == "OPEN":
                            if st.button("🚨 Close", key=f"close_{o['ticket']}", use_container_width=True):
                                r = close_manual_position(o["ticket"])
                                if r["success"]:
                                    st.toast(f"Position #{o['ticket']} closed.", icon="✅")
                                    st.rerun()
                                else:
                                    st.error(f"Close failed: {r.get('error')}")
        elif not armed_setups:
            st.info("No active armed setups or pending M15 Sniper orders. Configure and arm a setup above.")

    # ── Recorded Manual Trades History (Audit Trail) ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        section_header("📜", "Recorded Manual Trades History (Audit Trail)")
        st.caption("Permanent record of all manual precision M15 trades recorded in SignalDatabase with resolved outcomes.")

        manual_trades = []
        try:
            import sqlite3
            with db._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, symbol, signal, price_at_signal, exit_price,
                           sl_price, tp_price, suggested_lots, outcome, exit_reason,
                           duration_seconds, mt5_ticket
                    FROM signals
                    WHERE is_manual = 1 OR model_version = 'manual_m15'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """)
                manual_trades = [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to query manual trade history: {e}")

        if manual_trades:
            # Summary KPIs for manual trading
            m_completed = [t for t in manual_trades if t.get("outcome") in ("SUCCESS", "FAIL")]
            m_wins = len([t for t in m_completed if t.get("outcome") == "SUCCESS"])
            m_wr = (m_wins / len(m_completed) * 100) if m_completed else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Recorded Trades", str(len(manual_trades)))
            k2.metric("Win Rate", f"{m_wr:.1f}%", f"{m_wins} wins / {len(m_completed)} closed")
            k3.metric("Resolved", str(len(m_completed)))
            k4.metric("Active / Pending", str(len(manual_trades) - len(m_completed)))

            # Table of recorded trades
            trade_rows = []
            for t in manual_trades:
                dur_m = (t.get("duration_seconds") or 0) // 60
                dur_str = f"{dur_m}m" if dur_m > 0 else "<1m"
                trade_rows.append({
                    "Timestamp": t.get("timestamp", "")[:19].replace("T", " "),
                    "Symbol": t.get("symbol"),
                    "Signal": t.get("signal"),
                    "Lots": t.get("suggested_lots"),
                    "Entry": f"{float(t.get('price_at_signal') or 0):.5f}",
                    "Exit": f"{float(t.get('exit_price') or 0):.5f}" if t.get("exit_price") else "—",
                    "TP": f"{float(t.get('tp_price') or 0):.5f}" if t.get("tp_price") else "—",
                    "SL": f"{float(t.get('sl_price') or 0):.5f}" if t.get("sl_price") else "—",
                    "Outcome": t.get("outcome"),
                    "Duration": dur_str,
                    "Ticket": f"#{t.get('mt5_ticket')}" if t.get("mt5_ticket") else "—",
                    "Reason": t.get("exit_reason") or "—"
                })

            st.dataframe(
                pd.DataFrame(trade_rows),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No past manual trades recorded in database yet. Trades will appear here permanently once armed or placed.")




def _show_dynamic_ytd_cockpit(symbol: str, all_pairs: list):
    """
    Dedicated cockpit for the 🌟 Dynamic YTD Model (Daily Winning Asset Strategy).
    Displays:
    - Real-time Winning Asset status for the selected symbol across each activated model
    - Global model-by-model winning assets breakdown
    - Daily 24h autonomous cycle countdown and status
    - Instant 'Recompute Whitelists Now' button
    - Fast pair switching and live MT5 execution controls
    """
    from core.dynamic_model_whitelist import get_dynamic_whitelist_manager, is_pair_whitelisted_for_model
    from core.model_gatekeeper import load_gatekeeper_config, save_gatekeeper_config
    
    dw_mgr = get_dynamic_whitelist_manager()
    summary = dw_mgr.get_all_models_summary()
    as_of_date = summary.get("as_of_date", "Today")
    models_data = summary.get("models", {})
    gate_cfg = load_gatekeeper_config()
    is_dyn_live = bool(gate_cfg.get("dynamic_ytd_model", True))

    # Header Card
    with st.container(border=True):
        hc1, hc2, hc3 = st.columns([2.8, 1.2, 1.0])
        with hc1:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;">'
                '<span style="font-size:1.8rem;">🌟</span>'
                '<div>'
                '<h3 style="margin:0;padding:0;font-size:1.35rem;font-weight:700;color:var(--text-primary);">'
                'Dynamic YTD Model (Daily Winning Asset Strategy)'
                '</h3>'
                '<span style="font-size:0.82rem;color:var(--text-secondary);">'
                'Autonomous 24h Cycle · Daily Winning Asset Whitelist Filter (Net R ≥ 0.0)'
                '</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with hc2:
            st.caption(f"📅 Active Whitelist Date: **{as_of_date}**")
            st.caption("🔄 Auto-Rollover: **00:01 UTC Daily**")
        with hc3:
            badge_html = '<div style="margin-top:4px;padding:6px 12px;background:rgba(255,214,0,0.15);border:1px solid #ffd60066;border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.82rem">🌟 LIVE ACTIVE</div>' if is_dyn_live else '<div style="margin-top:4px;padding:6px 12px;background:rgba(255,82,82,0.1);border:1px solid rgba(255,82,82,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ff5252;font-size:0.82rem">⏸️ DISABLED</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
            t_dyn = st.toggle("Live Execution", value=is_dyn_live, key="toggle_dyn_cockpit_live")
            if t_dyn != is_dyn_live:
                gate_cfg["dynamic_ytd_model"] = t_dyn
                save_gatekeeper_config(gate_cfg)
                action = "Activated" if t_dyn else "Deactivated"
                st.toast(f"🌟 Dynamic YTD Model {action}!", icon="🌟" if t_dyn else "⏸️")
                st.rerun()

    # Controls Row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1.5, 1.5])
    with ctrl_col1:
        st.markdown(f"**Asset Inspector: `{symbol}`**")
        st.caption("Inspect whether this asset is an approved winning asset for each active model.")
    with ctrl_col2:
        if st.button("🔄 Recompute Whitelists Now", key="btn_cockpit_recompute_whitelists", use_container_width=True):
            with st.spinner("Recomputing YTD winning assets across all models..."):
                dw_mgr.compute_ytd_whitelists()
                st.toast("✅ Dynamic Model Whitelists recomputed!", icon="🌟")
                st.rerun()
    with ctrl_col3:
        if st.button("⚡ Scan All Pairs", key="btn_cockpit_scan_all", use_container_width=True):
            with st.spinner("Scanning all pairs for winning setups..."):
                from core.confluence_model import scan_and_execute_all_pairs
                res = scan_and_execute_all_pairs()
                n_setups = len(res.get("setups", []))
                n_exec = len(res.get("executed", []))
                st.toast(f"Scan complete: {n_setups} setups detected, {n_exec} orders placed!", icon="🎯")
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Asset Status Card for the Selected Symbol
    with st.container(border=True):
        st.markdown(f"#### 🔍 Status for `{symbol}` Across All Activated Models")
        cols = st.columns(len(models_data) if models_data else 1)
        for i, (m_key, m_info) in enumerate(models_data.items()):
            w_pairs = m_info.get("winning_pairs", [])
            p_stats = m_info.get("pair_stats", {}).get(symbol, {})
            is_win = symbol in w_pairs
            net_r = p_stats.get("net_r", 0.0)
            wr = p_stats.get("win_rate", 0.0)
            trades = p_stats.get("trades", 0)
            short_name = m_info.get("name", m_key).split("(")[0].strip()

            with cols[i]:
                if is_win:
                    st.markdown(
                        f"<div style='padding:12px;border-radius:10px;background:rgba(0,230,118,0.08);border:1px solid #00e67644;text-align:center;'>"
                        f"<div style='font-weight:700;color:var(--text-primary);font-size:0.85rem;'>{short_name}</div>"
                        f"<div style='color:#00e676;font-size:1.1rem;font-weight:800;margin:4px 0;'>✅ APPROVED</div>"
                        f"<div style='font-size:0.75rem;color:var(--text-secondary);'>+{net_r:.2f}R · {wr:.0f}% WR</div>"
                        f"<div style='font-size:0.7rem;color:var(--text-secondary);'>{trades} closed trades</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='padding:12px;border-radius:10px;background:rgba(255,82,82,0.08);border:1px solid rgba(255,82,82,0.25);text-align:center;'>"
                        f"<div style='font-weight:700;color:var(--text-primary);font-size:0.85rem;'>{short_name}</div>"
                        f"<div style='color:#ff5252;font-size:1.1rem;font-weight:800;margin:4px 0;'>🚫 BENCHED</div>"
                        f"<div style='font-size:0.75rem;color:var(--text-secondary);'>{net_r:+.2f}R · {wr:.0f}% WR</div>"
                        f"<div style='font-size:0.7rem;color:var(--text-secondary);'>{trades} closed trades</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Model-by-Model Winning Assets Matrix
    st.markdown("#### 📋 Winning Assets per Activated Strategy Engine")
    for m_key, m_info in models_data.items():
        w_pairs = m_info.get("winning_pairs", [])
        b_pairs = m_info.get("benched_pairs", [])
        total_trades = m_info.get("total_trades_ytd", 0)
        m_name = m_info.get("name", m_key)
        
        from core.model_gatekeeper import is_model_live_authorized
        is_live = is_model_live_authorized(m_key)
        badge_str = "🟢 LIVE ACTIVE" if is_live else "👻 SHADOW MODE"
        badge_bg = "rgba(0,230,118,0.12);border:1px solid #00e67644;color:#00e676" if is_live else "rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);color:#ffd600"

        with st.container(border=True):
            mc1, mc2 = st.columns([3.8, 1.2])
            with mc1:
                st.markdown(f"**{m_name}** &nbsp; <span style='background:{badge_bg};font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;'>{badge_str}</span>", unsafe_allow_html=True)
                st.caption(f"YTD Closed Deals: **{total_trades}** | Approved Winning Assets: **{len(w_pairs)} pairs** | Benched: **{len(b_pairs)} pairs**")
            with mc2:
                tot_p = len(w_pairs) + len(b_pairs)
                pct_app = (len(w_pairs) / tot_p * 100) if tot_p > 0 else 0
                st.metric("Winning Assets", f"{len(w_pairs)} pairs", f"{pct_app:.0f}% Approved")

            if w_pairs:
                p_stats = m_info.get("pair_stats", {})
                badges_html = " ".join([
                    f'<span style="display:inline-block;margin:3px;padding:3px 10px;background:rgba(0,230,118,0.12);border:1px solid #00e67655;border-radius:12px;font-size:0.8rem;font-weight:700;color:#00e676;">'
                    f'✅ {p} <span style="font-weight:400;color:var(--text-secondary);font-size:0.75rem;">(+{p_stats.get(p, {}).get("net_r", 0.0):.1f}R · {p_stats.get(p, {}).get("win_rate", 0):.0f}% WR)</span>'
                    f'</span>'
                    for p in w_pairs
                ])
                st.markdown(f"**Winning Assets (Live Trading Authorized):**<br>{badges_html}", unsafe_allow_html=True)
            else:
                st.warning("No winning assets currently meet the hurdle for this model.")

            with st.expander(f"🔍 View Complete Per-Pair Performance Table ({len(w_pairs) + len(b_pairs)} pairs)"):
                p_stats = m_info.get("pair_stats", {})
                if p_stats:
                    rows = []
                    for p, p_data in p_stats.items():
                        rows.append({
                            "Symbol": p,
                            "Status": "✅ APPROVED" if p_data.get("status") == "APPROVED" else "🚫 BENCHED",
                            "Net R": p_data.get("net_r", 0.0),
                            "PnL ($)": p_data.get("pnl_usd", 0.0),
                            "Win Rate (%)": p_data.get("win_rate", 0.0),
                            "Trades": p_data.get("trades", 0),
                            "Record (W-L-BE)": f"{p_data.get('wins', 0)}W - {p_data.get('losses', 0)}L - {p_data.get('be', 0)}BE",
                        })
                    df_p = pd.DataFrame(rows).sort_values("Net R", ascending=False)
                    st.dataframe(
                        df_p,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Net R": st.column_config.NumberColumn("Realized Net R", format="%+.2f R"),
                            "PnL ($)": st.column_config.NumberColumn("Realized PnL", format="$%+.2f"),
                            "Win Rate (%)": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                        }
                    )


def _show_confluence_cockpit(symbol: str, all_pairs: list, is_ml_mode: bool = True, active_model_key: Optional[str] = None):
    """
    Automated Confluence Day/Swing Breakout & Reclamation Terminal ("Magic Candle Sniper").
    Full technical analysis suite with:
    - 21:00 UTC Previous Day High & Low Lines
    - Prior M15 Swing High & Low Lines
    - 3-Candle Institutional Liquidity Sweep & Reclamation state machine
    - Automatic Pending Stop Order execution on Master MT5 + Fan-out to Copy Trading Followers
    - Interactive Candlestick Charting with Day/Swing line overlays
    - Multi-Pair Radar Heatmap
    """
    try:
        import importlib
        import core.confluence_model
        try:
            importlib.reload(core.confluence_model)
        except Exception:
            pass
        from core.confluence_model import (
            get_day_lines,
            get_swing_lines,
            evaluate_confluence_setup,
            execute_confluence_setup,
            scan_and_execute_all_pairs,
            load_confluence_config,
            save_confluence_config,
            get_symbol_sl_buffer,
            get_symbol_be_offset,
            DEFAULT_SYMBOLS,
        )
        from core.manual_model import get_pip_size, get_mt5
    except ImportError as e:
        st.error(f"Confluence Model module not available: {e}")
        return

    engine = load_engine()
    inf_engine = load_inference_v2()
    db = get_db()
    conf_cfg = load_confluence_config()

    if not active_model_key:
        active_model_key = "confluence_ml_p60" if is_ml_mode else "confluence_std_p25"

    MODEL_INFO_MAP = {
        "confluence_ml_p60": {
            "flag": "enable_ml_p60_model",
            "name": "🧠 Confluence AI Quality Gate (P60 · 60% Partial + BE+2p)",
            "short_toggle": "🧠 AI Gate (P60) Active",
            "is_ml": True,
            "has_partial": True,
            "default_active": True,
        },
        "confluence_std_p25": {
            "flag": "enable_std_p25_model",
            "name": "⚡ Confluence Standard (P25 · 25% Partial + BE+2p)",
            "short_toggle": "⚡ Standard (P25) Active",
            "is_ml": False,
            "has_partial": True,
            "default_active": True,
        },
        "confluence_ml_m15": {
            "flag": "enable_ml_model",
            "name": "🧠 Confluence AI Gate (Original · Fixed 1.5R)",
            "short_toggle": "🧠 Original AI Gate Active",
            "is_ml": True,
            "has_partial": False,
            "default_active": False,
        },
        "confluence_m15": {
            "flag": "enable_standard_model",
            "name": "⚡ Confluence Standard (Original · Fixed 1.5R)",
            "short_toggle": "⚡ Original Standard Active",
            "is_ml": False,
            "has_partial": False,
            "default_active": False,
        },
    }
    model_meta = MODEL_INFO_MAP.get(active_model_key, MODEL_INFO_MAP["confluence_ml_p60"])
    model_flag = model_meta["flag"]
    model_display_name = model_meta["name"]
    toggle_label = model_meta["short_toggle"]

    # ── Top Bar: Pair Selector + Live Master Quote + Auto-Trade Status ─────────
    col_sym, col_quote, col_auto, col_scan = st.columns([1.6, 2.8, 1.8, 1.0])
    with col_sym:
        confluence_sym = st.selectbox(
            "Confluence Instrument", all_pairs,
            index=all_pairs.index(symbol) if symbol in all_pairs else 0,
            key="confluence_sym_selector",
            label_visibility="collapsed"
        )

    with col_quote:
        try:
            from core.mt5_connector import MT5Connector
            _mt5_conn = MT5Connector().get_connection()
            if _mt5_conn is None:
                import MetaTrader5 as _mt5_conn
                _mt5_conn.initialize()
            _si = _mt5_conn.symbol_info(confluence_sym)
            _acc = _mt5_conn.account_info()
            if _si and _acc:
                _sp_pips = _si.spread * _si.point / get_pip_size(confluence_sym)
                _login_str = f"#{_acc.login}" if hasattr(_acc, 'login') else ""
                st.markdown(
                    f'<div style="padding:6px 14px;background:rgba(255,214,0,0.05);border:1px solid rgba(255,214,0,0.2);border-radius:8px;display:flex;gap:16px;align-items:center">'
                    f'<span style="font-family:monospace;font-size:0.92rem;font-weight:700;color:#29b6f6">Bid: {_si.bid:.5f}</span>'
                    f'<span style="font-family:monospace;font-size:0.92rem;font-weight:700;color:#00e676">Ask: {_si.ask:.5f}</span>'
                    f'<span style="font-size:0.75rem;color:var(--text-secondary)">Spread: {_sp_pips:.1f}p</span>'
                    f'<span style="font-size:0.75rem;color:var(--text-secondary)">Master {_login_str}: <b>${_acc.balance:,.2f}</b></span>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.caption("Awaiting MT5 quotes...")
        except Exception:
            st.caption("Connect MT5 for live quotes.")

    with col_auto:
        is_model_active = bool(conf_cfg.get(model_flag, model_meta["default_active"]))
        if st.toggle(toggle_label, value=is_model_active, key=f"toggle_active_{active_model_key}"):
            if not is_model_active:
                conf_cfg[model_flag] = True
                save_confluence_config(conf_cfg)
                try:
                    from core.model_gatekeeper import load_gatekeeper_config, save_gatekeeper_config
                    g_cfg = load_gatekeeper_config()
                    g_cfg[active_model_key] = True
                    save_gatekeeper_config(g_cfg)
                except Exception:
                    pass
                st.toast(f"🟢 {model_display_name} activated!", icon="🟢")
                st.rerun()
        else:
            if is_model_active:
                conf_cfg[model_flag] = False
                save_confluence_config(conf_cfg)
                try:
                    from core.model_gatekeeper import load_gatekeeper_config, save_gatekeeper_config
                    g_cfg = load_gatekeeper_config()
                    g_cfg[active_model_key] = False
                    save_gatekeeper_config(g_cfg)
                except Exception:
                    pass
                st.toast(f"⏸️ {model_display_name} deactivated!", icon="🔴")
                st.rerun()

    with col_scan:
        if st.button("⚡ Scan All", key="btn_scan_all_confluence", use_container_width=True, help="Scan all watchlist pairs immediately for confluence setups"):
            with st.spinner("Scanning pairs for Confluence setups..."):
                scan_res = scan_and_execute_all_pairs()
                n_setups = len(scan_res.get("setups", []))
                n_exec = len(scan_res.get("executed", []))
                st.toast(f"Scan complete: {n_setups} setups detected, {n_exec} orders placed!", icon="🎯")
                time.sleep(0.5)
                st.rerun()

    # If currently viewed model is deactivated, display clear status alert
    if not is_model_active:
        st.warning(
            f"⏸️ **{model_display_name} is currently DEACTIVATED.** "
            f"The background daemon will NOT execute trades for this engine. Flip the toggle above to activate.",
            icon="🔴"
        )

    # ── Configuration Parameters ─────────────────────────────────────────────
    sl_buffer = get_symbol_sl_buffer(confluence_sym, conf_cfg)
    rrr = float(conf_cfg.get("rrr", 1.5))
    risk_type = conf_cfg.get("risk_type", "percent")
    risk_value = float(conf_cfg.get("risk_value", 0.5))

    # ── Evaluate Confluence Setup for Selected Symbol ────────────────────────
    try:
        setup_info = evaluate_confluence_setup(
            symbol=confluence_sym,
            rrr=rrr,
            sl_buffer_pips=sl_buffer,
            risk_type=risk_type,
            risk_value=risk_value,
            active_model=active_model_key,
        )
    except TypeError:
        setup_info = evaluate_confluence_setup(
            symbol=confluence_sym,
            rrr=rrr,
            sl_buffer_pips=sl_buffer,
            risk_type=risk_type,
            risk_value=risk_value,
        )
        if isinstance(setup_info, dict):
            setup_info["model_version"] = active_model_key

    pip_sz = get_pip_size(confluence_sym)
    day_lines = setup_info.get("day_lines") or {}
    swing_lines = setup_info.get("swing_lines") or {}
    upper_day = day_lines.get("upper_day_line")
    lower_day = day_lines.get("lower_day_line")
    upper_swing = swing_lines.get("upper_swing_line")
    lower_swing = swing_lines.get("lower_swing_line")

    # Fetch live price for distance calculations
    last_price = 0.0
    mt5_inst = get_mt5()
    if mt5_inst:
        try:
            t = mt5_inst.symbol_info_tick(confluence_sym)
            if t:
                last_price = float(t.bid)
        except Exception:
            pass

    # ── High-Visibility Status Banner ─────────────────────────────────────────
    if setup_info.get("setup_detected"):
        dir_str = setup_info["direction"]
        dir_color = "#00FF88" if dir_str == "BUY" else "#FF4466"
        entry_val = setup_info["entry"]
        sl_val = setup_info["sl"]
        tp_val = setup_info["tp"]
        line_lbl = setup_info.get("line_type", "Reference Line")
        reclaimed_lvl = setup_info.get("reclaimed_line", 0.0)
        lots_val = setup_info.get("lots", 0.01)
        sl_p = setup_info.get("sl_pips", 0.0)
        tp_p = setup_info.get("tp_pips", 0.0)

        part_p = setup_info.get("partial_target_price")
        part_r = setup_info.get("partial_target_ratio")
        if part_p and part_r:
            part_badge = f"<span style='font-size:0.8rem;color:#FFD600'>🎯 Part TP: {part_p:.5f} ({int(round(part_r * 100))}% · BE+2p)</span>"
        else:
            part_badge = f"<span style='font-size:0.8rem;color:#29b6f6'>🏁 Fixed 1:{rrr} R:R (Full TP/SL · Single-Asset)</span>"

        st.markdown(f"""
        <div style="background:rgba(255,214,0,0.08);border:1px solid #ffd600;padding:12px 18px;border-radius:10px;margin:10px 0 14px 0;box-shadow:0 0 16px rgba(255,214,0,0.15)">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
                    <span style="background:#FFD600;color:#000;padding:4px 12px;border-radius:6px;font-family:'Inter',sans-serif;font-size:0.75rem;font-weight:900;letter-spacing:0.05em">SETUP CONFIRMED</span>
                    <span style="font-weight:800;color:#ffffff;font-size:1.0rem">🎯 {dir_str}_STOP ARMED · {confluence_sym}</span>
                    <span style="color:{dir_color};font-weight:700;font-size:0.95rem">{dir_str} {lots_val} Lots @ {entry_val:.5f}</span>
                    <span style="font-size:0.78rem;color:var(--text-secondary)">Reclaimed {line_lbl} ({reclaimed_lvl:.5f})</span>
                </div>
                <div style="display:flex;gap:12px;align-items:center">
                    {part_badge}
                    <span style="font-size:0.8rem;color:#00FF88">TP: {tp_val:.5f} (+{tp_p:.1f}p) [1:{rrr}]</span>
                    <span style="font-size:0.8rem;color:#FF4466">SL: {sl_val:.5f} (-{sl_p:.1f}p · {sl_buffer:.1f}p buf)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(0,230,118,0.04);border:1px solid rgba(0,230,118,0.2);padding:10px 16px;border-radius:8px;margin:8px 0 12px 0;display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:0.82rem;color:var(--text-secondary)">
                🟢 <b>Confluence Engine Active</b>: Monitoring {confluence_sym} for 21:00 UTC Day Lines & M15 Swing Breakout/Reclamation.
            </span>
            <span style="font-size:0.75rem;color:var(--text-muted);font-family:monospace">
                Roll: 21:00 UTC &middot; Buffer: {sl_buffer:.1f}p &middot; R:R: 1:{rrr}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Reference Lines Radar (4 Metric Cards) ────────────────────────────────
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        if upper_day:
            dist_pips = (upper_day - last_price) / pip_sz if last_price > 0 else 0.0
            st.metric("Upper Day Line (21:00 UTC)", f"{upper_day:.5f}", f"{dist_pips:+.1f} pips away")
        else:
            st.metric("Upper Day Line", "—", "Awaiting Data")
    with c_m2:
        if lower_day:
            dist_pips = (lower_day - last_price) / pip_sz if last_price > 0 else 0.0
            st.metric("Lower Day Line (21:00 UTC)", f"{lower_day:.5f}", f"{dist_pips:+.1f} pips away")
        else:
            st.metric("Lower Day Line", "—", "Awaiting Data")
    with c_m3:
        if upper_swing:
            dist_pips = (upper_swing - last_price) / pip_sz if last_price > 0 else 0.0
            st.metric("Upper Swing Line (Prior High)", f"{upper_swing:.5f}", f"{dist_pips:+.1f} pips away")
        else:
            st.metric("Upper Swing Line", "None Found", "Market at Highs")
    with c_m4:
        if lower_swing:
            dist_pips = (lower_swing - last_price) / pip_sz if last_price > 0 else 0.0
            st.metric("Lower Swing Line (Prior Low)", f"{lower_swing:.5f}", f"{dist_pips:+.1f} pips away")
        else:
            st.metric("Lower Swing Line", "None Found", "Market at Lows")

    # ── Main Two-Column Layout (Chart & State Machine) ────────────────────────
    col_chart, col_state = st.columns([3.2, 1.8])

    with col_chart:
        # Fetch Candlestick Chart Data
        df_chart = pd.DataFrame()
        try:
            df_chart = inf_engine.data_engine.fetch(confluence_sym, interval="15m", days=4, use_cache=False)
        except Exception as e:
            logger.warning(f"Could not fetch {confluence_sym} data for confluence chart: {e}")

        # Construct Chart Levels
        chart_lvls = {
            "upper_day": upper_day,
            "lower_day": lower_day,
            "upper_swing": upper_swing,
            "lower_swing": lower_swing,
        }
        if setup_info.get("setup_detected"):
            chart_lvls["entry"] = setup_info.get("entry")
            chart_lvls["sl"] = setup_info.get("sl")
            chart_lvls["tp"] = setup_info.get("tp")

        # Render Lightweight Candlestick Chart
        render_chart(df_chart, confluence_sym, key=f"conf_m15_{confluence_sym.lower()}", levels=chart_lvls)

    with col_state:
        st.markdown("#### 🎯 3-Candle Confluence Tracker")
        st.caption("Institutional Liquidity Sweep & Reclamation Sequence")

        c1 = setup_info.get("candle_1")
        c2 = setup_info.get("magic_candle")
        c3 = setup_info.get("entry_candle")

        # Candle 1 Card
        with st.container(border=True):
            st.markdown("**1️⃣ Candle 1: Breakout Candle**")
            if c1:
                c1_t = c1["time"].strftime("%H:%M UTC") if hasattr(c1["time"], "strftime") else str(c1["time"])[:16]
                c1_color = "🟢 BULLISH" if c1["is_bullish"] else "🔴 BEARISH"
                st.caption(f"Time: `{c1_t}` &middot; Type: **{c1_color}**")
                st.markdown(f"`O: {c1['open']:.5f} | H: {c1['high']:.5f} | L: {c1['low']:.5f} | C: {c1['close']:.5f}`")
            else:
                st.caption("Awaiting M15 candle data...")

        # Candle 2 Card (Magic Candle)
        with st.container(border=True):
            st.markdown("**2️⃣ Candle 2: Magic Candle (Reclamation)**")
            if c2:
                c2_t = c2["time"].strftime("%H:%M UTC") if hasattr(c2["time"], "strftime") else str(c2["time"])[:16]
                c2_color = "🟢 BULLISH" if c2["is_bullish"] else "🔴 BEARISH"
                st.caption(f"Time: `{c2_t}` &middot; Type: **{c2_color}**")
                st.markdown(f"`O: {c2['open']:.5f} | H: {c2['high']:.5f} | L: {c2['low']:.5f} | C: {c2['close']:.5f}`")
                if setup_info.get("setup_detected"):
                    st.success(f"✨ Reclaimed **{setup_info['line_type']}** (`{setup_info['reclaimed_line']:.5f}`)!")
                else:
                    st.caption("No reclamation cross of Day/Swing line.")
            else:
                st.caption("Awaiting M15 candle data...")

        # Candle 3 Card (Entry Candle)
        with st.container(border=True):
            st.markdown("**3️⃣ Candle 3: Entry Candle (Wick Touch)**")
            if c3:
                c3_t = c3["time"].strftime("%H:%M UTC") if hasattr(c3["time"], "strftime") else str(c3["time"])[:16]
                c3_exp = c3["close_time"].strftime("%H:%M UTC") if hasattr(c3["close_time"], "strftime") else str(c3["close_time"])[:16]
                st.caption(f"Current Forming: `{c3_t}` &middot; Window Closes: `{c3_exp}`")
                if setup_info.get("setup_detected"):
                    ml_data = setup_info.get("ml_filter", {})
                    if ml_data.get("evaluated"):
                        p_score = ml_data.get("confidence_score_pct", 50.0)
                        passed_gate = ml_data.get("passed", True)
                        badge_color = "#10b981" if passed_gate else "#ef4444"
                        status_txt = "PASSED QUALITY GATE" if passed_gate else "SUPPRESSED (BELOW GATE)"
                        st.markdown(
                            f"<div style='background-color:rgba(255,255,255,0.05);padding:8px 12px;border-radius:6px;border-left:4px solid {badge_color};margin-bottom:8px;'>"
                            f"<span style='color:{badge_color};font-weight:700;'>🤖 ML Win Probability: {p_score}%</span> &middot; <span style='font-size:0.85rem;color:#ccc;'>{status_txt}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    actual_buf = setup_info.get("sl_buffer_pips", sl_buffer)
                    be_p = setup_info.get("be_offset_pips", 2.0)
                    st.markdown(f"**Target Wick Entry**: `{setup_info['entry']:.5f}`")
                    st.markdown(f"**Stop Loss**: `{setup_info['sl']:.5f}` (`{actual_buf:.1f}p` buffer)")
                    st.markdown(f"**Take Profit**: `{setup_info['tp']:.5f}` (`1:{rrr}` R:R)")
                    if setup_info.get("partial_target_price"):
                        p_ratio_pct = int(round(setup_info.get("partial_target_ratio", 0.5) * 100))
                        st.markdown(f"**Partial Profit**: `{setup_info['partial_target_price']:.5f}` ({p_ratio_pct}% of TP · moves SL to BE+{be_p:.1f}p)")
                    else:
                        st.markdown(f"**Profit Target**: Full Target (Fixed 1:{rrr} R:R · No Partial)")

                    if st.button("⚡ Execute Pending Order Now", key="btn_exec_conf_now", type="primary", use_container_width=True):
                        with st.spinner("Submitting Pending Stop Order to MT5 Master & Followers..."):
                            exec_res = execute_confluence_setup(setup_info)
                            if exec_res.get("success"):
                                st.success(f"✅ Order Placed! Ticket #{exec_res.get('ticket')}")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(f"Execution Error: {exec_res.get('error')}")
                else:
                    st.caption("Waiting for active confluence setup...")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Watchlist Radar Heatmap Table ─────────────────────────────────────────
    section_header("📡", "Confluence Multi-Pair Watchlist Radar")
    st.caption("Real-time surveillance across major pairs for 21:00 UTC Day Lines and Swing reclamations.")

    watch_symbols = conf_cfg.get("symbols", DEFAULT_SYMBOLS)
    radar_rows = []

    for s in watch_symbols:
        try:
            try:
                s_setup = evaluate_confluence_setup(s, rrr=rrr, sl_buffer_pips=None, risk_value=risk_value, active_model=active_model_key)
            except TypeError:
                s_setup = evaluate_confluence_setup(s, rrr=rrr, sl_buffer_pips=None, risk_value=risk_value)
                if isinstance(s_setup, dict):
                    s_setup["model_version"] = active_model_key
            s_day = s_setup.get("day_lines") or {}
            s_swing = s_setup.get("swing_lines") or {}

            s_up_d = s_day.get("upper_day_line")
            s_lo_d = s_day.get("lower_day_line")
            s_up_s = s_swing.get("upper_swing_line")
            s_lo_s = s_swing.get("lower_swing_line")

            # Status label
            if s_setup.get("setup_detected"):
                s_dir = s_setup.get("direction")
                status_lbl = f"🟢 BUY SETUP CONFIRMED" if s_dir == "BUY" else f"🔴 SELL SETUP CONFIRMED"
                ml_s = s_setup.get("ml_filter", {})
                ml_col_val = f"{ml_s.get('confidence_score_pct')}%" if ml_s.get("evaluated") else "—"
            else:
                status_lbl = "⚪ MONITORING"
                ml_col_val = "—"

            s_buf_val = s_setup.get("sl_buffer_pips", get_symbol_sl_buffer(s, conf_cfg))
            radar_rows.append({
                "Symbol": s,
                "SL Buffer": f"{s_buf_val:.1f}p",
                "Upper Day Line": f"{s_up_d:.5f}" if s_up_d else "—",
                "Lower Day Line": f"{s_lo_d:.5f}" if s_lo_d else "—",
                "Upper Swing Line": f"{s_up_s:.5f}" if s_up_s else "—",
                "Lower Swing Line": f"{s_lo_s:.5f}" if s_lo_s else "—",
                "Signal State": status_lbl,
                "ML Quality": ml_col_val,
                "Action": "Ready to Trade" if s_setup.get("setup_detected") else "Watching"
            })
        except Exception:
            pass

    if radar_rows:
        st.dataframe(pd.DataFrame(radar_rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical Confluence Trades Table ────────────────────────────────────
    section_header("📜", "Automated Confluence Trade History & Signals")
    recent_signals = db.get_recent_signals(limit=50, include_hidden=True)
    conf_trades = [s for s in recent_signals if str(s.get("model_version", "")).startswith("confluence_")]

    if conf_trades:
        trade_rows = []
        for t in conf_trades:
            ts_str = str(t.get("timestamp", ""))[:16].replace("T", " ")
            trade_rows.append({
                "Time (UTC)": ts_str,
                "Symbol": t.get("symbol"),
                "Signal": t.get("signal"),
                "Entry": f"{float(t.get('price_at_signal', 0)):.5f}",
                "SL": f"{float(t.get('sl_price', 0)):.5f}",
                "TP": f"{float(t.get('tp_price', 0)):.5f}",
                "Lots": t.get("suggested_lots"),
                "Ticket": f"#{t.get('mt5_ticket')}" if t.get('mt5_ticket') else "—",
                "Outcome": t.get("outcome", "ACTIVE"),
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No Confluence trades executed yet. Automated trades will appear here as soon as setups confirm.")

    # ── Strategy Parameters Expander ──────────────────────────────────────────
    with st.expander("⚙️ Confluence Engine Settings & Tuning", expanded=False):
        c_set1, c_set2, c_set3 = st.columns(3)
        with c_set1:
            new_buffer = st.number_input(
                "Default SL Buffer (pips beyond magic wick)",
                min_value=0.5, max_value=25.0, value=float(sl_buffer), step=0.5,
                key="conf_set_buffer"
            )
        with c_set2:
            new_rrr = st.number_input(
                "Default Take-Profit R:R Ratio",
                min_value=1.0, max_value=5.0, value=float(rrr), step=0.25,
                key="conf_set_rrr"
            )
        with c_set3:
            new_risk = st.number_input(
                "Risk % per Trade",
                min_value=0.1, max_value=5.0, value=float(risk_value), step=0.1,
                key="conf_set_risk"
            )

        inc_wicks = st.toggle(
            "Include Candle Wicks for Line Interaction",
            value=bool(conf_cfg.get("include_wicks", True)),
            help="When enabled, candle wicks touching or piercing Day/Swing lines count as valid line interactions alongside candle bodies.",
            key="conf_set_include_wicks"
        )

        new_syms = st.multiselect(
            "Monitored Watchlist Pairs",
            options=all_pairs,
            default=[s for s in watch_symbols if s in all_pairs],
            key="conf_set_syms"
        )

        if st.button("💾 Save Confluence Settings", key="btn_save_conf_settings"):
            conf_cfg["sl_buffer_pips"] = float(new_buffer)
            conf_cfg["rrr"] = float(new_rrr)
            conf_cfg["risk_value"] = float(new_risk)
            conf_cfg["include_wicks"] = bool(inc_wicks)
            conf_cfg["symbols"] = new_syms
            save_confluence_config(conf_cfg)
            st.toast("✅ Confluence Model settings saved successfully!", icon="💾")
            time.sleep(0.5)
            st.rerun()




@st.cache_data(ttl=120, show_spinner=False)
def _get_cached_model_comparison(risk_val: float, start_date_val: Optional[str], end_date_val: Optional[str]) -> pd.DataFrame:
    from core.performance_report import PerformanceReporter
    return PerformanceReporter().get_model_comparison_breakdown(
        risk_per_trade=risk_val,
        start_date=start_date_val,
        end_date=end_date_val
    )

@st.cache_data(ttl=120, show_spinner=False)
def _get_cached_performance_matrix(period: str, mode_key: str, risk_val: float, start_date_val: Optional[str], end_date_val: Optional[str], account_size: float) -> pd.DataFrame:
    from core.performance_report import PerformanceReporter
    return PerformanceReporter().get_performance_matrix(
        period=period,
        mode=mode_key,
        risk_per_trade=risk_val,
        start_date=start_date_val,
        end_date=end_date_val,
        account_size=account_size,
        use_close_time=True
    )


def render_periodic_performance_matrix():
    section_header("📈", "Performance & Return Matrix (Weekly & Monthly)")
    
    try:
        from core.performance_report import PerformanceReporter
        reporter = PerformanceReporter()
    except Exception as e:
        st.error(f"Failed to load PerformanceReporter: {e}")
        return

    import re
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2.2, 1.4, 1.4, 1.0])
    with col_ctrl1:
        mode_opt = st.selectbox(
            "Evaluation Policy",
            [
                "🌟 Dynamic YTD Model (Daily Winning Assets · Live MT5 Fills)",
                "🌟 Dynamic YTD Model (Daily Winning Assets · All: Live + Shadow)",
                "🧠 Confluence ML P60 (Live Fills Only · 60% TP + 2p BE)",
                "⚡ Confluence Standard P25 (Live Fills Only · 25% TP + 2p BE)",
                "🧠 Confluence ML P60 (All: Live + Shadow · 60% TP + 2p BE)",
                "⚡ Confluence Standard P25 (All: Live + Shadow · 25% TP + 2p BE)",
                "🧠 Confluence M15 AI Quality Gate (Live Fills Only · Fixed 1.5R)",
                "⚡ Confluence M15 Standard (Live Fills Only · Fixed 1.5R)",
                "🎯 Combined Confluence Live (All Live Wick Models)",
                "🏦 Master MT5 Executed Trades (All Live Broker Fills)",
                "🏆 Aggregate: Selected Live Models Only",
                "🌟 Aggregate: All Strategy Models (Live + Shadow)",
                "🧠 Confluence AI Quality Gate (All: Live + Shadow · Fixed 1.5R)",
                "⚡ Confluence Standard Rule-Based (All: Live + Shadow · Fixed 1.5R)",
                "🛡️ Confluence Suppressed Trades (ML Filter Blocked)",
                "🌐 Foundation V1 Macro AI (All: Live + Shadow)",
                "🎯 Manual M15 Wick Sniper Trades (Executed Fills)",
                "🎯 Live Production Policy (61%+ Forex / 55%+ Commodities)",
                "📲 Live Telegram Alerts",
                "📊 All 50.0%+ Baseline Signals"
            ],
            key="perf_matrix_policy_mode"
        )
    with col_ctrl2:
        risk_opt = st.selectbox(
            "Account / Base Risk",
            [
                "$50 (0.5% on $10k)",
                "$100 (1.0% on $10k)",
                "$500 (0.5% on $100k Prop)"
            ],
            key="perf_matrix_risk_mode"
        )
    # Dynamic Month Definitions
    now_dt = datetime.now()
    cur_month_name = now_dt.strftime("%B %Y")
    cur_month_start = now_dt.strftime("%Y-%m-01")
    first_of_cur = now_dt.replace(day=1)
    last_day_prev = first_of_cur - timedelta(days=1)
    prev_month_name = last_day_prev.strftime("%B %Y")
    prev_month_start = last_day_prev.strftime("%Y-%m-01")
    prev_month_end = last_day_prev.strftime("%Y-%m-%d 23:59:59")

    with col_ctrl3:
        timeframe_opt = st.selectbox(
            "Timeframe Scope",
            [
                "📅 All Active (Aug 2026 – Present)",
                f"🗓️ Current Month ({cur_month_name})",
                f"📜 Previous Month ({prev_month_name})",
                "🌐 Full History (All Data)"
            ],
            key="perf_matrix_timeframe"
        )
    with col_ctrl4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        send_tg = st.button("📲 Telegram", key="perf_matrix_tg_btn", use_container_width=True, help="Send performance scorecard to Telegram")

    # Accurate Risk & Account Size Parsing (handles $50, $100, $500 without substring collisions)
    m_risk = re.search(r'\$(\d+)', risk_opt)
    risk_val = float(m_risk.group(1)) if m_risk else 50.0
    account_size = 100000.0 if "100k" in risk_opt else 10000.0

    # Policy Key Mapping
    if "Dynamic YTD Model" in mode_opt and "Live MT5" in mode_opt:
        mode_key = "dynamic_ytd_live"
    elif "Dynamic YTD Model" in mode_opt:
        mode_key = "dynamic_ytd_all"
    elif "Confluence ML P60" in mode_opt and "Live Fills" in mode_opt:
        mode_key = "confluence_ml_p60_live"
    elif "Confluence ML P60" in mode_opt:
        mode_key = "confluence_ml_p60"
    elif "Confluence Standard P25" in mode_opt and "Live Fills" in mode_opt:
        mode_key = "confluence_std_p25_live"
    elif "Confluence Standard P25" in mode_opt:
        mode_key = "confluence_std_p25"
    elif "Confluence M15 AI Quality Gate (Live" in mode_opt:
        mode_key = "confluence_ml"
    elif "Confluence M15 Standard (Live" in mode_opt:
        mode_key = "confluence_standard"
    elif "Combined Confluence Live" in mode_opt or "Automated Confluence (Live" in mode_opt:
        mode_key = "confluence"
    elif "Master MT5" in mode_opt:
        mode_key = "mt5_live"
    elif "Aggregate: Selected Live" in mode_opt:
        mode_key = "aggregate_live"
    elif "Aggregate: All Strategy Models" in mode_opt:
        mode_key = "aggregate_all"
    elif "Confluence AI Quality Gate (All" in mode_opt:
        mode_key = "confluence_ml_all"
    elif "Confluence Standard Rule-Based (All" in mode_opt:
        mode_key = "confluence_standard_all"
    elif "Suppressed" in mode_opt:
        mode_key = "confluence_ml_suppressed"
    elif "Foundation V1" in mode_opt:
        mode_key = "foundation_all"
    elif "All Traded Models" in mode_opt:
        mode_key = "all_models"
    elif "Manual" in mode_opt:
        mode_key = "manual"
    elif "Telegram" in mode_opt:
        mode_key = "telegram_live"
    elif "Production" in mode_opt:
        mode_key = "production"
    else:
        mode_key = "baseline"

    # Timeframe Range Mapping
    if "Current Month" in timeframe_opt:
        start_date_val = cur_month_start
        end_date_val = None
    elif "Previous Month" in timeframe_opt:
        start_date_val = prev_month_start
        end_date_val = prev_month_end
    elif "All Active" in timeframe_opt:
        start_date_val = "2026-08-01"
        end_date_val = None
    else: # Full History
        start_date_val = None
        end_date_val = None

    if send_tg:
        try:
            from core.notifications import NotificationManager
            notif = NotificationManager()
            if notif.send_periodic_performance_report(
                period="both",
                risk_per_trade=risk_val,
                mode=mode_key,
                start_date=start_date_val,
                end_date=end_date_val,
                account_size=account_size
            ):
                st.toast("✅ Scorecard dispatched to Telegram!", icon="🚀")
                st.success("Scorecard sent to Telegram!")
            else:
                st.warning("Telegram disabled or failed to dispatch. Check config.yaml.")
        except Exception as ex:
            st.error(f"Telegram error: {ex}")

    t_comp, t_month, t_week = st.tabs(["⚖️ Model Comparison (Active vs Shadow)", "🗓️ Monthly Performance", "📅 Weekly Performance"])

    with t_comp:
        st.markdown("##### ⚖️ Multi-Model Performance Matrix (Active vs Background Shadow)")
        st.caption("Compare live broker executions against non-active models and ML quality-gated suppressed trades running continuously in shadow mode.")
        try:
            df_comp = _get_cached_model_comparison(risk_val, start_date_val, end_date_val)
            if not df_comp.empty:
                st.dataframe(
                    df_comp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model Engine": st.column_config.TextColumn("Model Engine", width="large"),
                        "Execution Mode": "Mode",
                        "Active Setups": st.column_config.NumberColumn("Active Armed", format="%d"),
                        "Total Trades": st.column_config.NumberColumn("Closed Deals", format="%d"),
                        "Record (W-L)": "Record",
                        "Win Rate (%)": "Win Rate",
                        "Net Edge (R)": "Net Edge",
                        "Net PnL ($)": "Net PnL",
                        "Profit Factor": "Profit Factor",
                    }
                )
            else:
                st.info("No comparative model data available for this timeframe scope.")
        except Exception as ex:
            st.warning(f"Could not load model comparison: {ex}")

    with t_month:
        try:
            df_m = _get_cached_performance_matrix(
                period="monthly",
                mode_key=mode_key,
                risk_val=risk_val,
                start_date_val=start_date_val,
                end_date_val=end_date_val,
                account_size=account_size
            )
            if not df_m.empty:
                # Highlight Current / Active Month
                curr = df_m.iloc[0]
                curr_period = str(curr['Period'])
                curr_t = int(curr['Trades'])
                curr_w = int(curr['Wins'])
                curr_l = int(curr['Losses'])
                curr_be = int(curr.get('Breakeven', 0))
                curr_wr = float(curr['Win Rate (%)'])
                curr_r = float(curr['Net R'])
                curr_pnl = float(curr['Net PnL ($)'])
                curr_ret = float(curr['Return (%)'])

                st.markdown(f"##### 🗓️ Active Month Performance: **{curr_period}**")
                m1, m2, m3, m4 = st.columns(4)
                record_str = f"{curr_w}W – {curr_l}L" + (f" – {curr_be}BE" if curr_be > 0 else "")
                m1.metric("Month Closed Setups", f"{curr_t}", record_str)
                m2.metric("Month Win Rate", f"{curr_wr:.1f}%", f"{curr_wr-40.0:+.1f}% vs BE")
                m3.metric("Month Realized Edge", f"{curr_r:+.2f}R", "1:1.5 RRR Target")
                m4.metric("Month Net Realized PnL", f"${curr_pnl:+,.2f}", f"{curr_ret:+.2f}% on ${account_size:,.0f}")

                # If viewing multi-month scope, show period aggregate toggle
                if len(df_m) > 1:
                    with st.expander("📊 View Cumulative Scope Totals", expanded=False):
                        tot_trades = int(df_m['Trades'].sum())
                        tot_wins = int(df_m['Wins'].sum())
                        tot_losses = int(df_m['Losses'].sum())
                        tot_be = int(df_m['Breakeven'].sum()) if 'Breakeven' in df_m.columns else 0
                        tot_pnl = float(df_m['Net PnL ($)'].sum())
                        tot_r = float(df_m['Net R'].sum())
                        tot_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
                        tot_ret = (tot_pnl / account_size) * 100.0

                        c1, c2, c3, c4 = st.columns(4)
                        tot_rec = f"{tot_wins}W – {tot_losses}L" + (f" – {tot_be}BE" if tot_be > 0 else "")
                        c1.metric("Total Scope Trades", f"{tot_trades}", tot_rec)
                        c2.metric("Overall Win Rate", f"{tot_wr:.1f}%", f"{tot_wr-40.0:+.1f}% vs BE")
                        c3.metric("Cumulative Edge", f"{tot_r:+.2f}R", "All Months")
                        c4.metric("Total Realized Profit", f"${tot_pnl:+,.2f}", f"{tot_ret:+.2f}% on ${account_size:,.0f}")

                st.dataframe(
                    df_m,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Period": "Month",
                        "Trades": st.column_config.NumberColumn("Setups", format="%d"),
                        "Wins": st.column_config.NumberColumn("Wins", format="%d"),
                        "Losses": st.column_config.NumberColumn("Losses", format="%d"),
                        "Breakeven": st.column_config.NumberColumn("BE", format="%d"),
                        "Win Rate (%)": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                        "Net R": st.column_config.NumberColumn("Realized R", format="%+.2fR"),
                        "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                        "Net PnL ($)": st.column_config.NumberColumn("Net PnL ($)", format="$%+.2f"),
                        "Return (%)": st.column_config.NumberColumn("Return (%)", format="%+.2f%%")
                    }
                )
            else:
                st.info("No data available for selected monthly policy and timeframe.")
        except Exception as ex:
            st.warning(f"Could not load monthly performance: {ex}")

    with t_week:
        try:
            df_w = _get_cached_performance_matrix(
                period="weekly",
                mode_key=mode_key,
                risk_val=risk_val,
                start_date_val=start_date_val,
                end_date_val=end_date_val,
                account_size=account_size
            )
            if not df_w.empty:
                # Highlight Current / Active Week
                curr_w = df_w.iloc[0]
                curr_w_period = str(curr_w['Period'])
                w_trades = int(curr_w['Trades'])
                w_wins = int(curr_w['Wins'])
                w_losses = int(curr_w['Losses'])
                w_be = int(curr_w.get('Breakeven', 0))
                w_wr = float(curr_w['Win Rate (%)'])
                w_r = float(curr_w['Net R'])
                w_pnl = float(curr_w['Net PnL ($)'])
                w_ret = float(curr_w['Return (%)'])

                st.markdown(f"##### 📅 Active Week Performance: **{curr_w_period}**")
                wm1, wm2, wm3, wm4 = st.columns(4)
                w_rec_str = f"{w_wins}W – {w_losses}L" + (f" – {w_be}BE" if w_be > 0 else "")
                wm1.metric("Week Closed Setups", f"{w_trades}", w_rec_str)
                wm2.metric("Week Win Rate", f"{w_wr:.1f}%", f"{w_wr-40.0:+.1f}% vs BE")
                wm3.metric("Week Realized Edge", f"{w_r:+.2f}R", "1:1.5 RRR Target")
                wm4.metric("Week Net Realized PnL", f"${w_pnl:+,.2f}", f"{w_ret:+.2f}% on ${account_size:,.0f}")

                st.dataframe(
                    df_w,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Period": "Week (UTC)",
                        "Trades": st.column_config.NumberColumn("Setups", format="%d"),
                        "Wins": st.column_config.NumberColumn("Wins", format="%d"),
                        "Losses": st.column_config.NumberColumn("Losses", format="%d"),
                        "Breakeven": st.column_config.NumberColumn("BE", format="%d"),
                        "Win Rate (%)": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                        "Net R": st.column_config.NumberColumn("Realized R", format="%+.2fR"),
                        "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                        "Net PnL ($)": st.column_config.NumberColumn("Net PnL ($)", format="$%+.2f"),
                        "Return (%)": st.column_config.NumberColumn("Return (%)", format="%+.2f%%")
                    }
                )
            else:
                st.info("No data available for selected weekly policy and timeframe.")
        except Exception as ex:
            st.warning(f"Could not load weekly performance: {ex}")


# =============================================================================
# VIEW 4: Analytics (Performance Audit)
# =============================================================================
def show_analytics():
    hero_banner("Analytics Suite", "Signal history, outcomes, and win rate analytics")

    # Render Weekly and Monthly Performance Matrix Card
    render_periodic_performance_matrix()
    st.markdown("<br><hr style='opacity:0.15;'><br>", unsafe_allow_html=True)

    db = get_db()
    
    # 1. Fetch Recent Data (Increased limit to ensure window coverage)
    signals = db.get_recent_signals(limit=2000)

    if not signals:
        st.info("📊 No signal history. Start the Sentinel to collect data.")
        return

    # REMOVED 48h Filter (User requested full history visibility)
    # cutoff_time = datetime.now() - timedelta(hours=48)
    
    recent_signals = []
    for s in signals:
        recent_signals.append(s)
            
    if not recent_signals:
        st.info("📊 No data found.")
        return

    df = pd.DataFrame(recent_signals)
    if 'signal' in df.columns:
        df = df[~df['signal'].isin(['WAIT', 'HEARTBEAT'])]
        
    if df.empty:
        st.info("📊 No actionable signals found in history.")
        return
        
    if 'outcome' not in df.columns:
        df['outcome'] = 'ACTIVE'

    # 3. Calculate KPIs
    # Completed: SUCCESS or FAIL
    completed = df[df['outcome'].isin(['SUCCESS', 'FAIL'])]
    wins = len(completed[completed['outcome'] == 'SUCCESS']) if not completed.empty else 0
    win_rate = (wins / len(completed)) * 100 if not completed.empty else 0
    
    # Active: Must be ACTIVE AND (BUY or SELL). exclude WAIT.
    # Exclude hidden (shadow) trades so the KPI only reflects true live MT5 positions.
    if 'is_hidden' in df.columns:
        active_df = df[(df['outcome'] == 'ACTIVE') & (df['signal'].isin(['BUY', 'SELL'])) & (df['is_hidden'] == 0)]
    else:
        active_df = df[(df['outcome'] == 'ACTIVE') & (df['signal'].isin(['BUY', 'SELL']))]
    active_count = len(active_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Win Rate", f"{win_rate:.1f}%", f"{wins} wins", "accent-green"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Closed Trades", str(len(completed)), "Resolved", "accent-cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Active Trades", str(active_count), "Currently open", "accent-gold"), unsafe_allow_html=True)
    with c4:
        best = ""
        if not completed.empty:
            ps = completed.groupby('symbol')['outcome'].apply(lambda s: (s == 'SUCCESS').sum() / len(s) * 100)
            if not ps.empty:
                best = f"{ps.idxmax()} ({ps.max():.0f}%)"
        st.markdown(kpi_card("Best Pair", best or "N/A", "Highest win rate", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # 3. Check for external filters (e.g. from KPI card or session state)
    default_outcomes = ["ACTIVE", "SUCCESS", "FAIL"]
    filter_mode = st.session_state.pop("analytics_filter", None) or st.query_params.get("filter")
    
    # "closed" filter = Only TP/SL outcomes (ignore expired timeouts)
    if filter_mode == "closed":
        default_outcomes = ["SUCCESS", "FAIL"]
        st.info("🎯 Showing Completed Trades (TP/SL Hit Only)")
    elif filter_mode == "active":
        default_outcomes = ["ACTIVE"]
        st.info("⚡ Showing Active Running Trades")
    elif filter_mode == "expired":
        default_outcomes = ["SUCCESS", "FAIL", "EXPIRED", "N/A"]
        st.info("🔍 Showing All History (Including Timeouts)")
    elif filter_mode == "all":
        default_outcomes = ["ACTIVE", "SUCCESS", "FAIL"]
        st.info("📊 Showing All Live & Resolved Trade Outcomes")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sym_filter = st.multiselect("Filter Pair", sorted(df['symbol'].unique()))
    with fc2:
        sig_filter = st.multiselect("Filter Signal", ["BUY", "SELL", "WAIT"], default=["BUY", "SELL"])
    with fc3:
        out_filter = st.multiselect("Filter Outcome", ["ACTIVE", "SUCCESS", "FAIL", "EXPIRED", "N/A"],
                                     default=default_outcomes)

    filtered = df.copy()
    
    # Clearly label hidden/benched signals so users don't think they are live MT5 trades
    if 'is_hidden' in filtered.columns:
        filtered.loc[filtered['is_hidden'] == 1, 'outcome'] = 'SHADOW'

    if sym_filter: filtered = filtered[filtered['symbol'].isin(sym_filter)]
    if sig_filter: filtered = filtered[filtered['signal'].isin(sig_filter)]
    if out_filter: 
        # Allow filtering to still catch shadow trades if ACTIVE was selected
        filtered = filtered[filtered['outcome'].isin(out_filter) | (filtered['outcome'] == 'SHADOW')]

    if 'confidence' in filtered.columns:
        filtered['confidence'] = filtered['confidence'].apply(
            lambda x: float(x or 0) * 100 if float(x or 0) <= 1.0 else float(x or 0)
        )

    display_cols = [
        'timestamp', 'symbol', 'signal', 'confidence', 'price_at_signal', 
        'tp_price', 'sl_price', 'exit_price', 'exit_reason', 'mt5_ticket', 'exit_time', 'duration_seconds', 
        'outcome', 'regime', 'rsi', 'adx', 'atr', 'vix_proxy', 'yield_slope',
        'macd', 'stoch_k', 'stoch_d', 'cci', 'bb_position'
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True,
                 column_config={
                     "timestamp": "Time", "symbol": "Pair", "signal": "Direction",
                     "price_at_signal": st.column_config.NumberColumn("Entry", format="%.5f"),
                     "tp_price": st.column_config.NumberColumn("TP", format="%.5f"),
                     "sl_price": st.column_config.NumberColumn("SL", format="%.5f"),
                     "exit_price": st.column_config.NumberColumn("Exit Price", format="%.5f"),
                     "exit_reason": "Exit Reason / Profit",
                     "mt5_ticket": "MT5 Ticket",
                     "exit_time": "Exit Time",
                     "duration_seconds": st.column_config.NumberColumn("Duration (s)", format="%d"),
                     "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
                     "outcome": "Outcome",
                     "regime": "Regime",
                     "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                     "adx": st.column_config.NumberColumn("ADX", format="%.1f"),
                     "atr": st.column_config.NumberColumn("ATR", format="%.5f"),
                     "vix_proxy": st.column_config.NumberColumn("VIX Proxy", format="%.4f"),
                     "yield_slope": st.column_config.NumberColumn("Yield Slope", format="%.4f"),
                     "macd": st.column_config.NumberColumn("MACD", format="%.6f"),
                     "stoch_k": st.column_config.NumberColumn("Stoch %K", format="%.1f"),
                     "stoch_d": st.column_config.NumberColumn("Stoch %D", format="%.1f"),
                     "cci": st.column_config.NumberColumn("CCI", format="%.1f"),
                     "bb_position": st.column_config.NumberColumn("BB Pos", format="%.2f")
                 })


# =============================================================================
# VIEW 5: Performance Matrix (Real-Time Audit)
# =============================================================================
def show_performance_matrix():
    from core.performance_gate import get_performance_gate
    db = get_db()
    gate = get_performance_gate()

    hero_banner("Performance Matrix", "Real-time AI surveillance, rolling window analytics, and periodic performance scorecard")

    # Auto-refresh every 60 seconds so new trade closures appear automatically
    import time as _time
    if 'perf_matrix_last_refresh' not in st.session_state:
        st.session_state['perf_matrix_last_refresh'] = _time.time()
    elapsed = _time.time() - st.session_state['perf_matrix_last_refresh']
    seconds_remaining = max(0, 60 - int(elapsed))
    if elapsed >= 60:
        st.session_state['perf_matrix_last_refresh'] = _time.time()
        st.rerun()
    st.caption(f"🔄 Auto-refreshes in {seconds_remaining}s — or click a control to refresh now.")

    # Render Weekly and Monthly Return Performance Matrix
    try:
        render_periodic_performance_matrix()
    except Exception as ex:
        st.error(f"Error displaying performance scorecard: {ex}")
    st.markdown("<br><hr style='opacity:0.15;'><br>", unsafe_allow_html=True)


    @st.fragment(run_every=timedelta(minutes=5))
    def _matrix_grid():
        # 1. Active Surveillance (Live vs Shadow)
        section_header("🛰️", "Active Signal Surveillance")
        raw_active = db.get_active_signals(include_hidden=True)
        # FILTER: Show only real trade signals (BUY/SELL). Skip neutral 'WAIT' noise.
        active = [s for s in raw_active if s.get('signal') in ('BUY', 'SELL')]
        
        RANGING_APPROVED_SET = {'EURAUD', 'AUDNZD', 'GBPUSD', 'XAUUSD', 'USOIL.cash', 'USDJPY', 'EURNZD', 'USDSGD'}
        if active:
            df_active = pd.DataFrame(active)
            # Add Display columns
            df_active['Type'] = df_active['is_hidden'].apply(lambda x: "🛸 SHADOW" if x else "🚀 REAL")
            df_active['Conviction'] = df_active['confidence'].apply(lambda x: f"{x:.1%}")
            # Regime column
            def _regime_label(row):
                r = str(row.get('regime') or '').upper()
                if 'RANGING' in r:
                    return '↔️ RANGING'
                elif 'TRENDING' in r:
                    return '📈 TRENDING'
                elif 'CRISIS' in r:
                    return '⚡ CRISIS'
                return r or '—'
            df_active['Regime'] = df_active.apply(_regime_label, axis=1)
            
            def _model_badge(row):
                mv = str(row.get('model_version') or '')
                if mv == 'confluence_ml_p60':
                    return '🧠 Confluence ML P60'
                elif mv == 'confluence_std_p25':
                    return '⚡ Confluence Std P25'
                elif mv == 'confluence_ml_m15':
                    return '🧠 Confluence ML M15'
                elif mv == 'confluence_m15':
                    return '⚡ Confluence Std M15'
                elif mv == 'manual_m15' or row.get('is_manual') == 1:
                    return '🎯 Manual Sniper'
                elif mv in ('v1', 'foundation_tft', 'foundation'):
                    return '🌐 Foundation V1'
                return mv or '—'
            df_active['Model Engine'] = df_active.apply(_model_badge, axis=1)

            show_cols = ['symbol', 'signal', 'Model Engine', 'Regime', 'Type', 'Conviction', 'confidence_tier', 'timestamp']
            st.dataframe(
                df_active[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Pair", "signal": "Signal", "Model Engine": "Strategy Engine", "Regime": "Market Regime",
                    "confidence_tier": "Tier %", "timestamp": "Detected"
                }
            )
        else:
            st.info("No active signals currently under surveillance.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Performance Thresholds (14-Day Window)
        section_header("📊", "14-Day Performance Matrix")

        col_m1, col_m2 = st.columns([2, 2])
        with col_m1:
            m_filter = st.selectbox(
                "Filter by Strategy Model",
                [
                    "🌟 All Traded Models",
                    "🌟 Dynamic YTD Model (Daily Winning Assets)",
                    "🧠 Confluence ML P60 (Partial 60% + BE+2p)",
                    "⚡ Confluence Standard P25 (Partial 25% + BE+2p)",
                    "🧠 Confluence ML M15 (Original Fixed 1.5R)",
                    "⚡ Confluence M15 Standard (Original Fixed 1.5R)",
                    "🎯 Manual M15 Wick Sniper",
                    "🤖 Foundation v1 Macro AI"
                ],
                key="perf_matrix_14d_model_filter"
            )
        with col_m2:
            scope_filter = st.selectbox(
                "Execution Scope",
                [
                    "🏦 Live Executed Trades (Broker Fills)",
                    "👁️ All Models (Live + Shadow Paper)"
                ],
                key="perf_matrix_14d_scope_filter"
            )
        live_only_choice = "Live" in scope_filter

        model_param = None
        if "Dynamic YTD" in m_filter:
            model_param = "dynamic_ytd_model"
        elif "P60" in m_filter:
            model_param = "confluence_ml_p60"
        elif "P25" in m_filter:
            model_param = "confluence_std_p25"
        elif "Confluence ML M15" in m_filter:
            model_param = "confluence_ml_m15"
        elif "Confluence M15 Standard" in m_filter or "Confluence" in m_filter:
            model_param = "confluence_m15"
        elif "Manual" in m_filter:
            model_param = "manual_m15"
        elif "Foundation" in m_filter:
            model_param = "v1"

        stats_14 = db.get_performance_matrix_stats(14, model_version=model_param, live_only=live_only_choice)
        if stats_14:
            df_14 = pd.DataFrame(stats_14)
            df_14['Win Rate'] = df_14.apply(lambda row: (row['wins'] / row['total_trades']) if row['total_trades'] > 0 else 0, axis=1)

            # Pull regime breakdown per symbol from recent resolved signals
            import sqlite3
            _db_path = str(db.db_path) if hasattr(db, 'db_path') else None
            regime_map = {}
            if _db_path:
                try:
                    with sqlite3.connect(_db_path) as _conn:
                        ticket_filter = "AND mt5_ticket IS NOT NULL" if live_only_choice else ""
                        _cur = _conn.execute(f"""
                            SELECT symbol,
                                   SUM(CASE WHEN regime LIKE '%RANGING%' THEN 1 ELSE 0 END) AS ranging_cnt,
                                   SUM(CASE WHEN regime LIKE '%TRENDING%' THEN 1 ELSE 0 END) AS trending_cnt,
                                   SUM(CASE WHEN model_version = 'confluence_ml_p60' THEN 1 ELSE 0 END) AS p60_cnt,
                                   SUM(CASE WHEN model_version = 'confluence_std_p25' THEN 1 ELSE 0 END) AS p25_cnt,
                                   SUM(CASE WHEN model_version = 'confluence_ml_m15' THEN 1 ELSE 0 END) AS ml_cnt,
                                   SUM(CASE WHEN model_version = 'confluence_m15' THEN 1 ELSE 0 END) AS conf_cnt,
                                   SUM(CASE WHEN regime LIKE '%MANUAL%' OR is_manual = 1 THEN 1 ELSE 0 END) AS manual_cnt
                            FROM signals
                            WHERE outcome IN ('SUCCESS','FAIL')
                              AND timestamp >= datetime('now','-14 days')
                              {ticket_filter}
                            GROUP BY symbol
                        """)
                        for _row in _cur.fetchall():
                            _sym, _r, _t, _p60, _p25, _ml, _c, _m = _row
                            parts = []
                            if _p60 > 0: parts.append(f"🧠 {_p60} P60")
                            if _p25 > 0: parts.append(f"⚡ {_p25} P25")
                            if _ml > 0: parts.append(f"🧠 {_ml} ML")
                            if _c > 0: parts.append(f"⚡ {_c} Std")
                            if _m > 0: parts.append(f"🎯 {_m} M")
                            if _t > 0: parts.append(f"📈 {_t} T")
                            if _r > 0: parts.append(f"↔️ {_r} R")
                            regime_map[_sym] = " / ".join(parts) if parts else '—'
                except Exception:
                    pass

            df_14['Regime Mix'] = df_14['symbol'].map(lambda s: regime_map.get(s, '—'))
            
            # Render custom layout to allow clicking pair to navigate
            cols = st.columns([1.5, 2, 1, 1.5, 1, 1, 2])
            cols[0].markdown("**Pair**")
            cols[1].markdown("**Regime (14-Day)**")
            cols[2].markdown("**Volume**")
            cols[3].markdown("**Win Rate**")
            cols[4].markdown("**Wins**")
            cols[5].markdown("**Losses**")
            cols[6].markdown("**Last Activity**")
            st.markdown("<hr style='margin:0.25rem 0 0.75rem 0; opacity:0.1;'>", unsafe_allow_html=True)
            
            for _, row_data in df_14.iterrows():
                sym = row_data['symbol']
                cols = st.columns([1.5, 2, 1, 1.5, 1, 1, 2])
                
                # Dynamic navigation button
                if cols[0].button(f"📈 {sym}", key=f"act14_{sym}", use_container_width=True):
                    st.session_state['pair_selector'] = sym
                    st.session_state['nav_to_terminal'] = True
                    st.rerun()
                    
                cols[1].text(row_data['Regime Mix'])
                cols[2].text(str(row_data['total_trades']))
                
                # Show win rate nicely
                wr = row_data['Win Rate']
                cols[3].text(f"{wr * 100:.0f}%")
                
                cols[4].text(str(int(row_data['wins'])))
                cols[5].text(str(int(row_data['losses'])))
                cols[6].text(str(row_data['last_trade'])[:19] if row_data['last_trade'] else '—')
        else:
            st.info("Insufficient trading data in the 14-day window.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Institutional Certification (Whitelist)
        section_header("🛡️", "Institutional Certification Status")
        # Try-catch sync to prevent crashing if DB is locked
        # Recompute from DB to ensure we reflect the latest resolved trades
        gate.recompute_from_db(lookback_days=14)
        logger.info("Dashboard recomputed performance matrix from DB.")
            
        matrix = gate.performance_matrix
        cert_records = []
        # Approved ranging list to distinguish RANGING vs TRENDING certifications
        RANGING_APPROVED_SET = {'EURAUD', 'AUDNZD', 'GBPUSD', 'XAUUSD', 'USOIL.cash', 'USDJPY', 'EURNZD', 'USDSGD'}
        if matrix:
            for sym, contents in matrix.items():
                for k, v in contents.items():
                    if not isinstance(v, dict):
                        continue
                        
                    # Check if this is a direct tier (Legacy) or a Direction dict (New)
                    if 'status' in v:
                        # Legacy Format: sym -> tier -> data (skip, has no direction)
                        pass
                    else:
                        # New Format: sym -> direction -> tier -> data
                        # Only show BUY or SELL — skip ALL
                        if k not in ('BUY', 'SELL'):
                            continue
                        for t_str, data in v.items():
                            if isinstance(data, dict) and data.get('status') == 'APPROVED':
                                cert_records.append({
                                    "Symbol": sym,
                                    "Direction": k, "Tier": f"{t_str}%",
                                    "Acc": data.get('accuracy', 0.0), "Trades": data.get('trades', 0),
                                    "Source": data.get('source', 'System')
                                })
        
        if cert_records:
            df_cert = pd.DataFrame(cert_records)
            
            # Render interactive columns for whitelisted symbols
            cols = st.columns([1.5, 1.5, 1.5, 1.5, 1, 2])
            cols[0].markdown("**Pair**")
            cols[1].markdown("**Direction**")
            cols[2].markdown("**Strategy Tier**")
            cols[3].markdown("**Realized Acc**")
            cols[4].markdown("**Trades**")
            cols[5].markdown("**Source**")
            st.markdown("<hr style='margin:0.25rem 0 0.75rem 0; opacity:0.1;'>", unsafe_allow_html=True)
            
            for idx, row_data in df_cert.iterrows():
                sym = row_data['Symbol']
                cols = st.columns([1.5, 1.5, 1.5, 1.5, 1, 2])
                
                # Clickable symbol button
                if cols[0].button(f"🛡️ {sym}", key=f"cert_{sym}_{idx}", use_container_width=True):
                    st.session_state['pair_selector'] = sym
                    st.session_state['nav_to_terminal'] = True
                    st.rerun()
                    
                cols[1].text(row_data.get('Direction', 'ALL'))
                cols[2].text(row_data.get('Tier', '—'))
                cols[3].text(f"{row_data.get('Acc', 0.0) * 100:.1f}%")
                cols[4].text(str(row_data.get('Trades', 0)))
                cols[5].text(row_data.get('Source', '—'))
        else:
            st.warning("No pairs currently meet the 70% institutional certification threshold.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Model Registry (All-Time Models)
        section_header("📋", "Historical Model Registry & Dynamic Whitelists")
        t_dyn, t_reg_model, t_reg_sym = st.tabs(["🌟 Dynamic YTD Model Whitelists", "🤖 Strategy Model Engines", "🌐 By Pair Symbol"])

        with t_dyn:
            from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
            dw_mgr = get_dynamic_whitelist_manager()
            summary = dw_mgr.get_all_models_summary()
            as_of_date = summary.get("as_of_date", "Today")
            models_data = summary.get("models", {})

            col_dyn_desc, col_dyn_btn = st.columns([3.5, 1.5])
            with col_dyn_desc:
                st.markdown(
                    f"##### 🌟 Autonomous Dynamic Winning Asset Selector\n"
                    f"Every day at **00:00 UTC rollover**, each activated model recalculates its YTD profitable pairs "
                    f"($\\text{{Net }} R \\ge 0.0$, at least breakeven). "
                    f"**Only approved winning assets are traded live in MT5.** "
                    f"Chronic underperforming pairs are safely diverted to background shadow mode."
                )
                st.caption(f"📅 Active Whitelist Baseline: As of **{as_of_date}** · Auto-Refresh: **Daily at 00:01 UTC**")
            with col_dyn_btn:
                if st.button("🔄 Recompute Whitelists Now", key="btn_recompute_dyn_whitelists", use_container_width=True, help="Force recalculate YTD Net R across all closed trades right now"):
                    with st.spinner("Recomputing YTD winning assets for all models..."):
                        dw_mgr.compute_ytd_whitelists()
                        _get_cached_model_comparison.clear()
                        _get_cached_performance_matrix.clear()
                        st.toast("✅ Dynamic Model Whitelists recalculated successfully!", icon="🌟")
                        st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Render model cards
            for m_key, m_info in models_data.items():
                w_pairs = m_info.get("winning_pairs", [])
                b_pairs = m_info.get("benched_pairs", [])
                total_trades = m_info.get("total_trades_ytd", 0)
                m_name = m_info.get("name", m_key)
                
                # Check live authorization status from gatekeeper and YTD activation
                from core.model_gatekeeper import is_model_live_authorized
                is_live = is_model_live_authorized(m_key)
                is_active_ytd = dw_mgr.is_model_active_under_ytd(m_key)

                badge_str = "🟢 LIVE ACTIVE" if is_live else "👻 SHADOW MODE"
                badge_bg = "rgba(0,230,118,0.12);border:1px solid #00e67644;color:#00e676" if is_live else "rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);color:#ffd600"
                ytd_badge_html = "<span style='background:rgba(0,230,118,0.15);border:1px solid #00e67655;color:#00e676;font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;'>🟢 ACTIVE UNDER YTD</span>" if is_active_ytd else "<span style='background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:var(--text-secondary);font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:600;'>⚪ DEACTIVATED UNDER YTD</span>"

                with st.container(border=True):
                    mc1, mc2 = st.columns([3.6, 1.4])
                    with mc1:
                        st.markdown(f"**{m_name}** &nbsp; <span style='background:{badge_bg};font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;'>{badge_str}</span> &nbsp; {ytd_badge_html}", unsafe_allow_html=True)
                        st.caption(f"YTD Closed Trades: **{total_trades}** | Approved Winning Assets: **{len(w_pairs)} pairs** | Benched: **{len(b_pairs)} pairs**")
                    with mc2:
                        t_sm = st.toggle("Active under YTD", value=is_active_ytd, key=f"dyn_reg_toggle_{m_key}")
                        if t_sm != is_active_ytd:
                            dw_mgr.set_sub_model_status(m_key, t_sm)
                            _get_cached_model_comparison.clear()
                            _get_cached_performance_matrix.clear()
                            act_txt = "Activated" if t_sm else "Deactivated"
                            st.toast(f"{m_name} {act_txt} under Dynamic YTD Model!", icon="🌟" if t_sm else "⚪")
                            st.rerun()

                    # Display badges for winning pairs
                    if w_pairs:
                        p_stats = m_info.get("pair_stats", {})
                        badges_html = " ".join([
                            f'<span style="display:inline-block;margin:3px;padding:3px 10px;background:rgba(0,230,118,0.12);border:1px solid #00e67655;border-radius:12px;font-size:0.8rem;font-weight:700;color:#00e676;">'
                            f'✅ {p} <span style="font-weight:400;color:var(--text-secondary);font-size:0.75rem;">(+{p_stats.get(p, {}).get("net_r", 0.0):.1f}R · {p_stats.get(p, {}).get("win_rate", 0):.0f}% WR)</span>'
                            f'</span>'
                            for p in w_pairs
                        ])
                        st.markdown(f"**Winning Assets (Live Trading Authorized):**<br>{badges_html}", unsafe_allow_html=True)
                    else:
                        st.warning("No winning assets currently meet the hurdle for this model.")

                    # Collapsible complete table
                    with st.expander(f"🔍 View Full Per-Pair Performance Table ({len(w_pairs) + len(b_pairs)} pairs)"):
                        p_stats = m_info.get("pair_stats", {})
                        if p_stats:
                            rows = []
                            for p, p_data in p_stats.items():
                                rows.append({
                                    "Symbol": p,
                                    "Status": "✅ APPROVED" if p_data.get("status") == "APPROVED" else "🚫 BENCHED",
                                    "Net R": p_data.get("net_r", 0.0),
                                    "PnL ($)": p_data.get("pnl_usd", 0.0),
                                    "Win Rate (%)": p_data.get("win_rate", 0.0),
                                    "Trades": p_data.get("trades", 0),
                                    "Record (W-L-BE)": f"{p_data.get('wins', 0)}W - {p_data.get('losses', 0)}L - {p_data.get('be', 0)}BE",
                                })
                            df_p = pd.DataFrame(rows).sort_values("Net R", ascending=False)
                            st.dataframe(
                                df_p,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Net R": st.column_config.NumberColumn("Realized Net R", format="%+.2f R"),
                                    "PnL ($)": st.column_config.NumberColumn("Realized PnL", format="$%+.2f"),
                                    "Win Rate (%)": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                                }
                            )

        with t_reg_model:
            model_stats = db.get_model_engine_registry_stats()
            if model_stats:
                df_mreg = pd.DataFrame(model_stats)
                MODEL_NAME_MAP = {
                    "dynamic_ytd_model": "🌟 Dynamic YTD Model (Daily Winning Assets)",
                    "confluence_ml_p60": "🧠 Confluence ML M15 (60% TP + 2p BE)",
                    "confluence_std_p25": "⚡ Confluence Standard M15 (25% TP + 2p BE)",
                    "confluence_ml_m15": "🧠 Confluence AI Quality Gate (Fixed 1.5R)",
                    "confluence_m15": "⚡ Confluence Standard Rule-Based (Fixed 1.5R)",
                    "manual_m15": "🎯 Manual M15 Wick Sniper",
                    "v1": "🌐 Foundation V1 Macro AI",
                    "foundation_tft": "🌐 Foundation TFT Specialist",
                    "ensemble_specialist": "🤖 Ensemble Specialist",
                }
                df_mreg["Strategy Engine"] = df_mreg["model_engine"].map(lambda k: MODEL_NAME_MAP.get(k, k))
                df_mreg["Win Rate (%)"] = df_mreg.apply(lambda r: (r["wins"] / r["closed_trades"] * 100) if r["closed_trades"] > 0 else 0, axis=1)
                st.dataframe(
                    df_mreg[["Strategy Engine", "total_setups", "active_setups", "closed_trades", "wins", "losses", "Win Rate (%)", "avg_confidence", "last_seen"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Strategy Engine": st.column_config.TextColumn("Strategy Model Engine", width="large"),
                        "total_setups": st.column_config.NumberColumn("Total Setups", format="%d"),
                        "active_setups": st.column_config.NumberColumn("Active Armed", format="%d"),
                        "closed_trades": st.column_config.NumberColumn("Closed Deals", format="%d"),
                        "wins": st.column_config.NumberColumn("Wins", format="%d"),
                        "losses": st.column_config.NumberColumn("Losses", format="%d"),
                        "Win Rate (%)": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                        "avg_confidence": st.column_config.NumberColumn("Avg Conf", format="%.1%"),
                        "last_seen": "Last Active"
                    }
                )
            else:
                st.info("No strategy model registry records available.")

        with t_reg_sym:
            registry = db.get_model_registry_stats()
            if registry:
                df_reg = pd.DataFrame(registry)
                df_reg['All-Time WR'] = df_reg.apply(lambda row: (row['all_time_wins'] / row['all_time_trades'] * 100) if row['all_time_trades'] > 0 else 0, axis=1)
                st.dataframe(
                    df_reg, use_container_width=True, hide_index=True,
                    column_config={
                        "symbol": "Pair Symbol",
                        "All-Time WR": st.column_config.ProgressColumn("All-Time WR", format="%.1f%%", min_value=0, max_value=100),
                        "all_time_confidence": st.column_config.NumberColumn("Avg Conf", format="%.1%"),
                        "last_seen": "Last Active"
                    }
                )

    _matrix_grid()

    if st.button("🔄 Force Refresh Matrix"):
        st.rerun()


# =============================================================================
# VIEW 6: Control Panel (Settings)
# =============================================================================
def show_control_panel():
    import yaml

    hero_banner("Control Panel", "API configuration, notifications, and system preferences")

    t1, t2 = st.tabs(["🔌 Data Provider", "📲 Notifications"])

    config_path = PROJECT_ROOT / "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return

    with t1:
        section_header("📡", "Data Provider")
        active = config.get('data_provider', {}).get('active', 'mt5')

        st.markdown(f"""
        <div class="glass-card" style="display: flex; align-items: center; gap: 12px; padding: 16px 20px;">
            <div class="status-dot"></div>
            <span style="font-weight: 600;">Active Provider:</span>
            <span style="font-family: var(--font-mono); color: var(--accent-cyan); font-weight: 700;">{active.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        new_provider = st.selectbox("Select Active Data Provider", ["mt5", "yfinance"], index=0 if active == 'mt5' else 1)

        if st.button("💾 Save Provider Settings", use_container_width=True):
            config.setdefault('data_provider', {})['active'] = new_provider
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            st.toast(f"Provider switched to {new_provider.upper()}", icon="🚀")
            time.sleep(1)
            st.rerun()

    with t2:
        section_header("🔔", "Telegram Bot")

        notif = config.get('notifications', {}).get('telegram', {})
        enabled = notif.get('enabled', False)
        token = notif.get('bot_token', '')
        chat = notif.get('chat_id', '')

        if enabled and token and chat:
            st.markdown("""
            <div class="glass-card" style="display: flex; align-items: center; gap: 12px; padding: 16px 20px;">
                <div class="status-dot"></div>
                <span style="font-weight: 600; color: var(--success);">Telegram Connected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="display: flex; align-items: center; gap: 12px; padding: 16px 20px;">
                <div style="width:8px;height:8px;border-radius:50%;background:var(--signal-sell);"></div>
                <span style="color: var(--text-secondary);">Telegram Not Configured</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔔 Send Test Alert", use_container_width=True):
            try:
                import requests
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                resp = requests.post(url, data={"chat_id": chat, "text": "⚡ ForexAlert: Test alert!"})
                if resp.status_code == 200:
                    st.balloons()
                    st.success("Test message sent!")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Failed: {e}")


# =============================================================================
# VIEW 7: Fleet Status (Inlined — no external file dependency)
# =============================================================================
def show_fleet_status():
    import re

    hero_banner("Fleet Status", "Real-time training monitor for Global Foundation Intelligence v2")

    def strip_ansi(text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def parse_training_log(log_path, tail_bytes=50000):
        if not os.path.exists(log_path):
            return None
        
        with open(log_path, 'rb') as f:
            try:
                f.seek(-tail_bytes, os.SEEK_END)
                content = f.read().decode('utf-8', errors='ignore')
            except OSError:
                f.seek(0)
                content = f.read().decode('utf-8', errors='ignore')
                
        lines = content.splitlines()
        
        status = {
            "symbols_processed": [],
            "total_symbols": 30, # Updated to 30 for v2 (29 pairs + GOLD)
            "current_symbol": "None",
            "phase": "Preparing Data",
            "keras_progress": None,
            "last_update": None,
            "metrics": {"accuracy": 0.0, "loss": 0.0},
            "history": {"step": [], "accuracy": [], "loss": []},
            "eta": "Calculating...",
            "start_time": None
        }
        
        log_time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
        epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
        keras_steps_pattern = re.compile(r'(\d+)/(\d+)\s+.*accuracy:\s+([\d\.]+)\s+-\s+loss:\s+([\d\.]+)')
        sequence_pattern = re.compile(r'^\d{4}.*INFO\]\s+([A-Z]+):\s+\d+,\d+\s+sequences')
        
        total_epochs = 60
        completed_epochs = 0
        total_steps = 1158

        for line in lines:
            line = strip_ansi(line)
            
            time_match = log_time_pattern.match(line)
            if time_match:
                status["last_update"] = time_match.group(1)

            if "FOUNDATION BRAIN v2 - TRAINING START" in line:
                status["phase"] = "Initializing V2 Brain"
                
            if "Fetching" in line and "from MT5" in line:
                status["phase"] = "Fetching MT5 Historical Data"
                
            if "sequences" in line and "INFO]" in line and "Total" not in line:
                seq_match = sequence_pattern.search(line)
                if seq_match:
                    symbol = seq_match.group(1)
                    if symbol not in status["symbols_processed"] and len(symbol) <= 7:
                        status["symbols_processed"].append(symbol)
                    status["current_symbol"] = symbol
                    
            if "Total sequences across all pairs" in line:
                status["phase"] = "Building Data Corpus"

            if "Starting TFT model fit" in line:
                status["phase"] = "Model Training"
                status["current_symbol"] = "Global Brain v2"
                
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                completed_epochs = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                status["phase"] = f"Epoch {completed_epochs}/{total_epochs}"
                status["current_symbol"] = "Global Brain v2"

            keras_match = keras_steps_pattern.search(line)
            if keras_match:
                current_step = int(keras_match.group(1))
                total_steps = int(keras_match.group(2))
                train_acc = float(keras_match.group(3))
                train_loss = float(keras_match.group(4))
                
                status["keras_progress"] = ((completed_epochs - 1) * total_steps + current_step, total_steps * total_epochs)
                status["metrics"]["accuracy"] = train_acc
                status["metrics"]["loss"] = train_loss
                
                global_step = (completed_epochs * total_steps) + current_step
                status["history"]["step"].append(global_step)
                status["history"]["accuracy"].append(train_acc)
                status["history"]["loss"].append(train_loss)

        return status

    log_dir = PROJECT_ROOT / "logs"
    log_file = log_dir / "foundation_v2_training.log"

    if not log_file.exists():
        st.info("⏳ No active training log found. Run training to populate this view.")
        st.markdown(f"Expected log at: `{log_file}`")
        return

    status = parse_training_log(str(log_file))
    if not status:
        st.error("Could not parse log file.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Phase", status["phase"], accent="accent-cyan"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Fleet Progress", f"{len(status['symbols_processed'])}/{status['total_symbols']}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Accuracy", f"{status['metrics']['accuracy']:.1%}", accent="accent-green"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Last Sync", status["last_update"] or "--", accent="accent-gold"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if status["keras_progress"]:
        curr, total = status["keras_progress"]
        pct = max(0.0, min(1.0, curr / max(total, 1)))
        
        # Determine Epoch Number
        epoch_str = "Initializing..."
        if "Epoch" in status["phase"]:
            epoch_str = status["phase"]
            
        st.markdown(f"**Training Progress:** {epoch_str} — {pct:.1%} Complete")
        st.progress(pct)

    if status["history"]["accuracy"]:
        section_header("📊", "Accuracy Trend")
        chart_df = pd.DataFrame(status["history"]).set_index("step")
        st.line_chart(chart_df[["accuracy", "loss"]], use_container_width=True)

    section_header("🛰️", "Sequence Building Grid")
    all_syms = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
        "GBPJPY","EURJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","GBPCHF",
        "EURGBP","AUDNZD","NZDCHF","NZDCAD","CADCHF","AUDCHF","EURCAD",
        "GBPNZD","EURNZD","GBPCAD","USDSGD","EURAUD","EURCHF","GBPAUD",
        "AUDCAD","GOLD"
    ]
    
    rows = [all_syms[i:i+6] for i in range(0, len(all_syms), 6)]
    for row in rows:
        cols = st.columns(6)
        for i, sym in enumerate(row):
            done = sym in status["symbols_processed"]
            active = sym == status["current_symbol"]
            
            if done:
                bg = "rgba(0,255,136,0.1)"
                border = "1px solid var(--success)"
                icon = "✅ "
            elif active:
                bg = "rgba(0,229,255,0.05)"
                border = "2px solid var(--accent-cyan)"
                icon = "⚙️ "
            else:
                bg = "rgba(255,255,255,0.03)"
                border = "1px solid var(--border-glass)"
                icon = "⏳ "
                
            cols[i].markdown(f'<div style="padding:10px;border-radius:8px;background:{bg};border:{border};text-align:center;font-size:0.75rem;font-weight:600;">{icon}{sym}</div>', unsafe_allow_html=True)

    with st.expander("📜 Show Raw Training Logs"):
        with open(str(log_file), 'r', encoding='utf-8', errors='ignore') as f:
            tail = f.readlines()[-50:]
        st.code("".join([strip_ansi(l) for l in tail]), language="text")

    if st.button("🔄 Refresh", type="primary"):
        st.rerun()


# =============================================================================
# =============================================================================
# AUTH — Persistent Session Restore
# =============================================================================
# Streamlit session_state is wiped whenever the WebSocket reconnects
# (browser refresh, page reload, server restart, fragment timers, etc.).
# We persist auth using:
#   1. HTTP / WebSocket cookies (apex_session) via st.context.cookies
#   2. URL query param (?t=)
#   3. Browser localStorage client auto-restore
# This ensures that refreshing the browser anywhere in the app (including subpages
# like /copy-trading, /market, /terminal) seamlessly keeps the user logged in.

import importlib
import core.sessions as _sessions_mod
try:
    importlib.reload(_sessions_mod)
except Exception:
    pass

validate_session = _sessions_mod.validate_session
create_session   = _sessions_mod.create_session
delete_session   = _sessions_mod.delete_session
purge_expired    = _sessions_mod.purge_expired
touch_session    = getattr(_sessions_mod, "touch_session", lambda tok: None)


# ── 1. Initialize defaults ────────────────────────────────────────────────────
for _key in ('authenticated', 'user_email', 'user_name', 'user_role', 'user_id', '_session_token'):
    if _key not in st.session_state:
        st.session_state[_key] = False if _key == 'authenticated' else ''

# ── 2. Restore session from Cookie, URL token, or localStorage ───────────────
if not st.session_state.get('authenticated'):
    _token_candidate = ""

    # Priority 1: Check browser cookie via st.context.cookies (sent with HTTP / WS request)
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            _c = st.context.cookies.get("apex_session", "")
            if _c and len(_c) == 64:
                _token_candidate = _c
    except Exception:
        pass

    # Priority 2: Check URL query parameter ?t=
    if not _token_candidate:
        _q = st.query_params.get("t", "")
        if _q and len(_q) == 64:
            _token_candidate = _q

    if _token_candidate:
        _user = validate_session(_token_candidate)
        if _user:
            st.session_state['authenticated']   = True
            st.session_state['user_email']      = _user['email']
            st.session_state['user_name']       = _user['name']
            st.session_state['user_role']       = _user['role']
            st.session_state['user_id']         = _user['id']
            st.session_state['_session_token']  = _token_candidate
            try:
                touch_session(_token_candidate)
            except Exception:
                pass
        else:
            # Token failed validation (expired or revoked): clear it
            st.session_state["_stale_token_cleanup"] = True

# ── 3. Top-level pending-auth handler (fires right after login form submit) ───
# landing.py forms set _pending_auth then call st.rerun().
# We catch it HERE at the absolute top level (no column/tab context).
if "_pending_auth" in st.session_state:
    _p = st.session_state.pop("_pending_auth")
    _tok = _p.get("token", "")
    st.session_state["authenticated"]  = True
    st.session_state["user_email"]     = _p["email"]
    st.session_state["user_name"]      = _p["name"]
    st.session_state["user_role"]      = _p["role"]
    st.session_state["user_id"]        = _p["id"]
    st.session_state["_session_token"] = _tok
    # Embed token in URL so reconnects auto-restore the session
    if _tok:
        st.query_params["t"] = _tok
    st.rerun()

# Ensure system owner (Malick Trabi) accounts always hold the admin role
if st.session_state.get("authenticated"):
    _em = str(st.session_state.get("user_email", "")).lower()
    if _em in ("malicktra99@gmail.com", "malicktra90@gmail.com") or "malick" in _em:
        st.session_state["user_role"] = "admin"

# ── 4. Periodically purge expired tokens (lightweight, ~1ms) ─────────────────
try:
    purge_expired()
except Exception:
    pass

# ── 5. Import Landing Page ────────────────────────────────────────────────────
from landing import show_landing

if not st.session_state['authenticated']:
    # Handle explicit logout or stale token cleanup
    if st.session_state.pop("_just_logged_out", False) or st.session_state.pop("_stale_token_cleanup", False):
        st.html("""
        <script>
        (function() {
            document.cookie = "apex_session=; path=/; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT";
            try { localStorage.removeItem("apex_session"); } catch(e) {}
        })();
        </script>
        """, unsafe_allow_javascript=True)
    else:
        # Check localStorage to silently auto-restore if cookie was omitted
        st.html("""
        <script>
        (function() {
            try {
                var tok = localStorage.getItem("apex_session");
                if (tok && tok.length === 64 && !window.location.search.includes("t=")) {
                    var url = new URL(window.location.href);
                    url.searchParams.set("t", tok);
                    window.location.replace(url.toString());
                }
            } catch(e) {}
        })();
        </script>
        """, unsafe_allow_javascript=True)

    # Clear any stale token from URL if it failed validation
    if st.query_params.get('t'):
        st.query_params.clear()
    show_landing()
    st.stop()

else:
    # DASHBOARD MODE
    # Ensure persistent session token is synced to browser cookie and localStorage
    _cur_tok = st.session_state.get("_session_token", "")
    if _cur_tok and len(_cur_tok) == 64:
        st.html(f"""
        <script>
        (function() {{
            var tok = "{_cur_tok}";
            try {{
                document.cookie = "apex_session=" + tok + "; path=/; max-age=2592000; SameSite=Lax";
                localStorage.setItem("apex_session", tok);
            }} catch(e) {{}}
        }})();
        </script>
        """, unsafe_allow_javascript=True)
        if st.query_params.get("t") != _cur_tok:
            st.query_params["t"] = _cur_tok

    # Define Pages
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


    pg_home = st.Page(show_command_center, title="Command Center", icon="⚡", default=True, url_path="home")
    # Note: url_path="market" works if authenticated.
    pg_market = st.Page(show_market_overview, title="Market Overview", icon="🌍", url_path="market")
    pg_terminal = st.Page(show_trading_terminal, title="Trading Terminal", icon="📈", url_path="terminal")
    pg_analytics = st.Page(show_analytics, title="Analytics Suite", icon="📊", url_path="analytics")
    pg_models = st.Page(show_performance_matrix, title="Performance Matrix", icon="🛡️", url_path="audit")
    pg_fleet = st.Page(show_fleet_status, title="Fleet Status", icon="📊", url_path="fleet")

    path_profile      = os.path.join(BASE_DIR, "pages", "1_User_Profile.py")
    path_vault        = os.path.join(BASE_DIR, "pages", "2_Financials_Vault.py")
    path_settings     = os.path.join(BASE_DIR, "pages", "3_System_Settings.py")
    path_copy_trading = os.path.join(BASE_DIR, "pages", "4_Copy_Trading.py")
    path_admin        = os.path.join(BASE_DIR, "pages", "5_Admin_Panel.py")

    pg_profile      = st.Page(path_profile,      title="Profile & Subscription", icon="👤", url_path="profile")
    pg_vault        = st.Page(path_vault,         title="Master API Vault",       icon="🔐", url_path="vault")
    pg_settings_ext = st.Page(path_settings,     title="System Settings",        icon="⚙️", url_path="settings")
    pg_copy_trading = st.Page(path_copy_trading,  title="Copy Trading Hub",       icon="🔁", url_path="copy-trading")
    pg_admin        = st.Page(path_admin,         title="Fleet & Subscribers",    icon="🛠️", url_path="admin")

    # Build Navigation — admin gets strict access to system controls & fleet
    is_admin = st.session_state.get("user_role") == "admin"
    if is_admin:
        nav_dict = {
            "Intelligence":   [pg_home, pg_market, pg_terminal],
            "Analytics":      [pg_analytics, pg_models],
            "Services":       [pg_copy_trading],
            "Account":        [pg_profile],
            "Administration": [pg_admin, pg_fleet, pg_vault, pg_settings_ext],
        }
    else:
        nav_dict = {
            "Intelligence":   [pg_home, pg_market, pg_terminal],
            "Analytics":      [pg_analytics, pg_models],
            "Services":       [pg_copy_trading],
            "Account":        [pg_profile],
        }
    pg = st.navigation(nav_dict)

    # Sidebar Logo/Footer (Stays constant)
    with st.sidebar:
        sidebar_logo()
        # st.navigation handles the menu rendering automatically here

    # ── Global Page Switcher ────────────────────────────────────────────────
    target_page = st.session_state.pop("nav_target", None)
    if not target_page:
        target_page = st.query_params.get("nav")
        if target_page:
            try:
                del st.query_params["nav"]
            except Exception:
                pass
    if target_page == "terminal" or st.session_state.pop("nav_to_terminal", False):
        st.switch_page(pg_terminal)
    elif target_page == "analytics":
        st.switch_page(pg_analytics)
    elif target_page == "market":
        st.switch_page(pg_market)
    elif target_page == "audit":
        st.switch_page(pg_models)
    elif target_page == "fleet":
        st.switch_page(pg_fleet)
    elif target_page == "home":
        st.switch_page(pg_home)
    elif target_page == "copy_trading":
        st.switch_page(pg_copy_trading)
    elif target_page == "profile":
        st.switch_page(pg_profile)
    elif target_page == "admin" and is_admin:
        st.switch_page(pg_admin)
    elif target_page in ("settings", "control") and is_admin:
        st.switch_page(pg_settings_ext)

    # Run!
    pg.run()

    # Sidebar Footer & Global Rerun (Native approach)
    with st.sidebar:
        st.markdown("---")
        # ── Logged-in user info + Log Out ──────────────────────────────────
        _uname = st.session_state.get("user_name", "")
        _uemail = st.session_state.get("user_email", "")
        _urole  = st.session_state.get("user_role", "subscriber")
        if _uname:
            role_icon = "🛠️" if _urole == "admin" else "👤"
            st.markdown(
                f"<div style='font-size:0.82rem; color:var(--text-muted); margin-bottom:4px;'>"
                f"{role_icon} <strong style='color:var(--text-primary);'>{_uname}</strong><br>"
                f"<span style='font-size:0.75rem;'>{_uemail}</span></div>",
                unsafe_allow_html=True,
            )
            if st.button("🚪 Log Out", use_container_width=True, key="logout_btn"):
                # Delete server-side session token so URL ?t= can't restore it
                _tok = st.session_state.get("_session_token", "")
                if _tok:
                    try:
                        delete_session(_tok)
                    except Exception:
                        pass
                for _k in ('authenticated', 'user_email', 'user_name', 'user_role', 'user_id', '_session_token'):
                    st.session_state[_k] = False if _k == 'authenticated' else ''
                st.query_params.clear()
                st.session_state["_just_logged_out"] = True
                st.rerun()
        st.markdown("")
        sidebar_footer()

    # Auto-refresh is now handled inline via st_autorefresh in show_trading_terminal().
    # Command Center and Market Overview use manual refresh via the sidebar button.
# Force Reload: 2026-04-19 21:30 — Performance Matrix
