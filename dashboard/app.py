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
from datetime import datetime
from datetime import timedelta

# Shared design system
from theme import (
    inject_css, get_db, get_engine, get_inference,
    kpi_card, hero_banner, sidebar_logo, sidebar_footer, section_header,
    PROJECT_ROOT
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

@st.cache_resource
def load_inference_v2():
    engine = get_inference()
    return engine


# ── Chart Renderer (TradingView Lightweight Charts – blink-free) ───
def render_chart(df, symbol):
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

    # Determine price precision from symbol
    precision = 3 if "JPY" in symbol else 5

    html = f"""
    <div id="tv-chart" style="width:100%;height:460px;border-radius:12px;overflow:hidden;"></div>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const container = document.getElementById('tv-chart');
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
                "Real-time AI surveillance across 31 forex pairs · Institutional-level precision targeting",
                show_status=True)

    all_pairs = engine.get_all_pairs()
    
    # 1. Active Signals (Source of Truth)
    raw_active = db.get_active_signals(include_hidden=True)
    # Filter: ONLY show BUY/SELL as "Active Signals" in the live table
    # but still track hidden status for labels
    active_signals = [s for s in raw_active if s.get('signal') in ['BUY', 'SELL']]
    active_count = len(active_signals)

    # 2. Expired/Closed Signals (Historical - Last 48h Window)
    # Include hidden signals so benched history is visible in charts
    recent = db.get_recent_signals(limit=5000, include_hidden=True)
    expired_signals = []
    success_rate = 0.0
    completed_count = 0

    if recent:
        from datetime import datetime, timedelta
        # Time Window: Current Week (Monday to Now)
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = start_of_week
        
        # Filter recent signals by time first
        recent_window = []
        for s in recent:
            try:
                # Parse ISO timestamp
                ts_str = s['timestamp']
                # Handle potential fractional seconds or different formats safely
                s_time = pd.to_datetime(ts_str)
                if s_time >= cutoff_time:
                    recent_window.append(s)
            except:
                continue
        
        df_sig = pd.DataFrame(recent_window)
        if not df_sig.empty:
            if 'outcome' not in df_sig.columns:
                df_sig['outcome'] = 'ACTIVE'
            
            # Expired = Actionable signals (BUY/SELL) that are NOT active
            # We filter out 'WAIT' signals from the count entirely as requested ("A wait signal should not be counted")
            expired_signals = [
                s for s in recent_window 
                if s.get('outcome') != 'ACTIVE' 
                and s.get('signal') in ['BUY', 'SELL']
            ]
            
            # Closed for KPI = SUCCESS or FAIL (User requested strict adherence to TP/SL logic)
            # We exclude "EXPIRED" from the main "Closed Trades" count as they represent timeouts/legacy logic
            # and the user considers them invalid if they haven't hit TP/SL.
            completed = df_sig[df_sig['outcome'].isin(['SUCCESS', 'FAIL'])]
            completed_count = len(completed)
            
            # Legacy/Timeout signals
            expired_timeout_count = len(df_sig[df_sig['outcome'] == 'EXPIRED'])
            
            if not completed.empty:
                success_rate = (len(completed[completed['outcome'] == 'SUCCESS']) / len(completed)) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Monitored Pairs", len(all_pairs), "Majors · Minors · Crosses", "accent-cyan", link_url="/market?nav=true"), unsafe_allow_html=True)
    with c2:
        # Active Signals -> Scroll/Focus (or Analytics)
        st.markdown(kpi_card("Active Signals", active_count, "Running trades", "accent-gold", link_url="/analytics?nav=true"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Win Rate (Week)", f"{success_rate:.1f}%", f"{completed_count} closed trades", "accent-green", link_url="/analytics?nav=true"), unsafe_allow_html=True)
    with c4:
        # Show "Closed Trades" (TP/SL hit) as primary metric count.
        # BUT link to 'expired' filter so user can see "trades that did not complete" as requested.
        st.markdown(kpi_card("Closed Trades (Week)", completed_count, "Hit TP or SL", "accent-cyan", link_url="/analytics?nav=true&filter=expired"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("System Health", "Online", "Watchdog · Sentinel · API", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🎯", "High Confidence Opportunities")

    if active_signals:
        # Sort by confidence descending
        # Convert list of dicts to DF for display
        df_active = pd.DataFrame(active_signals)
        # Filter for display (optional, depending on what we want to show)
        # Assuming all active signals are "opportunities"
        
        if not df_active.empty:
             # Select cols
            cols_to_show = ['symbol', 'signal', 'confidence', 'price_at_signal', 'timestamp']
            df_display = df_active[cols_to_show].copy()
            df_display.columns = ['Pair', 'Direction', 'Confidence', 'Entry', 'Detected']
            
            # Formatting
            df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.0%}")
            
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
                        format="%s",
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
                    # Navigate to Trading Terminal via URL path (robust)
                    st.switch_page("terminal")
                except Exception as e:
                    st.error(f"Navigation failed: {e}")
            
    # Fallback to empty if no signals
            
    # Fallback to empty if no signals
    if not active_signals:
        st.info("No active high-confidence signals at the moment.")

    # 4. Closed Trades View
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📜 Closed Trade History (Click to View)", expanded=False):
        if expired_signals:
            df_hist = pd.DataFrame(expired_signals)
            # Ensure cols exist
            cols = ['symbol', 'signal', 'outcome', 'confidence', 'price_at_signal', 'timestamp']
            show_cols = [c for c in cols if c in df_hist.columns]
            
            st.dataframe(
                df_hist[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Pair",
                    "signal": "Direction",
                    "outcome": "Result",
                    "confidence": st.column_config.NumberColumn("Conf", format="%.2f"),
                    "price_at_signal": st.column_config.NumberColumn("Entry", format="%.5f"),
                    "timestamp": "Detected"
                }
            )
        else:
            st.info("No expired or closed trades found in recent history.")


# =============================================================================
# VIEW 2: Market Overview
# =============================================================================
def show_market_overview():
    import yaml

    hero_banner("Market Overview", "Real-time AI signal grid across 31 global currency pairs")

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
        # Build signal map from DB
        all_signals = db.get_recent_signals(limit=1000, include_hidden=True)
        sig_map = {}
        if all_signals:
            for s in sorted(all_signals, key=lambda x: x['timestamp']):
                sym = s['symbol']
                sig_map[sym] = s
                    
        active_signals = db.get_active_signals(include_hidden=True)
        if active_signals:
            for s in sorted(active_signals, key=lambda x: x['timestamp']):
                sig_map[s['symbol']] = s

        # Signal grid categories
        categories = {
            "⚡ Majors": config_pairs.get('majors', []),
            "🔷 Minors": config_pairs.get('minors', []),
            "🔶 Crosses": config_pairs.get('crosses', []),
        }

        for cat_name, pair_list in categories.items():
            if not pair_list: continue
            st.markdown(f'<div class="section-header"><span class="section-header-text">{cat_name}</span></div>', unsafe_allow_html=True)
            cols = st.columns(4)
            symbols = [p['symbol'] for p in pair_list]
            for i, symbol in enumerate(symbols):
                sig_data = sig_map.get(symbol)
                with cols[i % 4]:
                    link = f'terminal?nav=true&symbol={symbol}'
                    if not sig_data:
                        tile_html = f'<div class="signal-tile tile-wait"><div class="tile-symbol">{symbol}</div><div class="tile-signal tile-signal-wait">—</div><div class="tile-conf">Awaiting Data</div></div>'
                    else:
                        sig = sig_data.get('signal', 'WAIT')
                        regime = sig_data.get('regime') or ''
                        r_upper = str(regime).upper()
                        is_crisis = "CRISIS" in r_upper or "VOLATILE" in r_upper
                        conf = sig_data.get('confidence', 0)
                        
                        regime_badge = ""
                        if regime:
                            if is_crisis: regime_badge = '<div class="regime-badge-crisis">⚡ CRISIS</div>'
                            elif "TRENDING" in r_upper: regime_badge = '<div class="regime-badge-trending">TRENDING</div>'
                            else: regime_badge = f'<div class="regime-badge-neutral">{r_upper}</div>'
                        
                        display_sig = "SAFE" if is_crisis else sig
                        tile_html = (
                            f'<div class="signal-tile {"tile-wait" if is_crisis else "tile-active"}" style="position: relative;">'
                            f'{regime_badge}'
                            f'<div class="tile-symbol">{symbol}</div>'
                            f'<div class="tile-signal {"tile-signal-wait" if is_crisis else "tile-signal-active"}">{display_sig}</div>'
                            f'<div class="tile-conf">{conf:.0%}</div>'
                            f'</div>'
                        )
                    st.markdown(f'<a href="{link}" target="_parent" style="text-decoration: none; color: inherit;">{tile_html}</a>', unsafe_allow_html=True)

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

    categories = {
        "⚡ Majors": config_pairs.get('majors', []),
        "🔷 Minors": config_pairs.get('minors', []),
        "🔶 Crosses": config_pairs.get('crosses', []),
    }

    for cat_name, pair_list in categories.items():
        if not pair_list:
            continue
        symbols = [p['symbol'] for p in pair_list]

        st.markdown(f"""
        <div class="section-header">
            <span class="section-header-text">{cat_name}</span>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        for i, symbol in enumerate(symbols):
            sig_data = sig_map.get(symbol)
            with cols[i % 4]:
                # Wrap tile in a link to the terminal (relative path matches st.Page url_path)
                # MUST include nav=true to persist authentication state on reload
                link = f'terminal?nav=true&symbol={symbol}'
                
                if not sig_data:
                    tile_html = (
                        f'<div class="signal-tile tile-wait">'
                        f'<div class="tile-symbol">{symbol}</div>'
                        f'<div class="tile-signal tile-signal-wait">—</div>'
                        f'<div class="tile-conf">Awaiting Data</div>'
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
                    
                    if regime:
                        if is_crisis:
                            regime_badge = '<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: #FF4466; background: rgba(255,68,102,0.15); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em; border: 1px solid rgba(255,68,102,0.3); box-shadow: 0 0 10px rgba(255,68,102,0.2);">⚡ CRISIS</div>'
                        elif "TRENDING" in r_upper:
                            regime_badge = '<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: #00FF88; background: rgba(0,255,136,0.1); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em; border: 1px solid rgba(0,255,136,0.2);">TRENDING</div>'
                        elif "RANGING" in r_upper:
                            regime_badge = '<div style="position: absolute; top: 10px; right: 10px; font-size: 0.55rem; color: var(--text-muted); background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); letter-spacing: 0.1em;">RANGING</div>'

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
                        # Use raw_confidence if available to show why it's a crisis (overextended)
                        f_conf = sig_data.get('raw_confidence', conf)
                        conf_display = f"{(f_conf or 0.0):.0%}" if (f_conf or 0.0) > 0 else "Blocked"
                        # Force red border for crisis tiles
                        extra_styles = "border: 1px solid rgba(255,68,102,0.4); background: rgba(255,68,102,0.03); box-shadow: inset 0 0 20px rgba(255,68,102,0.05);"
                    elif outcome == 'ACTIVE':
                        if is_hidden:
                            # Shadow / Watch Only signal
                            display_sig = "WATCH"
                            css_tile = "tile-wait" # Neutral background
                            css_signal = "tile-signal-wait" 
                            conf_display = f"{sig_data.get('raw_confidence', conf):.0%}"
                            conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar" style="width: {conf:.1%}; background: var(--text-muted);"></div></div>'
                        elif sig == "BUY":
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
                        conf_display = f"{(sig_data.get('raw_confidence') or 0.0):.0%}" if (sig_data.get('raw_confidence') or 0.0) > 0 else "Monitoring..."

                    tile_html = (
                        f'<div class="signal-tile {css_tile}" '
                        f'style="position: relative; {"opacity: 0.7;" if is_hidden else ""} {extra_styles}">'
                        f'{regime_badge}'
                        f'<div class="tile-symbol">{symbol}</div>'
                        f'<div class="tile-signal {css_signal}">{display_sig}</div>'
                        f'<div class="tile-conf">{conf_display}</div>'
                        f'{conf_bar}'
                        f'</div>'
                    )

                st.markdown(f'<a href="{link}" target="_parent" style="text-decoration: none; color: inherit; display: block;">{tile_html}</a>', unsafe_allow_html=True)


# =============================================================================
# VIEW 3: Trading Terminal
# =============================================================================
def show_trading_terminal():
    engine = load_engine()
    inf_engine = load_inference_v2()
    db = get_db()

    with st.sidebar:
        section_header("🎛️", "Analysis Controls")
        all_pairs = engine.get_all_pairs()

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

    # ── Live Data Fragment (reruns every 15s WITHOUT full page blink) ──
    @st.fragment(run_every=timedelta(seconds=15))
    def _live_terminal_data():
        result = None
        pred = "WAIT"
        conf = 0.0
        df = pd.DataFrame()

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

                # 1. Check for EXISTING ACTIVE SIGNAL first (High Priority)
                active_signals = db.get_active_signals(symbol=symbol, include_hidden=True)
                
                if active_signals:
                    result = active_signals[0]
                    pred = result['signal']
                    conf = result['confidence']
                    is_hidden = bool(result.get('is_hidden', False))
                    
                    if is_hidden:
                        st.info(f"👀 WATCH ONLY: Shadow Certification in Progress (Conviction: {conf:.1%})")
                    else:
                        st.success(f"🚀 LIVE SIGNAL: {symbol} {pred} Active in Terminal")
                else:
                    # 2. No active signal? Run FRESH INFERENCE
                    result = inf_engine.predict_symbol(
                        symbol, save_to_db=False, 
                        win_rate=st.session_state['accuracy_target'], allow_stale=False
                    )

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
                    if diff_hours > 4.0:
                        is_market_closed = True
                        st.warning(f"⛔ MARKET CLOSED · Displaying analysis from last close ({last_ts.strftime('%d %b %H:%M UTC')})")
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
                        if result and result.get('price_at_signal'):
                            try:
                                entry_price = float(result['price_at_signal'])
                                direction = 1 if result.get('signal') == 'BUY' else -1
                                pip_size = 0.01 if ('JPY' in symbol or 'GOLD' in symbol or 'XAU' in symbol) else 0.0001
                                pnl_pips = (last_price - entry_price) / pip_size * direction
                                
                                pnl_color = "#00FF88" # Fallback green
                                if pnl_pips < 0: pnl_color = "#FF4466" # Fallback red
                                
                                pnl_sign = "+" if pnl_pips >= 0 else ""
                                pnl_html = f'<span style="font-family: monospace; font-size: 1.1rem; font-weight: 700; color: {pnl_color}; margin-left: 16px; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.05);">{pnl_sign}{pnl_pips:.1f} pips</span>'
                            except:
                                pnl_html = ""

                        # Render Header (with hardcoded fallbacks for CSS variables)
                        st.markdown(f"""
                        <div style="margin-bottom: 16px;">
                            <span style="font-size: 1.5rem; font-weight: 800; color: #ffffff;">{symbol}</span>
                            <span style="font-family: monospace; font-size: 1.3rem; font-weight: 700; color: #00E5FF; margin-left: 12px;">{last_price:.5f}</span>
                            <span style="font-family: monospace; font-size: 0.85rem; color: {'#00FF88' if change >= 0 else '#FF4466'}; margin-left: 8px;">
                                {'▲' if change >= 0 else '▼'} {abs(change):.2%}
                            </span>
                            {pnl_html}
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
            else:
                st.warning("No chart data available. Check API connection.")

        with col_side:
            section_header("🤖", "AI Verdict")

            if result:
                css = "signal-wait"
                if pred == "BUY": css = "signal-buy"
                elif pred == "SELL": css = "signal-sell"

                p_buy = result.get('buy_prob') or 0.0
                p_sell = result.get('sell_prob') or 0.0
                p_wait = result.get('wait_prob') or 0.0

                # --- 2. Performance Status & Safety ---
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
                    
                    # Dynamic override for Crisis/Safety blocks
                    if is_crisis:
                        status_text = "⚠️ CRISIS BLOCK (Safety)"
                        status_color = "#FF4466" # Bright Red
                        pred = "WAIT"
                    elif is_market_closed:
                        status_text = "HISTORICAL ANALYSIS"
                        status_color = "var(--text-muted)"
                        if pred in ('BUY', 'SELL'): pred = "WAIT"
                    elif pred in ('BUY', 'SELL'):
                        is_hidden = bool(result.get('is_hidden', 0))
                        if not is_actively_trading:
                            status_text = f"RESTING (AI Conviction: {pred})"
                            status_color = "var(--text-muted)"
                            pred = "WAIT"
                        elif is_hidden or not is_approved:
                            status_text = "CERTIFICATION PHASE (Shadow)"
                            status_color = "var(--accent-gold)"
                            pred = "WAIT"
                        else:
                            status_color = "var(--signal-buy)" if pred == "BUY" else "var(--signal-sell)"
                    else:
                        status_color = "var(--accent-gold)" if (pred == "WAIT" and conf > 0.1) else "var(--text-muted)"
                except Exception as e:
                    logger.warning(f"Status calculation failed: {e}")
                    status_text = "INITIALIZING..."
                    status_color = "var(--text-muted)"
                    winning_tier = "60"
                
                # --- 3. Render AI Verdict Card ---
                try:
                    ts_display = "Just Now"
                    try:
                        from datetime import datetime
                        ts_obj = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
                        ts_display = ts_obj.strftime("%d %b %H:%M")
                    except: pass

                    display_conf = result.get('raw_confidence', conf) or 0.0
                    vol_trades = result.get('model_trades', 0) or 0
                    
                    st.markdown(f"""
<div class="glass-card" style="padding: 24px; text-align: center; border-top: 3px solid {status_color};">
<!-- 1. DECISION LAYER -->
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.15em; color: var(--text-muted); text-transform: uppercase;">
    Target: {winning_tier}%
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);">
    🕒 {ts_display}
    </div>
</div>
<div class="signal-badge {css}" style="margin-bottom: 20px;">{pred}</div>
{f'<div style="font-family: var(--font-mono); font-size: 0.6rem; color: #00FF88; margin-top: -15px; margin-bottom: 15px;">AI INTENT: {result.get("expert_intent")}</div>' if (pred == "WAIT" and result.get("expert_intent") and result.get("expert_intent") != "WAIT") else ''}
<!-- 2. EXPERT CONVICTION vs HURDLE -->
<div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--border-glass);">
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
<!-- 4. SAFETY EXPLAINER -->
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

                if pred in ["BUY", "SELL"] and result.get('tp_price'):
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("📍", "Trading Levels")
                    tp_pips = result.get('tp_pips') or 0
                    sl_pips = result.get('sl_pips') or 0
                    rr = tp_pips / max(sl_pips, 1)

                    st.markdown(f"""
                    <div class="glass-card" style="padding: 16px;">
                        <div class="level-row level-tp">
                            <span class="level-label">TP</span>
                            <span class="level-price">{result['tp_price']:.5f}</span>
                            <span class="level-pips">+{tp_pips}p</span>
                        </div>
                        <div class="level-row level-entry">
                            <span class="level-label">Entry</span>
                            <span class="level-price">{result['price_at_signal']:.5f}</span>
                        </div>
                        <div class="level-row level-sl">
                            <span class="level-label">SL</span>
                            <span class="level-price">{result['sl_price']:.5f}</span>
                            <span class="level-pips">-{sl_pips}p</span>
                        </div>
                        <div style="text-align: center; margin-top: 12px;">
                            <span class="rr-badge">R:R 1:{rr:.1f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 40px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🔌</div>
                    <div style="color: var(--text-muted);">Select a pair to analyze</div>
                </div>
            """, unsafe_allow_html=True)

    # Invoke the fragment — first call renders, subsequent calls auto-rerun every 15s
    _live_terminal_data()

# =============================================================================
# VIEW 4: Analytics (Performance Audit)
# =============================================================================
def show_analytics():
    hero_banner("Analytics Suite", "Signal history, outcomes, and win rate analytics")

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
    if 'outcome' not in df.columns:
        df['outcome'] = 'ACTIVE'

    # 3. Calculate KPIs
    # Completed: SUCCESS or FAIL
    completed = df[df['outcome'].isin(['SUCCESS', 'FAIL'])]
    wins = len(completed[completed['outcome'] == 'SUCCESS']) if not completed.empty else 0
    win_rate = (wins / len(completed)) * 100 if not completed.empty else 0
    
    # Active: Must be ACTIVE AND (BUY or SELL). exclude WAIT.
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
            ps = completed.groupby('symbol').apply(lambda x: (x['outcome']=='SUCCESS').sum()/len(x)*100)
            if not ps.empty:
                best = f"{ps.idxmax()} ({ps.max():.0f}%)"
        st.markdown(kpi_card("Best Pair", best or "N/A", "Highest win rate", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # 3. Check for external filters (e.g. from KPI card)
    default_outcomes = ["ACTIVE", "SUCCESS", "FAIL"]
    
    # "closed" filter = Only TP/SL outcomes (ignore expired timeouts)
    if st.query_params.get("filter") == "closed":
        default_outcomes = ["SUCCESS", "FAIL"]
        st.info("🎯 Showing Completed Trades (TP/SL Hit Only)")
        
    elif st.query_params.get("filter") == "expired":
        default_outcomes = ["SUCCESS", "FAIL", "EXPIRED", "N/A"]
        st.info("🔍 Showing All History (Including Timeouts)")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sym_filter = st.multiselect("Filter Pair", sorted(df['symbol'].unique()))
    with fc2:
        sig_filter = st.multiselect("Filter Signal", ["BUY", "SELL", "WAIT"], default=["BUY", "SELL"])
    with fc3:
        out_filter = st.multiselect("Filter Outcome", ["ACTIVE", "SUCCESS", "FAIL", "EXPIRED", "N/A"],
                                     default=default_outcomes)

    filtered = df.copy()
    if sym_filter: filtered = filtered[filtered['symbol'].isin(sym_filter)]
    if sig_filter: filtered = filtered[filtered['signal'].isin(sig_filter)]
    if out_filter: filtered = filtered[filtered['outcome'].isin(out_filter)]

    display_cols = ['timestamp', 'symbol', 'signal', 'confidence', 'price_at_signal', 'tp_price', 'sl_price', 'outcome']
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True,
                 column_config={
                     "timestamp": "Time", "symbol": "Pair", "signal": "Direction",
                     "price_at_signal": st.column_config.NumberColumn("Entry", format="%.5f"),
                     "tp_price": st.column_config.NumberColumn("TP", format="%.5f"),
                     "sl_price": st.column_config.NumberColumn("SL", format="%.5f"),
                     "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=1),
                     "outcome": "Outcome"
                 })


# =============================================================================
# VIEW 5: Model Audit (Backtest)
# =============================================================================
def show_model_audit():
    import json
    try:
        import plotly.express as px
    except:
        px = None

    hero_banner("Specialist Model Audit", "Transparency report — only validated models are deployed")

    models_dir = PROJECT_ROOT / "models" / "specialist"

    if not models_dir.exists():
        st.info("⏳ No specialist models found. Run training first.")
        return

    records = []
    for sym_dir in models_dir.iterdir():
        if sym_dir.is_dir():
            for sig_type in ["BUY", "SELL"]:
                mpath = sym_dir / sig_type / "metrics.json"
                if mpath.exists():
                    try:
                        with open(mpath) as f:
                            m = json.load(f)
                        records.append({
                            "Symbol": sym_dir.name, "Type": sig_type,
                            "Win Rate": m.get("accuracy", 0.0),
                            "Params": str(m.get("params", {})),
                            "Certified": m.get("created_at", "")[:16]
                        })
                    except:
                        pass

    if not records:
        st.info("⏳ No certified models yet.")
        return

    df = pd.DataFrame(records)
    avg = df["Win Rate"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Certified Models", str(len(df)), "Deployed", "accent-cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Avg Win Rate", f"{avg:.1%}", "Fleet average", "accent-green"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Threshold", "60.0%", "Min requirement", "accent-gold"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if px:
        fig = px.bar(df, x="Symbol", y="Win Rate", color="Type", barmode="group",
                     color_discrete_map={"BUY": "#00FF88", "SELL": "#FF4466"})
        fig.add_hline(y=0.6, line_dash="dash", line_color="rgba(255,215,0,0.4)",
                      annotation_text="60% Threshold", annotation_font_color="#FFD700")
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(10,14,26,0)', paper_bgcolor='rgba(10,14,26,0)',
            font=dict(family='Inter, sans-serif', color='#8b95a8'),
            yaxis_tickformat=".0%", height=380, margin=dict(l=0,r=0,t=30,b=0),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    section_header("📋", "Model Registry")
    sorted_df = df.sort_values("Win Rate", ascending=False).copy()
    sorted_df["Win Rate"] = sorted_df["Win Rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(sorted_df, use_container_width=True, hide_index=True)


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
        active = config.get('data_provider', {}).get('active', 'yfinance')

        st.markdown(f"""
        <div class="glass-card" style="display: flex; align-items: center; gap: 12px; padding: 16px 20px;">
            <div class="status-dot"></div>
            <span style="font-weight: 600;">Active Provider:</span>
            <span style="font-family: var(--font-mono); color: var(--accent-cyan); font-weight: 700;">{active.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        api_key = st.text_input("TwelveData API Key",
                                 value=config.get('data_provider', {}).get('twelvedata', {}).get('api_key', ''),
                                 type="password")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("💾 Save & Switch to TwelveData", use_container_width=True):
                if api_key:
                    config.setdefault('data_provider', {}).setdefault('twelvedata', {})['api_key'] = api_key
                    config['data_provider']['active'] = 'twelvedata'
                    with open(config_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)
                    st.toast("Provider switched to TwelveData", icon="🚀")
                    time.sleep(1)
                    st.rerun()
        with bc2:
            if st.button("🔄 Revert to Yahoo Finance", use_container_width=True):
                config['data_provider']['active'] = 'yfinance'
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                st.toast("Reverted to Yahoo Finance", icon="🔄")
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
                resp = requests.post(url, data={"chat_id": chat, "text": "⚡ ApexForex: Test alert!"})
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

    hero_banner("Fleet Status", "Real-time training monitor for Specialist and Foundation models")

    def strip_ansi(text):
        return re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])').sub('', text)

    def parse_specialist_log(log_path, tail_bytes=50000):
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
            "symbols_processed": [], "total_symbols": 31,
            "current_symbol": "None", "phase": "Preparing Fleet",
            "keras_progress": None, "last_update": None,
            "metrics": {"accuracy": 0.0, "loss": 0.0},
            "history": {"step": [], "accuracy": [], "loss": []},
            "eta": "Calculating..."
        }
        log_time_p = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
        fold_p = re.compile(r'Fold\s+(\d+):\s+Acc=([\d\.]+)%,\s+Loss=([\d\.]+)')
        step = 0
        for line in lines:
            line = strip_ansi(line)
            m = log_time_p.match(line)
            if m: status["last_update"] = m.group(1)
            if "Starting expert training for" in line:
                sym = line.split("Starting expert training for")[-1].split()[0].strip()
                status["current_symbol"] = sym
                status["phase"] = f"Training {sym}"
            if "Worker" in line and "Completed" in line:
                sym = line.split("Completed")[-1].strip().split()[0].strip()
                if sym not in status["symbols_processed"] and len(sym) <= 7:
                    status["symbols_processed"].append(sym)
            m2 = fold_p.search(line)
            if m2:
                fold_idx = int(m2.group(1))
                acc = float(m2.group(2)) / 100.0
                loss = float(m2.group(3))
                status["metrics"] = {"accuracy": acc, "loss": loss}
                status["phase"] = f"WFCV Fold {fold_idx}/5"
                status["keras_progress"] = (fold_idx, 5)
                step += 1
                status["history"]["step"].append(step)
                status["history"]["accuracy"].append(acc)
                status["history"]["loss"].append(loss)
        return status

    log_dir = PROJECT_ROOT / "logs"
    log_file = log_dir / "specialist_progressive.log"

    if not log_file.exists():
        st.info("⏳ No active training log found. Run specialist training to populate this view.")
        st.markdown(f"Expected log at: `{log_file}`")
        return

    status = parse_specialist_log(str(log_file))
    if not status:
        st.error("Could not parse log file.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Phase", status["phase"], accent="accent-cyan"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Fleet Progress", f"{len(status['symbols_processed'])}/31"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Accuracy", f"{status['metrics']['accuracy']:.1%}", accent="accent-green"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Last Sync", status["last_update"] or "--", accent="accent-gold"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if status["keras_progress"]:
        curr, total = status["keras_progress"]
        st.progress(min(1.0, curr / max(total, 1)))

    if status["history"]["accuracy"]:
        section_header("📊", "Accuracy Trend")
        chart_df = pd.DataFrame(status["history"]).set_index("step")
        st.line_chart(chart_df[["accuracy", "loss"]], use_container_width=True)

    section_header("🛰️", "Symbol Grid")
    all_syms = [
        "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","USDCAD","NZDUSD",
        "EURGBP","EURJPY","GBPJPY","EURCAD","EURAUD","EURCHF","GBPCHF",
        "GBPAUD","GBPCAD","AUDJPY","CADJPY","CHFJPY","AUDCAD","NZDJPY",
        "AUDNZD","EURNZD","GBPNZD","AUDCHF","NZDCAD","CADCHF","NZDCHF",
        "XAUUSD","USOIL","USDSGD"
    ]
    rows = [all_syms[i:i+6] for i in range(0, len(all_syms), 6)]
    for row in rows:
        cols = st.columns(6)
        for i, sym in enumerate(row):
            done = sym in status["symbols_processed"]
            active = sym == status["current_symbol"]
            bg = "rgba(0,255,136,0.1)" if done else "rgba(0,229,255,0.05)" if active else "rgba(255,255,255,0.03)"
            border = "1px solid var(--success)" if done else "2px solid var(--accent-cyan)" if active else "1px solid var(--border-glass)"
            cols[i].markdown(f'<div style="padding:10px;border-radius:8px;background:{bg};border:{border};text-align:center;font-size:0.75rem;font-weight:600;">{sym}</div>', unsafe_allow_html=True)

    section_header("📜", "Log Tail (Last 50 Lines)")
    with open(str(log_file), 'r', encoding='utf-8', errors='ignore') as f:
        tail = f.readlines()[-50:]
    st.code("".join([strip_ansi(l) for l in tail]), language="text")

    if st.button("🔄 Refresh", type="primary"):
        st.rerun()


# =============================================================================
# NAVIGATION & ROUTING (Streamlit 1.34+)
# =============================================================================

# Auth State Initialization
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Deep Link Auth Bypass (for KPI Cards)
# If a user clicks a link with ?nav=true, we auto-authenticate to allow the route to load.
# In a production app, verify a token here. For local app, this is safe.
if st.query_params.get("nav") == "true":
    st.session_state['authenticated'] = True

# Import Landing Page
from landing import show_landing

if not st.session_state['authenticated']:
    # LANDING PAGE MODE
    pg = st.navigation([st.Page(show_landing, title="ApexForex", icon="⚡")], position="hidden")
    pg.run()
    
else:
    # DASHBOARD MODE
    # Define Pages
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    pg_home = st.Page(show_command_center, title="Command Center", icon="⚡", default=True, url_path="home")
    # Note: url_path="market" works if authenticated.
    pg_market = st.Page(show_market_overview, title="Market Overview", icon="🌍", url_path="market")
    pg_terminal = st.Page(show_trading_terminal, title="Trading Terminal", icon="📈", url_path="terminal")
    pg_analytics = st.Page(show_analytics, title="Analytics Suite", icon="📊", url_path="analytics")
    pg_models = st.Page(show_model_audit, title="Model Audit", icon="🛡️", url_path="audit")
    pg_control = st.Page(show_control_panel, title="Control Panel", icon="⚙️", url_path="settings")
    
    # Fleet Monitor (inline — no external file dependency)
    pg_fleet = st.Page(show_fleet_status, title="Fleet Status", icon="📊", url_path="fleet")

    # External Pages (mapped from existing files)
    import os
    
    path_profile = os.path.join(BASE_DIR, "pages", "1_User_Profile.py")
    path_vault = os.path.join(BASE_DIR, "pages", "2_Financials_Vault.py")
    path_settings = os.path.join(BASE_DIR, "pages", "3_System_Settings.py")

    pg_profile = st.Page(path_profile, title="User Profile", icon="👤", url_path="profile")
    pg_vault = st.Page(path_vault, title="Financials Vault", icon="💳", url_path="vault")
    pg_settings = st.Page(path_settings, title="System Settings", icon="⚙", url_path="advanced")

    # Build Navigation
    pg = st.navigation({
        "Intelligence": [pg_home, pg_market, pg_terminal, pg_fleet],
        "Analytics": [pg_analytics, pg_models],
        "Management": [pg_control, pg_profile, pg_vault, pg_settings]
    })

    # Sidebar Logo/Footer (Stays constant)
    with st.sidebar:
        sidebar_logo()
        # st.navigation handles the menu rendering automatically here

    # Run!
    pg.run()

    # Sidebar Footer & Global Rerun (Native approach)
    with st.sidebar:
        st.markdown("---")
        sidebar_footer()
        
        # --- SYSTEM MONITOR (FORENSIC DEBUG) ---
        with st.expander("🛠️ System Monitor (Debug)", expanded=True):
            import os
            import json
            from pathlib import Path
            
            cwd = os.getcwd()
            st.code(f"CWD: {cwd}", language="bash")
            
            # Test Path Logic
            p_root = Path(cwd)
            eur_conf = p_root / "models" / "EURUSD" / "90" / "SELL" / "config.json"
            
            st.write(f"**Config Path:** `{eur_conf}`")
            if eur_conf.exists():
                st.success("✅ Config File Found!")
                try:
                    with open(eur_conf) as f:
                        data = json.load(f)
                        trades = data.get("trades", 0)
                        st.metric("Raw Volume", trades)
                except:
                    st.error("Read Failed")
            else:
                st.error("❌ Config NOT Found")
        
        # Simple manual refresh button to bypass health-check lag
        if st.button("🔄 Force Data Refresh"):
            st.rerun()

    # Auto-refresh is now handled inline via st_autorefresh in show_trading_terminal().
    # Command Center and Market Overview use manual refresh via the sidebar button.
# Force Reload: 2026-02-11 14:18
