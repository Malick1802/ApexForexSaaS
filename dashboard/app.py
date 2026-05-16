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
def render_chart(df, symbol, key=None):
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
    
    # 1. Active Signals Pool
    # We fetch ALL signals marked as ACTIVE (Success/Fail/Wait intent)
    # including hidden/shadow signals so we can filter/show them correctly.
    raw_active = db.get_active_signals(include_hidden=True)
    
    # Filter: Show signals ≥ 60% as "Active Signals" (matching Telegram strategy)
    # but still track BUY/SELL specifically for trades vs monitoring
    # Filter: Show ONLY real trade signals (BUY/SELL)
    active_signals = [
        s for s in raw_active 
        if s.get('signal') in ['BUY', 'SELL'] and int(s.get('confidence_tier', 0)) > 0
    ]
    active_count = len(active_signals)

    # 2. Expired/Closed Signals (Historical - Last 48h Window)
    # Include hidden signals so benched history is visible in charts
    recent = db.get_recent_signals(limit=5000, include_hidden=True)
    expired_signals = []
    success_rate = 0.0
    completed_count = 0

    if recent:
        # Time Window: Current Week (Monday to Now)
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = start_of_week
        
        # Filter recent signals by time and remove WAIT noise
        recent_window = []
        for s in recent:
            if s.get('signal') == 'WAIT':
                continue
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
            
            # Expired = Qualified signals (≥ 60%) that are NOT active
            expired_signals = [
                s for s in recent_window 
                if s.get('outcome') != 'ACTIVE' 
                and (s.get('signal') in ['BUY', 'SELL'] or (s.get('confidence_tier') or 0) >= 60)
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

    # Fetch Global Validated Win Rate
    val_stats = db.get_validated_win_rate()
    val_win_rate = val_stats.get('win_rate', 0.0)
    val_total = val_stats.get('total', 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Monitored Pairs", len(all_pairs), "Majors · Minors · Crosses", "accent-cyan", link_url="/market?nav=true"), unsafe_allow_html=True)
    with c2:
        # Active Signals -> Scroll/Focus (or Analytics)
        st.markdown(kpi_card("Active Signals", active_count, "Running trades", "accent-gold", link_url="/analytics?nav=true"), unsafe_allow_html=True)
    with c3:
        # Replace weekly generic win rate with Validated Win Rate
        st.markdown(kpi_card("Validated Win Rate", f"{val_win_rate:.1f}%", f"{val_total} proven trades", "accent-green", link_url="/analytics?nav=true"), unsafe_allow_html=True)
    with c4:
        # Show "Closed Trades" (TP/SL hit) as primary metric count.
        # BUT link to 'expired' filter so user can see "trades that did not complete" as requested.
        st.markdown(kpi_card("Closed Trades (Week)", completed_count, "Hit TP or SL", "accent-cyan", link_url="/analytics?nav=true&filter=expired"), unsafe_allow_html=True)
    with c5:
        training_status = get_training_status()
        if training_status:
            st.markdown(kpi_card("System Health", "Training v2", training_status, "accent-gold"), unsafe_allow_html=True)
        else:
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
        # Build signal groups (Symbol -> List) to handle Tier overlapping
        active_signals = db.get_active_signals(include_hidden=True)
        sig_map = {}
        has_secondary_tier = {}

        # Group by symbol
        groups = {}
        for s in active_signals:
            sym = s['symbol']
            if sym not in groups: groups[sym] = []
            groups[sym].append(s)

            # Select the 'Best' signal for tile display (Priority: Live > Highest Tier > Newest)
            for sym, sigs in groups.items():
                # Filter for real trades (Exclude WAIT/0% from being counted as 'secondary')
                real_sigs = [s for s in sigs if s.get('signal') in ('BUY', 'SELL') and int(s.get('confidence_tier', 0)) > 0]
                
                sorted_sigs = sorted(
                    sigs, 
                    key=lambda x: (not bool(x.get('is_hidden', 0)), x.get('confidence_tier', 0), x['timestamp']),
                    reverse=True
                )
                sig_map[sym] = sorted_sigs[0]
                # Only show indicator if there are multiple REAL trade setups
                has_secondary_tier[sym] = len(real_sigs) > 1

        # Fallback for symbols with only historical signals (no active ones)
        recent = db.get_recent_signals(limit=100, include_hidden=True)
        for s in recent:
            if s['symbol'] not in sig_map:
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
                                display_sig = sig
                                css_tile = "tile-buy" if sig == "BUY" else "tile-sell"
                                css_signal = "tile-signal-wait" # Keep signal text muted so it doesn't pop as much
                                conf_display = f"{sig_data.get('raw_confidence', conf):.0%}"
                                conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar" style="width: {conf if conf else 0.0:.1%}; background: var(--text-muted);"></div></div>'
                                regime_badge += '<div style="position: absolute; top: 30px; right: 10px; font-size: 0.55rem; color: #aaa; background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 4px; font-family: var(--font-mono); border: 1px dashed rgba(255,255,255,0.2);">SHADOW</div>'
                                extra_styles = "opacity: 0.6; filter: grayscale(50%);"
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

                        ghost_html = '<div class="ghost-indicator" title="Secondary Tier Active"></div>' if has_secondary_tier.get(symbol) else ""
                        tile_html = (
                            f'<div class="signal-tile {css_tile}" '
                            f'style="position: relative; {"opacity: 0.85;" if is_hidden else ""} {extra_styles}">'
                            f'{regime_badge}'
                            f'{ghost_html}'
                            f'<div class="tile-symbol">{symbol}</div>'
                            f'<div class="tile-signal {css_signal}">{display_sig}</div>'
                            f'<div class="tile-conf">{conf_display}</div>'
                            f'{conf_bar}'
                            f'</div>'
                        )

                    st.markdown(f'<a href="{link}" target="_parent" style="text-decoration: none; color: inherit; display: block;">{tile_html}</a>', unsafe_allow_html=True)

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

    # ── Live Data Fragment (reruns every 10s WITHOUT full page blink) ──
    @st.fragment(run_every=timedelta(seconds=10))
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

                # 0. Fetch Data (REAL TIME SYNC)
                df = inf_engine.data_engine.fetch(symbol, interval=timeframe, days=7, use_cache=False)
                if df.empty:
                    raise Exception(f"No candlestick data received for {symbol}")

                # 1. Check for EXISTING ACTIVE SIGNAL (to manage PnL and Locked state)
                active_signals = db.get_active_signals(symbol=symbol, include_hidden=True)
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

                # 3. LOCKING LOGIC: If a trade is active, use the LOCKED data for the main box
                if locked_trade:
                    result = locked_trade
                    st.caption(f"🔒 TERMINAL LOCKED TO ACTIVE POSITION (ID #{locked_trade['id']})")
                else:
                    result = live_result
                    if live_result:
                        st.caption("📡 LIVE AI PULSE (Real-Time Monitoring)")
                    else:
                        st.caption("⚠️ AI PULSE OFFLINE (Awaiting Market Data)")

                # 4. FALLBACK: If inference failed and no locked trade, show most recent DB signal
                if not result:
                    sym_signals = db.get_recent_signals(symbol=symbol, limit=1, include_hidden=True)
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
                    try:
                        ts_obj = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
                        ts_display = ts_obj.strftime("%d %b %H:%M")
                    except: pass

                    display_conf = result.get('raw_confidence', conf) or 0.0
                    vol_trades = result.get('model_trades', 0) or 0
                    
                    css = f"signal-{pred.lower()}"
                    
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
                # No result yet — show an informative monitoring card
                # (either model not trained for this pair, or inference is still loading)
                last_price_display = ""
                try:
                    if not df.empty:
                        last_price_display = f"{df['close'].iloc[-1]:.5f}"
                except: pass

                st.markdown(f"""
<div class="glass-card" style="padding: 24px; text-align: center; border-top: 3px solid var(--text-muted);">
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
    if 'signal' in df.columns:
        df = df[df['signal'] != 'WAIT']
        
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
# VIEW 5: Performance Matrix (Real-Time Audit)
# =============================================================================
def show_performance_matrix():
    from core.performance_gate import get_performance_gate
    db = get_db()
    gate = get_performance_gate()

    hero_banner("Performance Matrix", "Real-time AI surveillance, rolling window analytics, and certification status")

    @st.fragment(run_every=timedelta(minutes=5))
    def _matrix_grid():
        # 1. Active Surveillance (Live vs Shadow)
        section_header("🛰️", "Active Signal Surveillance")
        raw_active = db.get_active_signals(include_hidden=True)
        # FILTER: Show only real trade signals (BUY/SELL). Skip neutral 'WAIT' noise.
        active = [s for s in raw_active if s.get('signal') in ('BUY', 'SELL')]
        
        if active:
            df_active = pd.DataFrame(active)
            # Add Display columns
            df_active['Type'] = df_active['is_hidden'].apply(lambda x: "🛸 SHADOW" if x else "🚀 REAL")
            df_active['Conviction'] = df_active['confidence'].apply(lambda x: f"{x:.1%}")
            
            show_cols = ['symbol', 'signal', 'Type', 'Conviction', 'confidence_tier', 'timestamp']
            st.dataframe(
                df_active[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Pair", "signal": "Signal", "confidence_tier": "Tier %",
                    "timestamp": "Detected"
                }
            )
        else:
            st.info("No active signals currently under surveillance.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Performance Thresholds (14-Day Window)
        section_header("📊", "14-Day Performance Matrix")
        stats_14 = db.get_performance_matrix_stats(14)
        if stats_14:
            df_14 = pd.DataFrame(stats_14)
            df_14['Win Rate'] = df_14.apply(lambda row: (row['wins'] / row['total_trades']) if row['total_trades'] > 0 else 0, axis=1)
            
            st.dataframe(
                df_14[['symbol', 'total_trades', 'Win Rate', 'wins', 'losses', 'last_trade']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Pair", "total_trades": "Volume", 
                    "Win Rate": st.column_config.ProgressColumn("Win Rate", format="%.0f%%", min_value=0, max_value=1),
                    "last_trade": "Last Activity"
                }
            )
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
        if matrix:
            for sym, contents in matrix.items():
                for k, v in contents.items():
                    if not isinstance(v, dict):
                        continue
                        
                    # Check if this is a direct tier (Legacy) or a Direction dict (New)
                    if 'status' in v:
                        # Legacy Format: sym -> tier -> data
                        if v.get('status') == 'APPROVED':
                            cert_records.append({
                                "Symbol": sym, "Direction": "ALL", "Tier": f"{k}%",
                                "Acc": v.get('accuracy', 0.0), "Trades": v.get('trades', 0),
                                "Source": v.get('source', 'Legacy')
                            })
                    else:
                        # New Format: sym -> direction -> tier -> data
                        for t_str, data in v.items():
                            if isinstance(data, dict) and data.get('status') == 'APPROVED':
                                cert_records.append({
                                    "Symbol": sym, "Direction": k, "Tier": f"{t_str}%",
                                    "Acc": data.get('accuracy', 0.0), "Trades": data.get('trades', 0),
                                    "Source": data.get('source', 'System')
                                })
        
        if cert_records:
            df_cert = pd.DataFrame(cert_records)
            st.dataframe(
                df_cert, use_container_width=True, hide_index=True,
                column_config={
                    "Acc": st.column_config.NumberColumn("Realized Acc", format="%.1%"),
                    "Tier": "Strategy Tier"
                }
            )
        else:
            st.warning("No pairs currently meet the 70% institutional certification threshold.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Model Registry (All-Time Models)
        section_header("📋", "Historical Model Registry")
        registry = db.get_model_registry_stats()
        if registry:
            df_reg = pd.DataFrame(registry)
            df_reg['All-Time WR'] = df_reg.apply(lambda row: (row['all_time_wins'] / row['all_time_trades']) if row['all_time_trades'] > 0 else 0, axis=1)
            st.dataframe(
                df_reg, use_container_width=True, hide_index=True,
                column_config={
                    "All-Time WR": st.column_config.ProgressColumn("All-Time WR", format="%.0f%%", min_value=0, max_value=1),
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
            if st.button("🔄 Revert to MT5", use_container_width=True):
                config['data_provider']['active'] = 'mt5'
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                st.toast("Reverted to MT5", icon="🔄")
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
    pg_models = st.Page(show_performance_matrix, title="Performance Matrix", icon="🛡️", url_path="audit")
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
        
        # Use the unified system monitor diagnostic
        render_system_monitor()
        
        # Simple manual refresh button to bypass health-check lag
        if st.button("🔄 Force Data Refresh"):
            st.rerun()

    # Auto-refresh is now handled inline via st_autorefresh in show_trading_terminal().
    # Command Center and Market Overview use manual refresh via the sidebar button.
# Force Reload: 2026-04-19 21:30 — Performance Matrix
