import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime

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

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="ApexForex · AI Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()


# ── Cached Loaders ─────────────────────────────────────────
@st.cache_resource
def load_engine():
    return get_engine()

@st.cache_resource
def load_inference():
    return get_inference()


# ── Chart Renderer ─────────────────────────────────────────
def render_chart(df, symbol):
    if not PLOTLY_AVAILABLE or df.empty:
        st.warning("Chart unavailable.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#00FF88', increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_line_color='#FF4466', decreasing_fillcolor='rgba(255,68,102,0.3)',
        name='Price'
    ), row=1, col=1)

    colors = ['rgba(0,255,136,0.3)' if c >= o else 'rgba(255,68,102,0.3)'
              for c, o in zip(df['close'], df['open'])]
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                              marker_color=colors, name='Volume', showlegend=False),
                      row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(10, 14, 26, 0)',
        paper_bgcolor='rgba(10, 14, 26, 0)',
        font=dict(family='Inter, sans-serif', color='#8b95a8'),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False, height=420,
        yaxis=dict(gridcolor='rgba(255,255,255,0.03)', side='right'),
        yaxis2=dict(gridcolor='rgba(255,255,255,0.03)', side='right'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.02)'),
        xaxis2=dict(gridcolor='rgba(255,255,255,0.02)'),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# =============================================================================
# VIEW 1: Command Center (Home)
# =============================================================================
def show_command_center():
    engine = load_engine()
    db = get_db()

    hero_banner("Command Center",
                "Real-time AI surveillance across 31 forex pairs · 90%+ precision targeting",
                show_status=True)

    all_pairs = engine.get_all_pairs()
    recent = db.get_recent_signals(limit=500)
    active_count = 0
    success_rate = 0.0
    completed_count = 0

    if recent:
        df_sig = pd.DataFrame(recent)
        df_sig['timestamp'] = pd.to_datetime(df_sig['timestamp'])
        if 'outcome' not in df_sig.columns:
            df_sig['outcome'] = 'ACTIVE'
        try:
            # Latest signal per pair only (deduplication)
            latest_per_pair = df_sig.sort_values('timestamp', ascending=False).drop_duplicates(subset='symbol', keep='first')

            # Active = latest signal per pair that's BUY/SELL (not stale duplicates)
            active_trades = latest_per_pair[
                latest_per_pair['signal'].isin(['BUY', 'SELL'])
            ]
            active_count = len(active_trades)

            completed = df_sig[df_sig['outcome'].isin(['SUCCESS', 'FAIL'])]
            completed_count = len(completed)
            if not completed.empty:
                success_rate = (len(completed[completed['outcome'] == 'SUCCESS']) / len(completed)) * 100
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Monitored Pairs", len(all_pairs), "Majors · Minors · Crosses", "accent-cyan"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Active Signals", active_count, "Unique pairs", "accent-gold"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Win Rate", f"{success_rate:.1f}%", f"{completed_count} closed trades", "accent-green"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("System Health", "Online", "Watchdog · Sentinel · API", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🎯", "High Confidence Opportunities")

    if recent:
        # Deduplicate: latest signal per pair, BUY/SELL only, confidence > 75%
        seen = {}
        for s in recent:
            sym = s.get('symbol')
            if sym not in seen and s.get('signal') in ['BUY', 'SELL'] and s.get('confidence', 0) > 0.75:
                seen[sym] = s
        high_conf = list(seen.values())

        if high_conf:
            df_hc = pd.DataFrame(high_conf[:20])
            # Convert confidence 0.0-1.0 to percentage for display
            df_hc['confidence_pct'] = (df_hc['confidence'] * 100).round(1)
            display_cols = ['symbol', 'signal', 'confidence_pct', 'price_at_signal', 'timestamp']
            display_cols = [c for c in display_cols if c in df_hc.columns]
            st.dataframe(df_hc[display_cols], use_container_width=True, hide_index=True,
                         column_config={
                             "symbol": "Pair",
                             "signal": "Direction",
                             "confidence_pct": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
                             "price_at_signal": st.column_config.NumberColumn("Entry", format="%.5f"),
                             "timestamp": "Detected"
                         })
        else:
            st.info("📡 No high confidence signals (>75%) detected recently. The AI is monitoring...")
    else:
        st.info("📡 No signal data yet. Start the Sentinel to begin scanning.")


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

    # Build signal map from DB
    all_signals = db.get_recent_signals(limit=500)
    sig_map = {}
    if all_signals:
        for s in all_signals:
            sym = s['symbol']
            if sym not in sig_map:
                sig_map[sym] = s

    # Sidebar filters
    with st.sidebar:
        section_header("🎛️", "Filters")

        if 'accuracy_target' not in st.session_state:
            st.session_state['accuracy_target'] = '90%'
        accuracy_target = st.select_slider('Desired Accuracy',
            options=['60%', '70%', '80%', '90%', 'Apex'],
            value=st.session_state['accuracy_target'], key='accuracy_target')

        if 'confidence_thresh' not in st.session_state:
            st.session_state['confidence_thresh'] = 70
        confidence_thresh = st.slider("Confidence Filter", 50, 95,
            st.session_state['confidence_thresh'], key='confidence_thresh')

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
                if not sig_data:
                    st.markdown(f"""
                    <div class="signal-tile tile-wait">
                        <div class="tile-symbol">{symbol}</div>
                        <div class="tile-signal tile-signal-wait">—</div>
                        <div class="tile-conf">Awaiting Data</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sig = sig_data.get('signal', 'WAIT')
                    conf = sig_data.get('confidence', 0)

                    display_sig = sig
                    css_tile = "tile-wait"
                    css_signal = "tile-signal-wait"
                    conf_display = "Monitoring..."
                    conf_bar = ""

                    if sig in ["BUY", "SELL"] and conf >= (confidence_thresh / 100.0):
                        if sig == "BUY":
                            css_tile = "tile-buy"
                            css_signal = "tile-signal-buy"
                            conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar conf-bar-buy" style="width:{conf*100:.0f}%"></div></div>'
                        else:
                            css_tile = "tile-sell"
                            css_signal = "tile-signal-sell"
                            conf_bar = f'<div class="conf-bar-bg"><div class="conf-bar conf-bar-sell" style="width:{conf*100:.0f}%"></div></div>'
                        conf_display = f"{conf:.0%}"
                    else:
                        display_sig = "WAIT"

                    st.markdown(f"""
                    <div class="signal-tile {css_tile}">
                        <div class="tile-symbol">{symbol}</div>
                        <div class="tile-signal {css_signal}">{display_sig}</div>
                        <div class="tile-conf">{conf_display}</div>
                        {conf_bar}
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# VIEW 3: Trading Terminal
# =============================================================================
def show_trading_terminal():
    engine = load_engine()
    inf_engine = load_inference()
    db = get_db()

    with st.sidebar:
        section_header("🎛️", "Analysis Controls")
        all_pairs = engine.get_all_pairs()

        if 'pair_selector' not in st.session_state:
            st.session_state['pair_selector'] = "EURUSD" if "EURUSD" in all_pairs else all_pairs[0]

        symbol = st.selectbox("Select Pair", all_pairs, key='pair_selector')
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
        st.divider()

        if 'accuracy_target' not in st.session_state:
            st.session_state['accuracy_target'] = '90%'
        accuracy_target = st.select_slider('Desired Accuracy',
            options=['60%', '70%', '80%', '90%', 'Apex'],
            value=st.session_state['accuracy_target'], key='accuracy_target')

        tier_labels = {'60%': '⚡ Aggressive', '70%': '🚀 Growth',
                       '80%': '💎 Precision', '90%': '🏆 Expert', 'Apex': '👑 Institutional'}
        st.caption(f"**{tier_labels.get(accuracy_target, '')}**")
        st.divider()

        if 'confidence_thresh' not in st.session_state:
            st.session_state['confidence_thresh'] = 70
        confidence_thresh = st.slider("Confidence Filter", 50, 95,
            st.session_state['confidence_thresh'], key='confidence_thresh')

    # Main layout
    result = None
    pred = "WAIT"
    conf = 0.0
    df = pd.DataFrame()

    col_main, col_side = st.columns([3, 1])

    with col_main:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                df = engine.fetch(symbol, interval=timeframe, days=60, use_cache=True)

                db_signals = db.get_recent_signals(limit=1, symbol=symbol)
                latest_db = db_signals[0] if db_signals else None

                result = inf_engine.predict_symbol(
                    symbol, save_to_db=False,
                    win_rate=accuracy_target, allow_stale=True
                )

                if result:
                    if latest_db and latest_db.get('signal') == result.get('signal') and latest_db.get('signal') != "WAIT":
                        db_ts = pd.to_datetime(latest_db['timestamp'])
                        if (datetime.now() - db_ts.to_pydatetime()).total_seconds() < 14400:
                            result = latest_db
                    pred = result.get('signal', 'WAIT')
                    conf = result.get('confidence', 0)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                st.error(f"⚠️ API Rate Limit or Network Error: {e}")

            if not df.empty:
                try:
                    import pandas_ta as ta
                    last_price = df['close'].iloc[-1]
                    prev_price = df['close'].iloc[-2]
                    change = (last_price - prev_price) / prev_price
                    rsi_series = ta.rsi(df['close'], length=14)
                    current_rsi = rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty else 0.0
                    volatility = df['close'].pct_change().std() * 100

                    st.markdown(f"""
                    <div style="margin-bottom: 16px;">
                        <span style="font-size: 1.5rem; font-weight: 800; color: var(--text-primary);">{symbol}</span>
                        <span style="font-family: var(--font-mono); font-size: 1.3rem; font-weight: 700; color: var(--accent-cyan); margin-left: 12px;">{last_price:.5f}</span>
                        <span style="font-family: var(--font-mono); font-size: 0.85rem; color: {'var(--signal-buy)' if change >= 0 else 'var(--signal-sell)'}; margin-left: 8px;">
                            {'▲' if change >= 0 else '▼'} {abs(change):.2%}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Price", f"{last_price:.5f}", f"{change:+.2%}")
                    m2.metric("Volatility", f"{volatility:.3f}%")
                    m3.metric("RSI (14)", f"{current_rsi:.1f}",
                              "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
                except Exception:
                    pass

                render_chart(df, symbol)
            else:
                st.warning("No chart data available. Check API connection.")

    with col_side:
        section_header("🤖", "AI Verdict")

        if result:
            css = "signal-wait"
            if pred == "BUY": css = "signal-buy"
            elif pred == "SELL": css = "signal-sell"

            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; text-align: center;">
                <div class="signal-badge {css}">{pred}</div>
                <div style="margin-top: 16px;">
                    <div style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 700; color: var(--accent-cyan);">{conf:.0%}</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em;">Confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if pred in ["BUY", "SELL"] and result.get('tp_price'):
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("📍", "Trading Levels")
                tp_pips = result.get('tp_pips', 0)
                sl_pips = result.get('sl_pips', 0)
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


# =============================================================================
# VIEW 4: Analytics (Performance Audit)
# =============================================================================
def show_analytics():
    hero_banner("Analytics Suite", "Signal history, outcomes, and win rate analytics")

    db = get_db()
    signals = db.get_recent_signals(limit=500)

    if not signals:
        st.info("📊 No signal history. Start the Sentinel to collect data.")
        return

    df = pd.DataFrame(signals)
    if 'outcome' not in df.columns:
        df['outcome'] = 'ACTIVE'

    completed = df[df['outcome'].isin(['SUCCESS', 'FAIL'])]
    wins = len(completed[completed['outcome'] == 'SUCCESS']) if not completed.empty else 0
    win_rate = (wins / len(completed)) * 100 if not completed.empty else 0
    active = len(df[df['outcome'] == 'ACTIVE'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Win Rate", f"{win_rate:.1f}%", f"{wins} wins", "accent-green"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Closed Trades", str(len(completed)), "Resolved", "accent-cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Active Trades", str(active), "Currently open", "accent-gold"), unsafe_allow_html=True)
    with c4:
        best = ""
        if not completed.empty:
            ps = completed.groupby('symbol').apply(lambda x: (x['outcome']=='SUCCESS').sum()/len(x)*100)
            if not ps.empty:
                best = f"{ps.idxmax()} ({ps.max():.0f}%)"
        st.markdown(kpi_card("Best Pair", best or "N/A", "Highest win rate", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("📜", "Signal History")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sym_filter = st.multiselect("Filter Pair", sorted(df['symbol'].unique()))
    with fc2:
        sig_filter = st.multiselect("Filter Signal", ["BUY", "SELL", "WAIT"], default=["BUY", "SELL"])
    with fc3:
        out_filter = st.multiselect("Filter Outcome", ["ACTIVE", "SUCCESS", "FAIL", "N/A"],
                                     default=["ACTIVE", "SUCCESS", "FAIL"])

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
# NAVIGATION & ROUTING (Streamlit 1.34+)
# =============================================================================

# Define Pages
pg_home = st.Page(show_command_center, title="Command Center", icon="⚡", default=True)
pg_market = st.Page(show_market_overview, title="Market Overview", icon="🌍")
pg_terminal = st.Page(show_trading_terminal, title="Trading Terminal", icon="📈")
pg_analytics = st.Page(show_analytics, title="Analytics Suite", icon="📊")
pg_models = st.Page(show_model_audit, title="Model Audit", icon="🛡️")
pg_control = st.Page(show_control_panel, title="Control Panel", icon="⚙️")

# External Pages (mapped from existing files)
pg_profile = st.Page("pages/1_User_Profile.py", title="User Profile", icon="👤")
pg_vault = st.Page("pages/2_Financials_Vault.py", title="Financials Vault", icon="💳")
pg_settings = st.Page("pages/3_System_Settings.py", title="System Settings", icon="⚙")

# Build Navigation
pg = st.navigation({
    "Intelligence": [pg_home, pg_market, pg_terminal],
    "Analytics": [pg_analytics, pg_models],
    "Management": [pg_control, pg_profile, pg_vault, pg_settings]
})

# Sidebar Logo/Footer (Stays constant)
with st.sidebar:
    sidebar_logo()
    # st.navigation handles the menu rendering automatically here

# Run!
pg.run()

# Sidebar Footer
with st.sidebar:
    st.markdown("---")
    sidebar_footer()

# Handle Auto-refresh via a hidden state if needed, but st.navigation handles re-runs
# For specific views like Command Center, we still do the wait/rerun
if pg.title in ["Command Center", "Market Overview", "Trading Terminal"]:
    time.sleep(60)
    st.rerun()
