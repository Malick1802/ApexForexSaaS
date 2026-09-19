"""System Settings page — standalone sub-page with shared theme (Admin Only)."""
import streamlit as st
# ── Auth & Admin Guard ─────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Session expired or not logged in. Please log in again.")
    st.page_link("app.py", label="🔑 Go to Login", icon="🔑")
    st.stop()

if st.session_state.get("user_role") != "admin":
    st.error("⛔ Access restricted to platform administrators.")
    st.stop()
# ── End Auth Guard ─────────────────────────────────────────────────────────
import yaml
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import hero_banner, section_header, PROJECT_ROOT, inject_css
inject_css()

hero_banner("System Settings", "Notifications, technical configuration, and data management")

CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

config = load_config()

# Notifications
with st.container(border=True):
    section_header("🔔", "Notifications")

    notif = config.get('notifications', {}).get('telegram', {})
    enable_tg = st.toggle("Enable Telegram Alerts", value=notif.get('enabled', False))
    bot_token = st.text_input("Telegram Bot Token", value=notif.get('bot_token', ''), type="password", key="tg_token")
    chat_id = st.text_input("Telegram Chat ID", value=notif.get('chat_id', ''), key="tg_chat")

    if enable_tg and bot_token and chat_id:
        st.markdown("""
        <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.2);border-radius:20px;font-size:0.75rem;font-weight:600;color:var(--success);">
            <span class="status-dot"></span> Connected
        </span>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;background:rgba(255,82,82,0.08);border:1px solid rgba(255,82,82,0.2);border-radius:20px;font-size:0.75rem;font-weight:600;color:#FF5252;">
            <span style="width:8px;height:8px;border-radius:50%;background:#FF5252;"></span> Not Configured
        </span>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Apex Connect (Auto-Trading) ──────────────────────
with st.container(border=True):
    section_header("🤖", "Apex Connect (Auto-Trading)")

    mt5_config = config.get('mt5', {})
    enable_mt5 = st.toggle("Enable Apex Connect Bridge", value=mt5_config.get('enabled', False), help="Automatically execute new signals on your local MT5 terminal.")

    c1, c2 = st.columns(2)
    with c1:
        risk_options = ["Fixed Lot", "Account %"]
        current_risk = mt5_config.get('risk_type', 'fixed')
        idx = 1 if current_risk == 'percent' else 0
        risk_type_sel = st.selectbox("Risk Model", risk_options, index=idx, key="mt5_risk_type")

    with c2:
        default_val = mt5_config.get('risk_value', 0.01)
        if risk_type_sel == "Fixed Lot":
            risk_value = st.number_input("Lot Size", min_value=0.01, max_value=50.0, value=float(default_val), step=0.01, key="mt5_risk_val_lot")
            st.caption("Trades will use this exact lot size.")
        else:
            risk_value = st.number_input("Risk % per Trade", min_value=0.1, max_value=5.0, value=float(default_val), step=0.1, key="mt5_risk_val_pct")
            st.caption("Calculates lots based on Stop Loss distance and Account Equity.")

    if enable_mt5:
        st.markdown("""
        <div style="margin-top: 12px; padding: 12px; background: rgba(0, 230, 118, 0.05); border-left: 3px solid var(--accent-green); border-radius: 4px;">
            <small><strong>✅ Bridge Active:</strong> The 'apex_connect' service will monitor for new signals and execute them based on these settings.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top: 12px; padding: 12px; background: rgba(255, 255, 255, 0.03); border-left: 3px solid var(--text-tertiary); border-radius: 4px;">
            <small><strong>⏸️ Bridge Paused:</strong> Signals will be generated but NOT executed automatically.</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Technical Config
with st.container(border=True):
    section_header("🔧", "Technical Configuration")

    tech_config = config.get('technical', {})
    default_cur = tech_config.get('currency', 'USD')
    default_tz = tech_config.get('timezone', 'UTC')
    default_theme = tech_config.get('theme', 'Dark Mode')
    default_loglevel = tech_config.get('log_level', 'INFO')
    active_provider = config.get('data_provider', {}).get('active', 'mt5')

    cur_options = ["USD", "EUR", "GBP", "JPY"]
    tz_options = ["UTC", "EST", "PST", "GMT"]
    theme_options = ["Dark Mode", "Light Mode", "System Default"]
    loglevel_options = ["INFO", "DEBUG", "WARNING", "ERROR"]
    provider_options = ["mt5", "yfinance"]

    def get_idx(val, options, default=0):
        try:
            return options.index(val)
        except ValueError:
            return default

    col1, col2 = st.columns(2)
    with col1:
        cur_sel = st.selectbox("Default Currency", cur_options, index=get_idx(default_cur, cur_options), key="def_currency")
        tz_sel = st.selectbox("Timezone", tz_options, index=get_idx(default_tz, tz_options), key="tz")
        provider_sel = st.selectbox("Active Data Provider", provider_options, index=get_idx(active_provider, provider_options), key="provider")
    with col2:
        theme_sel = st.selectbox("Theme", theme_options, index=get_idx(default_theme, theme_options), key="theme_sel")
        loglevel_sel = st.selectbox("Log Level", loglevel_options, index=get_idx(default_loglevel, loglevel_options), key="log_level")

st.markdown("<br>", unsafe_allow_html=True)

# Data Management
with st.container(border=True):
    section_header("💾", "Data Management & Database Exports")

    def _get_cache_size():
        total_bytes = 0
        try:
            for p in PROJECT_ROOT.rglob("__pycache__"):
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            total_bytes += f.stat().st_size
            streamlit_cache = Path.home() / ".streamlit" / "cache"
            if streamlit_cache.exists():
                for f in streamlit_cache.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size
        except Exception:
            pass
        mb = total_bytes / (1024 * 1024)
        if mb >= 1.0:
            return f"{mb:.1f} MB"
        return f"{max(0.1, total_bytes / 1024):.1f} KB"

    c1, c2 = st.columns(2)
    with c1:
        cache_str = _get_cache_size()
        st.markdown(f"""
        <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
            Cache size: <span style="color: var(--accent-cyan); font-weight: 600;">{cache_str}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Platform cache cleared!")
            time.sleep(0.5)
            st.rerun()
            
    with c2:
        st.markdown("""
        <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
            Database Export: <span style="color: var(--accent-cyan); font-weight: 600;">Live CSV — always current</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        import sqlite3
        import io

        def _db_to_csv(db_path: Path, table: str, where_clause: str = "") -> str:
            """Read a full DB table and return as a CSV string."""
            import pandas as pd
            if not db_path.exists():
                return ""
            try:
                conn = sqlite3.connect(str(db_path))
                query = f"SELECT * FROM {table} {where_clause}"
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df.to_csv(index=False)
            except Exception as e:
                return ""

        signals_db  = PROJECT_ROOT / "signals.db"
        users_db    = PROJECT_ROOT / "user_accounts.db"
        portal_db   = PROJECT_ROOT / "portal_users.db"

        # Signals (Exclude SYSTEM heartbeats)
        signals_csv = _db_to_csv(signals_db, "signals", "WHERE symbol != 'SYSTEM'")
        if signals_csv:
            st.download_button(
                label="📥 Download Signals CSV",
                data=signals_csv,
                file_name="signals_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # MT5 Subscriber Accounts
        users_csv = _db_to_csv(users_db, "user_accounts")
        if users_csv:
            st.download_button(
                label="📥 Download MT5 Subscriber Accounts CSV",
                data=users_csv,
                file_name="user_accounts_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Portal logins
        portal_csv = _db_to_csv(portal_db, "portal_users")
        if portal_csv:
            st.download_button(
                label="📥 Download Portal Logins CSV",
                data=portal_csv,
                file_name="portal_users_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Inbound Leads & Upgrade Requests
        inq_csv = _db_to_csv(portal_db, "inquiries")
        if inq_csv:
            st.download_button(
                label="📥 Download Inbound Leads CSV",
                data=inq_csv,
                file_name="leads_export.csv",
                mime="text/csv",
                use_container_width=True
            )



# Save Settings
if st.button("💾 Save System Settings", type="primary", use_container_width=True):
    if 'notifications' not in config: config['notifications'] = {}
    config['notifications']['telegram'] = {
        'enabled': enable_tg,
        'bot_token': bot_token,
        'chat_id': chat_id,
        'channel_id': notif.get('channel_id', ''),
        'notify_shadow_trades': notif.get('notify_shadow_trades', False),
        'alert_threshold': notif.get('alert_threshold', 0.6)
    }
    
    if 'mt5' not in config: config['mt5'] = {}
    config['mt5']['enabled'] = enable_mt5
    config['mt5']['risk_type'] = 'percent' if risk_type_sel == 'Account %' else 'fixed'
    config['mt5']['risk_value'] = float(risk_value)
    
    config['technical'] = {
        'currency': cur_sel,
        'timezone': tz_sel,
        'theme': theme_sel,
        'log_level': loglevel_sel
    }
    
    if 'data_provider' not in config: config['data_provider'] = {}
    config['data_provider']['active'] = provider_sel
    
    save_config(config)
    st.toast("System settings saved!", icon="💾")
    time.sleep(1)
    st.rerun()
