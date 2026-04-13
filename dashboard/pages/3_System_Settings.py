"""System Settings page — standalone sub-page with shared theme."""
import streamlit as st
import yaml
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import inject_css, hero_banner, sidebar_logo, sidebar_footer, section_header, PROJECT_ROOT

st.set_page_config(page_title="System Settings · ApexForex", page_icon="⚙️", layout="wide")
inject_css()

with st.sidebar:
    sidebar_logo()
    st.markdown("""
    <div style="text-align:center; margin-bottom: 16px;">
        <a href="/" target="_self" style="color: var(--accent-cyan); text-decoration: none; font-weight: 600; font-size: 0.85rem;">← Back to Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
    sidebar_footer()

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
st.markdown('<div class="glass-section">', unsafe_allow_html=True)
section_header("🔔", "Notifications")

# Notifications
st.markdown('<div class="glass-section">', unsafe_allow_html=True)
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

st.markdown('</div>', unsafe_allow_html=True)

# ── NEW: Apex Connect (Auto-Trading) ───────────────────────
st.markdown('<div class="glass-section">', unsafe_allow_html=True)
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
        risk_value = st.number_input("Lot Size", min_value=0.01, max_value=50.0, value=float(default_val), step=0.01, key="mt5_risk_val")
        st.caption("Trades will use this exact lot size.")
    else:
        risk_value = st.number_input("Risk % per Trade", min_value=0.1, max_value=5.0, value=float(default_val), step=0.1, key="mt5_risk_val")
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
        <small><strong>⏸ Bridge Paused:</strong> Signals will be generated but NOT executed automatically.</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Technical Config
st.markdown('<div class="glass-section">', unsafe_allow_html=True)
section_header("🛠", "Technical Configuration")

col1, col2 = st.columns(2)
with col1:
    st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY"], key="def_currency")
    st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"], key="tz")
with col2:
    st.selectbox("Theme", ["Dark Mode", "Light Mode", "System Default"], key="theme_sel")
    st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], key="log_level")

st.markdown('</div>', unsafe_allow_html=True)

# Data Management
st.markdown('<div class="glass-section">', unsafe_allow_html=True)
section_header("💾", "Data Management")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
        Cache Size: <span style="color: var(--accent-cyan); font-weight: 600;">142 MB</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
with c2:
    st.markdown("""
    <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
        Export: <span style="color: var(--accent-cyan); font-weight: 600;">CSV</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("📥 Download Logs", "timestamp,event\n2026-02-04,login", "logs.csv", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Save
if st.button("💾 Save System Settings", type="primary", use_container_width=True):
    # Notifications
    if 'notifications' not in config: config['notifications'] = {}
    config['notifications']['telegram'] = {
        'enabled': enable_tg,
        'bot_token': bot_token,
        'chat_id': chat_id
    }
    
    # Apex Connect (MT5)
    config['mt5'] = {
        'enabled': enable_mt5,
        'risk_type': 'percent' if risk_type_sel == 'Account %' else 'fixed',
        'risk_value': float(risk_value),
        'max_open_trades': mt5_config.get('max_open_trades', 5) # Preserve or default
    }
    
    save_config(config)
    st.toast("System settings saved!", icon="💾")
