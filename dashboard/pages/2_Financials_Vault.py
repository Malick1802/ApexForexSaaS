"""Financials & Vault page — Admin-only master API key and brokerage credential vault."""
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
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import hero_banner, section_header, kpi_card, inject_css
inject_css()

# ── Env Loader & Writer ─────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

def load_env_values():
    values = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    values[k.strip()] = v
    return values

def save_env_values(new_values):
    lines = []
    updated_keys = set()
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, v = stripped.split("=", 1)
            k = k.strip()
            if k in new_values:
                val = new_values[k]
                new_lines.append(f'{k}="{val}"\n')
                updated_keys.add(k)
                continue
        new_lines.append(line)
        
    for k, v in new_values.items():
        if k not in updated_keys:
            new_lines.append(f'{k}="{v}"\n')
            
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

# Load existing values
env_data = load_env_values()

hero_banner("Master API Vault", "Secure master API key storage and brokerage infrastructure (Admin Only)")

t1, t2 = st.tabs(["🔑 Master Credentials", "🛡️ Security Guidelines"])

with t1:
    section_header("🔐", "Production Infrastructure Secrets")
    st.info("These credentials power the core background signal generation engine and master execution terminal. Never share these with subscribers.")

    with st.container(border=True):
        st.markdown("#### 📡 Market Data Providers")
        twelve_key = st.text_input("TwelveData API Key", value=env_data.get("TWELVEDATA_API_KEY", ""), type="password", key="av_key")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🤖 AI & Inference")
        openai_key = st.text_input("OpenAI API Key", value=env_data.get("OPENAI_API_KEY", ""), type="password", key="oai_key")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🏛️ Master Brokerage Terminal (FTMO / Main Executor)")
        mt_login = st.text_input("MetaTrader 5 Master Login", value=env_data.get("MT5_LOGIN", ""), key="mt_login")
        mt_pass = st.text_input("MetaTrader 5 Master Password", value=env_data.get("MT5_PASSWORD", ""), type="password", key="mt_pass")
        mt_server = st.text_input("MetaTrader 5 Master Server", value=env_data.get("MT5_SERVER", ""), key="mt_server")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("💾 Save All Infrastructure Keys", type="primary", use_container_width=True):
        updated_env = {
            "TWELVEDATA_API_KEY": twelve_key,
            "OPENAI_API_KEY": openai_key,
            "MT5_LOGIN": mt_login,
            "MT5_PASSWORD": mt_pass,
            "MT5_SERVER": mt_server
        }
        save_env_values(updated_env)
        st.toast("Infrastructure keys securely saved!", icon="✅")
        st.success("✅ Infrastructure keys securely saved to `.env` file.")

with t2:
    section_header("🛡️", "Operational Security Standards")
    st.markdown("""
    - **Subscriber Data Isolation**: Secondary subscriber terminals use individual credentials encrypted in `user_accounts.db` and run via isolated subprocess bridges.
    - **Master Credential Safety**: Only the master strategy execution loop references the `.env` credentials configured here.
    - **Role Verification**: Non-admin portal users are blocked at the server and routing layers from accessing this vault.
    """)
