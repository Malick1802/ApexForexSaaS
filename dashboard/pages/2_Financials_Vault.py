"""Financials & Vault page — standalone sub-page with shared theme."""
import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import inject_css, hero_banner, sidebar_logo, sidebar_footer, section_header, kpi_card

st.set_page_config(page_title="Financials · ApexForex", page_icon="💳", layout="wide")
inject_css()

with st.sidebar:
    sidebar_logo()
    st.markdown("""
    <div style="text-align:center; margin-bottom: 16px;">
        <a href="/" target="_self" style="color: var(--accent-cyan); text-decoration: none; font-weight: 600; font-size: 0.85rem;">← Back to Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
    sidebar_footer()

hero_banner("Financials & Vault", "Secure API key storage, subscription management, and billing")

tab1, tab2 = st.tabs(["🔑 API Vault", "💳 Billing & Subscription"])

with tab1:
    section_header("🔐", "Secure API Key Storage")
    st.info("Keys are stored locally in your `.env` file or secure environment variables.")

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("#### 📡 Market Data Providers")
    st.text_input("TwelveData API Key", type="password", key="td_key")
    st.text_input("AlphaVantage API Key", type="password", key="av_key")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI & Inference")
    st.text_input("OpenAI API Key", type="password", key="oai_key")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Brokerage Connections")
    st.text_input("MetaTrader 5 Login", key="mt_login")
    st.text_input("MetaTrader 5 Password", type="password", key="mt_pass")
    st.text_input("MetaTrader 5 Server", key="mt_server")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Save All Keys", use_container_width=True):
        st.success("✅ API Keys securely saved!")

with tab2:
    section_header("📋", "Subscription Status")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Current Plan", "Pro Trader", "✅ Active", "accent-gold"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Billing Cycle", "Monthly", "Auto-renewal", "accent-cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Next Payment", "Mar 04", "$49.00 USD", "accent-cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Payment Method</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:14px 20px;background:rgba(0,229,255,0.04);border:1px solid rgba(0,229,255,0.1);border-radius:12px;font-family:var(--font-mono);font-size:0.85rem;color:var(--text-primary);">
        💳 •••• •••• •••• 4242 <span style="color:var(--text-muted);font-family:var(--font-ui);font-size:0.75rem;">Visa</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Manage Subscription in Stripe →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Invoice History</div>', unsafe_allow_html=True)
    invoices = pd.DataFrame({
        "Date": ["Feb 01, 2026", "Jan 01, 2026"],
        "Amount": ["$49.00", "$49.00"],
        "Status": ["✅ Paid", "✅ Paid"],
        "Invoice": ["#INV-002", "#INV-001"]
    })
    st.dataframe(invoices, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
