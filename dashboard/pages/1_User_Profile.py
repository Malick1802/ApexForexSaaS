"""User Profile page — standalone sub-page with shared theme."""
import streamlit as st
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import inject_css, hero_banner, sidebar_logo, sidebar_footer

st.set_page_config(page_title="Profile · ApexForex", page_icon="👤", layout="wide")
inject_css()

with st.sidebar:
    sidebar_logo()
    st.markdown("""
    <div style="text-align:center; margin-bottom: 16px;">
        <a href="/" target="_self" style="color: var(--accent-cyan); text-decoration: none; font-weight: 600; font-size: 0.85rem;">← Back to Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
    sidebar_footer()

hero_banner("User Profile", "Manage your account details and organization settings")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 32px;">
        <img style="width:120px;height:120px;border-radius:50%;border:3px solid var(--accent-cyan);box-shadow:0 0 20px rgba(0,229,255,0.2);"
             src="https://ui-avatars.com/api/?name=Artem+K&background=0a1628&color=00E5FF&size=200&bold=true&font-size=0.4" />
        <div style="margin-top: 16px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: var(--text-primary);">Artem K.</div>
            <div style="display:inline-flex;align-items:center;gap:6px;padding:4px 14px;background:rgba(255,215,0,0.1);border:1px solid rgba(255,215,0,0.25);border-radius:20px;font-size:0.7rem;font-weight:700;color:var(--accent-gold);text-transform:uppercase;letter-spacing:0.08em;margin-top:8px;">
                👑 Premium Member
            </div>
            <div style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;font-weight:600;color:var(--success);margin-top:6px;">
                <div class="status-dot"></div> Online
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    fc1.text_input("First Name", "Artem")
    fc2.text_input("Last Name", "K.")
    st.text_input("Email", "artem@example.com")
    st.text_area("Bio", "Forex trader and developer building AI-powered SaaS solutions.", height=80)
    if st.button("💾 Update Profile", use_container_width=True):
        st.toast("Profile updated successfully!", icon="✅")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Organization</div>', unsafe_allow_html=True)
    st.text_input("Company Name", "Apex Forex Ltd.")
    st.text_input("Role", "Admin")
    st.markdown('</div>', unsafe_allow_html=True)
