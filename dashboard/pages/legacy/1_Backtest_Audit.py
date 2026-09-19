"""Backtest Audit â€” Redirects to main app Model Audit view."""
import streamlit as st
# ── Auth Guard ─────────────────────────────────────────────────────────────
# This page is served through app.py's st.navigation(). If accessed directly
# via URL (e.g. after browser refresh), session state may be empty — redirect
# the user back to the main app to re-authenticate.
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Session expired or not logged in. Please log in again.")
    st.page_link("app.py", label="🔑 Go to Login", icon="🔑")
    st.stop()
# ── End Auth Guard ─────────────────────────────────────────────────────────
st.set_page_config(page_title="ForexAlert", page_icon="âš¡", layout="wide")
st.markdown("""
<style>[data-testid="stSidebarNav"] { display: none !important; }</style>
<meta http-equiv="refresh" content="0;url=/">
""", unsafe_allow_html=True)
st.info("Redirecting to main dashboard...")
