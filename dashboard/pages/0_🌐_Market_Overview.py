"""Market Overview — Redirects to main app. Access via sidebar navigation."""
import streamlit as st
st.set_page_config(page_title="ApexForex", page_icon="⚡", layout="wide")
st.markdown("""
<style>[data-testid="stSidebarNav"] { display: none !important; }</style>
<meta http-equiv="refresh" content="0;url=/">
""", unsafe_allow_html=True)
st.info("Redirecting to main dashboard...")
