import streamlit as st
import pandas as pd
from theme import inject_css, hero_banner, sidebar_logo, sidebar_footer, section_header, kpi_card

st.set_page_config(page_title="Model Audit · ApexForex", page_icon="🛡️", layout="wide")
inject_css()

hero_banner("Model Audit", "Historical performance and certification audit")

st.info("Model Audit functionality is integrated into the Intelligence suite. Use the main navigation to access transparency reports.")
