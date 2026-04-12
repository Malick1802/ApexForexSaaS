import streamlit as st
import pandas as pd
from theme import inject_css, hero_banner, sidebar_logo, sidebar_footer, section_header, kpi_card

st.set_page_config(page_title="Market Overview · ApexForex", page_icon="🌍", layout="wide")
inject_css()

hero_banner("Market Surveillance", "Live institutional flow monitoring across global markets")

st.info("Market Overview functionality is integrated into the Command Center. Use the main navigation to access live signals.")
