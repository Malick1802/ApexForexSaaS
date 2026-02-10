"""
ApexForex Premium Design System
Shared CSS, components, and utilities for all dashboard pages.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# ── Project Root Setup ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "signals.db")


def init_page(title: str, icon: str = "⚡"):
    """Shared page initializer — call FIRST in every page."""
    st.set_page_config(
        page_title=f"{title} · ApexForex",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()


def get_db():
    """Get properly-pathed SignalDatabase instance."""
    from core.database import SignalDatabase
    return SignalDatabase(db_path=DB_PATH)


def get_engine():
    """Get DataEngine instance (cached)."""
    from data_pipeline import DataEngine
    return DataEngine()


def get_inference():
    """Get InferenceEngine instance (cached)."""
    from core.inference import InferenceEngine
    return InferenceEngine()


# =============================================================================
# PREMIUM CSS DESIGN SYSTEM
# =============================================================================
def inject_css():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── CSS Variables ─────────────────────────────── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #0f1629;
        --bg-card: rgba(15, 22, 41, 0.6);
        --bg-card-hover: rgba(20, 30, 55, 0.8);
        --bg-glass: rgba(15, 22, 41, 0.45);
        --border-glass: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(0, 229, 255, 0.15);
        --text-primary: #e8ecf4;
        --text-secondary: #8b95a8;
        --text-muted: #4a5568;
        --accent-cyan: #00E5FF;
        --accent-cyan-dim: rgba(0, 229, 255, 0.12);
        --accent-gold: #FFD700;
        --accent-gold-dim: rgba(255, 215, 0, 0.12);
        --signal-buy: #00FF88;
        --signal-buy-bg: rgba(0, 255, 136, 0.08);
        --signal-buy-border: rgba(0, 255, 136, 0.25);
        --signal-sell: #FF4466;
        --signal-sell-bg: rgba(255, 68, 102, 0.08);
        --signal-sell-border: rgba(255, 68, 102, 0.25);
        --signal-wait: #FFA726;
        --signal-wait-bg: rgba(255, 167, 38, 0.08);
        --signal-wait-border: rgba(255, 167, 38, 0.25);
        --success: #00E676;
        --fail: #FF5252;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.4);
        --shadow-glow-cyan: 0 0 20px rgba(0, 229, 255, 0.15);
        --shadow-glow-gold: 0 0 20px rgba(255, 215, 0, 0.15);
        --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ── Global ────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d1224 50%, var(--bg-secondary) 100%);
        font-family: var(--font-ui);
        color: var(--text-primary);
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Typography ────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { font-family: var(--font-ui) !important; letter-spacing: -0.02em; }
    h1 { font-weight: 800 !important; color: var(--text-primary) !important; font-size: 2rem !important; }
    h2 { font-weight: 700 !important; color: var(--text-primary) !important; font-size: 1.5rem !important; }
    h3 { font-weight: 600 !important; color: var(--text-secondary) !important; font-size: 1.15rem !important; }
    p, span, label, .stMarkdown { font-family: var(--font-ui) !important; }

    /* ── Sidebar ───────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1120 0%, #0a0e1a 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

    /* ── Glassmorphic Cards ─────────────────────────── */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 24px;
        box-shadow: var(--shadow-card);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-glow);
        box-shadow: var(--shadow-card), var(--shadow-glow-cyan);
        transform: translateY(-2px);
    }

    /* ── KPI Metric Cards ──────────────────────────── */
    .kpi-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 20px 24px;
        box-shadow: var(--shadow-card);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .kpi-card:hover::before { opacity: 1; }
    .kpi-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: var(--shadow-card), var(--shadow-glow-cyan);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: var(--font-mono);
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 6px;
    }
    .kpi-card.accent-cyan .kpi-value { color: var(--accent-cyan); }
    .kpi-card.accent-gold .kpi-value { color: var(--accent-gold); }
    .kpi-card.accent-green .kpi-value { color: var(--signal-buy); }
    .kpi-card.accent-red .kpi-value { color: var(--signal-sell); }

    /* ── Signal Badges ─────────────────────────────── */
    .signal-badge {
        padding: 24px;
        border-radius: var(--radius-lg);
        text-align: center;
        font-family: var(--font-ui);
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: 0.1em;
    }
    .signal-buy {
        background: var(--signal-buy-bg);
        color: var(--signal-buy);
        border: 1px solid var(--signal-buy-border);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.1);
        animation: pulse-buy 2s ease-in-out infinite;
    }
    .signal-sell {
        background: var(--signal-sell-bg);
        color: var(--signal-sell);
        border: 1px solid var(--signal-sell-border);
        box-shadow: 0 0 30px rgba(255, 68, 102, 0.1);
        animation: pulse-sell 2s ease-in-out infinite;
    }
    .signal-wait {
        background: var(--signal-wait-bg);
        color: var(--signal-wait);
        border: 1px solid var(--signal-wait-border);
    }

    @keyframes pulse-buy {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.1); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.25); }
    }
    @keyframes pulse-sell {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 102, 0.1); }
        50% { box-shadow: 0 0 40px rgba(255, 68, 102, 0.25); }
    }

    /* ── Trading Levels ────────────────────────────── */
    .level-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border-radius: var(--radius-sm);
        margin-bottom: 6px;
        font-family: var(--font-mono);
        font-size: 0.9rem;
        font-weight: 500;
    }
    .level-tp { background: rgba(0, 255, 136, 0.06); border-left: 3px solid var(--signal-buy); color: var(--signal-buy); }
    .level-entry { background: rgba(0, 229, 255, 0.06); border-left: 3px solid var(--accent-cyan); color: var(--accent-cyan); }
    .level-sl { background: rgba(255, 68, 102, 0.06); border-left: 3px solid var(--signal-sell); color: var(--signal-sell); }
    .level-label { font-family: var(--font-ui); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); min-width: 40px; }
    .level-price { flex: 1; text-align: right; }
    .level-pips { font-size: 0.7rem; opacity: 0.7; }

    /* ── Hero Banner ───────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.08) 0%, rgba(255, 215, 0, 0.04) 50%, rgba(0, 229, 255, 0.02) 100%);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-xl);
        padding: 32px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(0, 229, 255, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    .hero-subtitle { font-size: 0.9rem; color: var(--text-secondary); font-weight: 400; }
    .hero-status {
        display: inline-flex; align-items: center; gap: 8px;
        margin-top: 12px; padding: 6px 14px;
        background: rgba(0, 230, 118, 0.08);
        border: 1px solid rgba(0, 230, 118, 0.2);
        border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; color: var(--success);
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--success);
        animation: blink 1.5s ease-in-out infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* ── Streamlit Overrides ───────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        padding: 16px 20px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-glass);
    }
    [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-weight: 600 !important; text-transform: uppercase !important; font-size: 0.7rem !important; letter-spacing: 0.06em !important; }

    [data-testid="stDataFrame"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-glass);
        border-radius: var(--radius-md);
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        font-weight: 600;
        font-size: 0.85rem;
    }

    hr { border-color: var(--border-glass) !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.4); }

    /* ── Signal Tiles (Market Overview) ─────────────── */
    .signal-tile {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        display: block;
        cursor: pointer;
        min-height: 140px;
        text-decoration: none;
    }
    .signal-tile:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5); }
    .signal-tile::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; transition: opacity 0.3s; }

    .tile-buy { border-color: rgba(0, 255, 136, 0.15); }
    .tile-buy::before { background: linear-gradient(90deg, var(--signal-buy), transparent); opacity: 1; }
    .tile-buy:hover { border-color: rgba(0, 255, 136, 0.3); box-shadow: 0 12px 40px rgba(0, 255, 136, 0.1); }

    .tile-sell { border-color: rgba(255, 68, 102, 0.15); }
    .tile-sell::before { background: linear-gradient(90deg, var(--signal-sell), transparent); opacity: 1; }
    .tile-sell:hover { border-color: rgba(255, 68, 102, 0.3); box-shadow: 0 12px 40px rgba(255, 68, 102, 0.1); }

    .tile-wait { border-color: var(--border-glass); opacity: 0.6; }
    .tile-wait:hover { opacity: 0.85; }

    .tile-symbol { font-size: 1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
    .tile-signal { font-size: 1.6rem; font-weight: 800; margin: 8px 0; letter-spacing: 0.08em; }
    .tile-signal-buy { color: var(--signal-buy); }
    .tile-signal-sell { color: var(--signal-sell); }
    .tile-signal-wait { color: var(--text-muted); }
    .tile-conf { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); }

    .conf-bar-bg { width: 80%; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; margin: 8px auto 0; overflow: hidden; }
    .conf-bar { height: 100%; border-radius: 2px; transition: width 0.5s ease; }
    .conf-bar-buy { background: linear-gradient(90deg, var(--signal-buy), rgba(0,255,136,0.3)); }
    .conf-bar-sell { background: linear-gradient(90deg, var(--signal-sell), rgba(255,68,102,0.3)); }

    /* ── Section Headers ───────────────────────────── */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 20px; padding-bottom: 12px;
        border-bottom: 1px solid var(--border-glass);
    }
    .section-header-icon { font-size: 1.3rem; }
    .section-header-text { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }

    /* ── Logo ──────────────────────────────────────── */
    .logo-container { text-align: center; padding: 16px 0 8px; }
    .logo-text {
        font-family: var(--font-ui);
        font-size: 1.6rem; font-weight: 900; letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, #00B8D4 40%, var(--accent-gold) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .logo-sub { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-top: 2px; }
    .version-badge { text-align: center; padding: 8px; font-size: 0.65rem; color: var(--text-muted); border-top: 1px solid var(--border-glass); margin-top: 16px; }
    .version-badge strong { color: var(--accent-cyan); }

    /* ── Misc Utility ──────────────────────────────── */
    .glass-section {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: all 0.3s;
    }
    .glass-section:hover { border-color: var(--border-glow); }
    .section-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 16px; }

    .rr-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px;
        background: var(--accent-cyan-dim);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 20px;
        font-family: var(--font-mono); font-size: 0.8rem; font-weight: 600; color: var(--accent-cyan);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SHARED COMPONENTS
# =============================================================================

def kpi_card(label, value, delta="", accent=""):
    """Generate glassmorphic KPI card HTML."""
    css = f"kpi-card {accent}" if accent else "kpi-card"
    return f"""
    <div class="{css}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """


def hero_banner(title, subtitle, show_status=False):
    """Generate hero banner HTML."""
    status_html = """<div class="hero-status"><div class="status-dot"></div> All Systems Operational</div>""" if show_status else ""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        {status_html}
    </div>
    """, unsafe_allow_html=True)


def sidebar_logo():
    """Render premium sidebar logo."""
    st.markdown("""
    <div class="logo-container">
        <div class="logo-text">ApexForex</div>
        <div class="logo-sub">AI Trading Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def sidebar_footer():
    """Render sidebar version badge."""
    st.markdown("""
    <div class="version-badge">
        <strong>Apex</strong> v2.0.0 · © 2026 ApexForex
    </div>
    """, unsafe_allow_html=True)


def section_header(icon, text):
    """Render section divider header."""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-header-icon">{icon}</span>
        <span class="section-header-text">{text}</span>
    </div>
    """, unsafe_allow_html=True)
