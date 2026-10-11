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

# Strict Database Pathing: Ensure frontend and backend never split.
DB_PATH = str(PROJECT_ROOT / "signals.db")

def init_page(title: str, icon: str = "⚡"):
    """Shared page initializer — call SECOND in every page after navigation."""
    # Removed st.set_page_config because it is handled by the main app/navigation
    inject_css()


def get_db():
    """Get properly-pathed SignalDatabase instance.

    The old version called importlib.reload() on every call, which is slow and
    ran on every landing-page visit. Set APEX_DEV_RELOAD=1 to bring it back
    while developing core/database.py.
    """
    import core.database
    if os.environ.get("APEX_DEV_RELOAD") == "1":
        import importlib
        importlib.reload(core.database)
    from core.database import SignalDatabase
    return SignalDatabase(db_path=DB_PATH)


def get_engine():
    """Get DataEngine instance (cached)."""
    try:
        from data_pipeline import DataEngine
        return DataEngine()
    except Exception as e:
        st.error(f"⚠️ DataEngine Initialization Error: {e}")
        from data_pipeline import DataEngine
        # Fallback to mt5 if initialization fails
        return DataEngine(provider_name="mt5")


@st.cache_resource
def get_inference():
    """Get InferenceEngine instance (cached)."""
    # Cache buster to force Streamlit to drop old instance in memory
    try:
        from core.inference import InferenceEngine
        engine = InferenceEngine()
        return engine
    except Exception as e:
        st.error(f"⚠️ InferenceEngine Initialization Error: {e}")
        raise e


# =============================================================================
# PREMIUM CSS DESIGN SYSTEM
# =============================================================================
def inject_css():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── CSS Variables ─────────────────────────────── */
    :root {
        --bg-primary: #050811;
        --bg-secondary: #0a0f1e;
        --bg-card: rgba(13, 19, 36, 0.72);
        --bg-card-hover: rgba(18, 26, 48, 0.88);
        --bg-glass: rgba(13, 19, 36, 0.52);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(0, 229, 255, 0.22);
        --text-primary: #f0f4fc;
        --text-secondary: #8e9bb0;
        --text-muted: #7d8aa3; /* was #526079 (~3:1); now ~5.7:1 on the page background */
        --accent-cyan: #00E5FF;
        --accent-cyan-dim: rgba(0, 229, 255, 0.12);
        --accent-cyan-hover: #33ebff;
        --accent-gold: #FFD700;
        --accent-gold-dim: rgba(255, 215, 0, 0.12);
        --signal-buy: #00FF88;
        --signal-buy-bg: rgba(0, 255, 136, 0.09);
        --signal-buy-border: rgba(0, 255, 136, 0.3);
        --signal-sell: #FF4466;
        --signal-sell-bg: rgba(255, 68, 102, 0.09);
        --signal-sell-border: rgba(255, 68, 102, 0.3);
        --signal-wait: #FFA726;
        --signal-wait-bg: rgba(255, 167, 38, 0.09);
        --signal-wait-border: rgba(255, 167, 38, 0.3);
        --success: #00E676;
        --fail: #FF5252;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 22px;
        --shadow-card: 0 4px 24px -1px rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(255, 255, 255, 0.06);
        --shadow-glow-cyan: 0 0 24px rgba(0, 229, 255, 0.22);
        --shadow-glow-gold: 0 0 24px rgba(255, 215, 0, 0.22);
        --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ── Suppress rerun blink (keep content clear during fragment updates) ── */
    [data-stale="true"],
    .stale-element,
    .element-container[data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* Hide the running/status indicator bar */
    div[data-testid="stStatusWidget"],
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    /* Prevent any overlay dimming during reruns */
    .stApp > div[data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
    }

    /* ── Global ────────────────────────────────────── */
    .stApp {
        background: 
            radial-gradient(circle at 18% 12%, rgba(0, 229, 255, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 82% 18%, rgba(255, 215, 0, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 50% 65%, rgba(0, 229, 255, 0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 50% -10%, #0d1e3d 0%, #060a16 55%, #03050a 100%) !important;
        background-attachment: fixed !important;
        font-family: var(--font-ui);
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    #MainMenu, footer { visibility: hidden; }
    
    /* Ensure sidebar toggle is always visible and styled for the theme */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-radius: 50% !important;
        padding: 6px !important;
        margin: 10px !important;
        border: 1px solid var(--border-glass) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: rgba(0, 229, 255, 0.12) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--shadow-glow-cyan) !important;
    }
    
    /* Hide the Streamlit native Deploy button */
    .stDeployButton,
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* ── Typography ────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { font-family: var(--font-ui) !important; letter-spacing: -0.03em; }
    h1 { font-weight: 800 !important; color: var(--text-primary) !important; font-size: 2.1rem !important; }
    h2 { font-weight: 700 !important; color: var(--text-primary) !important; font-size: 1.55rem !important; }
    h3 { font-weight: 600 !important; color: var(--text-primary) !important; font-size: 1.2rem !important; }
    p, label, .stMarkdown { font-family: var(--font-ui) !important; }

    /* ── Main Content Spacing & Header Optimization ── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
    }
    .block-container,
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
    }

    /* ── Sidebar ───────────────────────────────────── */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b17 0%, #040711 100%) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.14) !important;
        position: relative !important;
        padding-top: 76px !important; /* Compact spacing beneath top logo */
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stSidebarHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        position: absolute !important;
        top: 10px !important;
        right: 12px !important;
        z-index: 100000 !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

    /* ── Sidebar Navigation Link Buttons ──────────────── */
    [data-testid="stSidebarNavLink"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid transparent !important;
        border-radius: var(--radius-md) !important;
        color: #8fa0be !important;
        padding: 10px 16px !important;
        margin-bottom: 6px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebarNavLink"]:hover,
    [data-testid="stSidebarNavLink"]:focus {
        background: rgba(0, 229, 255, 0.06) !important;
        border-color: rgba(0, 229, 255, 0.25) !important;
        color: var(--accent-cyan) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 16px rgba(0, 229, 255, 0.12) !important;
    }
    [data-testid="stSidebarNavLink"][data-selected="true"],
    [data-testid="stSidebarNavLink"][aria-current="page"] {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.18) 0%, rgba(0, 229, 255, 0.03) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.45) !important;
        border-left: 3px solid var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2) !important;
    }

    /* ── Glassmorphic Cards ─────────────────────────── */
    .glass-card {
        background: 
            radial-gradient(circle at 95% 5%, rgba(0, 229, 255, 0.05) 0%, transparent 50%),
            linear-gradient(145deg, rgba(13, 20, 38, 0.78) 0%, rgba(7, 11, 23, 0.9) 100%) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(0, 229, 255, 0.14) !important;
        border-radius: var(--radius-lg);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .glass-card:hover {
        background: 
            radial-gradient(circle at 95% 5%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
            linear-gradient(145deg, rgba(16, 25, 48, 0.88) 0%, rgba(9, 14, 28, 0.95) 100%) !important;
        border-color: rgba(0, 229, 255, 0.35) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 22px rgba(0, 229, 255, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* ── KPI Metric Cards ──────────────────────────── */
    .kpi-card {
        background: 
            radial-gradient(circle at 90% 10%, rgba(0, 229, 255, 0.07) 0%, transparent 60%),
            linear-gradient(145deg, rgba(14, 22, 42, 0.8) 0%, rgba(8, 12, 25, 0.9) 100%) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-top: 2px solid var(--accent-cyan) !important;
        border-radius: var(--radius-lg);
        padding: 16px 20px;
        box-shadow: 0 8px 28px -2px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), transparent);
        opacity: 0.8;
        transition: opacity 0.3s;
    }
    .kpi-card:hover {
        border-color: rgba(0, 229, 255, 0.45) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6), 0 0 24px rgba(0, 229, 255, 0.18) !important;
    }
    .kpi-card.accent-gold { border-top: 2px solid var(--accent-gold) !important; }
    .kpi-card.accent-gold::before { background: linear-gradient(90deg, var(--accent-gold), transparent); }
    .kpi-card.accent-green { border-top: 2px solid var(--signal-buy) !important; }
    .kpi-card.accent-green::before { background: linear-gradient(90deg, var(--signal-buy), transparent); }
    .kpi-card.accent-red { border-top: 2px solid var(--signal-sell) !important; }
    .kpi-card.accent-red::before { background: linear-gradient(90deg, var(--signal-sell), transparent); }

    a.kpi-card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        cursor: pointer;
    }
    a.kpi-card-link .kpi-card {
        cursor: pointer;
    }
    a.kpi-card-link:hover .kpi-card {
        transform: translateY(-3px);
        border-color: rgba(0, 229, 255, 0.55) !important;
        box-shadow: 0 12px 32px rgba(0, 229, 255, 0.22), 0 0 14px rgba(0, 229, 255, 0.15) !important;
    }
    a.kpi-card-link:hover .kpi-card.accent-gold {
        border-color: rgba(255, 171, 0, 0.55) !important;
        box-shadow: 0 12px 32px rgba(255, 171, 0, 0.22), 0 0 14px rgba(255, 171, 0, 0.15) !important;
    }
    a.kpi-card-link:hover .kpi-card.accent-green {
        border-color: rgba(0, 230, 118, 0.55) !important;
        box-shadow: 0 12px 32px rgba(0, 230, 118, 0.22), 0 0 14px rgba(0, 230, 118, 0.15) !important;
    }
    a.kpi-card-link:hover .kpi-card.accent-red {
        border-color: rgba(255, 23, 68, 0.55) !important;
        box-shadow: 0 12px 32px rgba(255, 23, 68, 0.22), 0 0 14px rgba(255, 23, 68, 0.15) !important;
    }
    a.kpi-card-link:hover .kpi-card-arrow {
        opacity: 1 !important;
        transform: translate(2px, -2px);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: var(--font-mono);
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        letter-spacing: -0.02em;
        text-shadow: 0 0 16px rgba(0, 229, 255, 0.25);
    }
    .kpi-delta {
        font-size: 0.76rem;
        color: #9ab0d3;
        margin-top: 6px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-card.accent-cyan .kpi-value { color: var(--accent-cyan); text-shadow: 0 0 18px rgba(0, 229, 255, 0.35); }
    .kpi-card.accent-gold .kpi-value { color: var(--accent-gold); text-shadow: 0 0 18px rgba(255, 215, 0, 0.35); }
    .kpi-card.accent-green .kpi-value { color: var(--signal-buy); text-shadow: 0 0 18px rgba(0, 255, 136, 0.35); }
    .kpi-card.accent-red .kpi-value { color: var(--signal-sell); text-shadow: 0 0 18px rgba(255, 68, 102, 0.35); }

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

    /* ── Trading Levels Uniform Grid & Cards ────────────────── */
    .trading-levels-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        width: 100%;
        box-sizing: border-box;
    }
    .trading-level-card {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 90px;
        min-height: 90px;
        max-height: 90px;
        box-sizing: border-box;
        padding: 10px 12px;
        border-radius: 8px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .trading-level-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .trading-level-card.tl-entry {
        background: rgba(0, 229, 255, 0.04);
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-left: 4px solid var(--accent-cyan);
    }
    .trading-level-card.tl-tp {
        background: rgba(0, 255, 136, 0.04);
        border: 1px solid rgba(0, 255, 136, 0.22);
        border-left: 4px solid var(--signal-buy);
    }
    .trading-level-card.tl-sl {
        background: rgba(255, 68, 102, 0.04);
        border: 1px solid rgba(255, 68, 102, 0.22);
        border-left: 4px solid var(--signal-sell);
    }
    .trading-level-card.tl-rr {
        background: rgba(255, 171, 0, 0.04);
        border: 1px solid rgba(255, 171, 0, 0.22);
        border-left: 4px solid var(--accent-gold);
    }
    .tl-header {
        font-family: var(--font-ui);
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        line-height: 1.2;
        margin: 0;
    }
    .tl-value {
        font-family: var(--font-mono);
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 2px 0;
    }
    .tl-entry .tl-value { color: var(--accent-cyan); }
    .tl-tp .tl-value { color: var(--signal-buy); }
    .tl-sl .tl-value { color: var(--signal-sell); }
    .tl-rr .tl-value { color: var(--accent-gold); }
    .tl-footer {
        font-family: var(--font-ui);
        font-size: 0.68rem;
        font-weight: 600;
        line-height: 1.2;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .tl-entry .tl-footer { color: var(--text-secondary); }
    .tl-tp .tl-footer { color: var(--signal-buy); }
    .tl-sl .tl-footer { color: var(--signal-sell); }
    .tl-rr .tl-footer { color: var(--accent-gold); }

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
        background: 
            radial-gradient(ellipse at 85% 0%, rgba(0, 229, 255, 0.14) 0%, transparent 60%),
            radial-gradient(circle at 10% 100%, rgba(255, 215, 0, 0.05) 0%, transparent 45%),
            linear-gradient(135deg, rgba(13, 22, 45, 0.9) 0%, rgba(7, 11, 23, 0.96) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.22) !important;
        border-top: 2px solid var(--accent-cyan) !important;
        border-radius: var(--radius-lg) !important;
        padding: 20px 28px !important;
        margin-bottom: 20px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 10px 32px -4px rgba(0, 0, 0, 0.55), 0 0 20px rgba(0, 229, 255, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(0, 229, 255, 0.3) 30%, rgba(255, 215, 0, 0.2) 70%, transparent 100%);
    }
    .hero-banner-inner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        position: relative;
        z-index: 2;
    }
    .hero-title {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0F7FA 40%, var(--accent-cyan) 85%, var(--accent-gold) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.03em !important;
        line-height: 1.2 !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .hero-subtitle {
        font-size: 0.88rem !important;
        color: #9ab0d3 !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        max-width: 820px !important;
    }
    .hero-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        background: rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 20px;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(0, 255, 136, 0.08);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 700;
        color: var(--signal-buy);
        box-shadow: 0 0 16px rgba(0, 255, 136, 0.15);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--signal-buy);
        box-shadow: 0 0 8px var(--signal-buy);
        animation: blink 1.5s ease-in-out infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.85); } }

    /* ── MoonX Pro Terminal Tokens & Dock ──────────────── */
    .moonx-ribbon-dock {
        background: rgba(8, 12, 24, 0.82);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: var(--radius-md);
        padding: 6px 0;
        margin-top: 0;
        margin-bottom: 14px;
        width: 100%;
        overflow: hidden;
        position: relative;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
        mask-image: linear-gradient(to right, transparent 0%, black 2.5%, black 97.5%, transparent 100%);
        -webkit-mask-image: linear-gradient(to right, transparent 0%, black 2.5%, black 97.5%, transparent 100%);
        transition: border-color 0.2s ease;
    }
    .moonx-ribbon-dock:hover {
        border-color: rgba(0, 229, 255, 0.35);
    }
    .moonx-ticker-track {
        display: flex;
        align-items: center;
        width: max-content;
        animation: moonxMarqueeLeft 40s linear infinite;
        will-change: transform;
    }
    .moonx-ticker-track:hover,
    .moonx-ribbon-dock:hover .moonx-ticker-track {
        animation-play-state: paused;
    }
    .moonx-ticker-group {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-right: 10px;
        flex-shrink: 0;
    }
    @keyframes moonxMarqueeLeft {
        0% {
            transform: translateX(0);
        }
        100% {
            transform: translateX(-50%);
        }
    }
    .moonx-brand-block {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .moonx-brand-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #00E5FF 0%, #0077B6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 2px 10px rgba(0, 229, 255, 0.35);
    }
    .moonx-brand-title {
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, #FFFFFF 0%, #E0F7FA 70%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .moonx-brand-subtitle {
        font-size: 0.68rem;
        color: var(--text-muted);
        letter-spacing: 0.02em;
    }
    .moonx-ticker-container {
        display: flex;
        align-items: center;
        gap: 8px;
        overflow-x: auto;
        padding: 2px 0;
    }
    .moonx-ticker-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 5px 11px;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: var(--font-mono);
        font-size: 0.76rem;
        transition: all 0.2s ease;
        white-space: nowrap;
        user-select: none;
    }
    .moonx-ticker-item:hover {
        background: rgba(0, 229, 255, 0.08);
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-1px);
    }
    .moonx-ticker-sym {
        color: var(--text-secondary);
        font-weight: 700;
        font-size: 0.72rem;
    }
    .moonx-ticker-price {
        color: var(--text-primary);
        font-weight: 700;
    }
    .moonx-ticker-delta-up {
        color: var(--signal-buy);
        font-weight: 700;
    }
    .moonx-ticker-delta-down {
        color: var(--signal-sell);
        font-weight: 700;
    }
    .moonx-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: var(--font-mono);
    }
    .moonx-pill-cyan {
        background: rgba(0, 229, 255, 0.1);
        color: var(--accent-cyan);
        border: 1px solid rgba(0, 229, 255, 0.3);
    }
    .moonx-pill-gold {
        background: rgba(255, 215, 0, 0.1);
        color: var(--accent-gold);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    .moonx-pill-green {
        background: rgba(0, 255, 136, 0.1);
        color: var(--signal-buy);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    .moonx-pill-red {
        background: rgba(255, 68, 102, 0.1);
        color: var(--signal-sell);
        border: 1px solid rgba(255, 68, 102, 0.3);
    }
    .moonx-pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        animation: moonx-pulse 2s infinite ease-in-out;
    }
    .moonx-pulse-green {
        background: var(--signal-buy);
        box-shadow: 0 0 8px var(--signal-buy);
    }
    .moonx-pulse-gold {
        background: var(--accent-gold);
        box-shadow: 0 0 8px var(--accent-gold);
    }
    .moonx-pulse-cyan {
        background: var(--accent-cyan);
        box-shadow: 0 0 8px var(--accent-cyan);
    }
    @keyframes moonx-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
    }

    /* ── Streamlit Overrides ───────────────────────── */
    [data-testid="stMetric"] {
        background: 
            radial-gradient(circle at 90% 10%, rgba(0, 229, 255, 0.06) 0%, transparent 60%),
            linear-gradient(145deg, rgba(14, 22, 42, 0.8) 0%, rgba(8, 12, 25, 0.9) 100%) !important;
        backdrop-filter: blur(24px) !important;
        padding: 18px 22px !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-top: 2px solid var(--accent-cyan) !important;
        box-shadow: 0 8px 28px -2px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 229, 255, 0.45) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 229, 255, 0.15) !important;
    }
    [data-testid="stMetricValue"] { 
        font-family: var(--font-mono) !important; 
        font-weight: 800 !important; 
        font-size: 1.95rem !important; 
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 0 16px rgba(0, 229, 255, 0.25) !important;
    }
    [data-testid="stMetricLabel"] { 
        font-weight: 700 !important; 
        text-transform: uppercase !important; 
        font-size: 0.72rem !important; 
        letter-spacing: 0.1em !important; 
        color: var(--text-muted) !important;
    }

    [data-testid="stDataFrame"] {
        background: rgba(8, 12, 24, 0.82) !important;
        border: 1px solid rgba(0, 229, 255, 0.18) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(16px) !important;
    }

    /* Modern Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(8, 13, 27, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0, 229, 255, 0.16) !important;
        border-radius: 30px !important;
        padding: 5px !important;
        gap: 6px !important;
        width: fit-content !important;
        margin-bottom: 22px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px !important;
        font-family: var(--font-ui) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 22px !important;
        color: var(--text-secondary) !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.2) 0%, rgba(0, 229, 255, 0.05) 100%) !important;
        border-color: rgba(0, 229, 255, 0.45) !important;
        color: var(--accent-cyan) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.22) !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: linear-gradient(145deg, rgba(13, 20, 38, 0.72) 0%, rgba(7, 11, 22, 0.85) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.15) !important;
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 16px !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 229, 255, 0.32) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), 0 0 16px rgba(0, 229, 255, 0.1) !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: var(--font-ui) !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        padding: 14px 22px !important;
        font-size: 0.96rem !important;
        letter-spacing: -0.01em !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--accent-cyan) !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        background: rgba(13, 19, 36, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        padding: 14px 20px !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stAlert"] [data-testid="stNotificationContentInfo"] {
        border-left: 4px solid var(--accent-cyan) !important;
    }
    div[data-testid="stAlert"] [data-testid="stNotificationContentSuccess"] {
        border-left: 4px solid var(--signal-buy) !important;
    }
    div[data-testid="stAlert"] [data-testid="stNotificationContentWarning"] {
        border-left: 4px solid var(--signal-wait) !important;
    }
    div[data-testid="stAlert"] [data-testid="stNotificationContentError"] {
        border-left: 4px solid var(--signal-sell) !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--accent-cyan) !important;
        box-shadow: 0 0 10px var(--accent-cyan) !important;
    }
    div[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div:first-child {
        background: linear-gradient(90deg, var(--accent-cyan) 0%, #00B8D4 100%) !important;
    }

    /* Progress Bar */
    div[data-testid="stProgress"] > div {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        height: 9px !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--accent-cyan) 0%, #00FF88 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.4) !important;
    }

    /* Form Controls */
    div[data-testid="stCheckbox"] input:checked + div,
    div[data-testid="stToggle"] input:checked + div {
        background-color: var(--accent-cyan) !important;
    }

    hr { border-color: var(--border-glass) !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.25); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.5); }

    .signal-tile {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 18px 14px 14px 14px;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        display: block;
        min-height: 160px;
        text-decoration: none;
        margin-bottom: 6px;
    }
    .signal-tile:hover {
        border-color: var(--border-glow);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 229, 255, 0.1);
    }
    a.signal-tile-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        cursor: pointer;
        margin-bottom: 12px;
    }
    a.signal-tile-link .signal-tile {
        cursor: pointer;
        transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.22s ease, box-shadow 0.22s ease, opacity 0.22s ease;
    }
    a.signal-tile-link:hover .signal-tile {
        transform: translateY(-4px);
        border-color: rgba(0, 229, 255, 0.55) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 229, 255, 0.2) !important;
        opacity: 1 !important;
    }
    a.signal-tile-link:hover .signal-tile.tile-buy {
        border-color: rgba(0, 255, 136, 0.55) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 255, 136, 0.2) !important;
    }
    a.signal-tile-link:hover .signal-tile.tile-sell {
        border-color: rgba(255, 68, 102, 0.55) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6), 0 0 20px rgba(255, 68, 102, 0.2) !important;
    }
    a.signal-tile-link:hover .tile-arrow {
        opacity: 1 !important;
        transform: translate(2px, -2px);
    }
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

    /* ── Ghost Indicator (Secondary Tier) ──────────────── */
    .ghost-indicator {
        position: absolute;
        bottom: 10px;
        right: 10px;
        width: 8px;
        height: 8px;
        background: var(--accent-cyan);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-cyan), 0 0 20px var(--accent-cyan);
        animation: ghost-pulse 2s infinite;
        z-index: 5;
    }
    @keyframes ghost-pulse {
        0% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(0, 229, 255, 0.4); }
        70% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 0 10px rgba(0, 229, 255, 0); }
        100% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(0, 229, 255, 0); }
    }

    /* ── Tier Stack (Terminal) ─────────────────────────── */
    .tier-stack-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        border-left: 3px solid transparent;
        transition: all 0.2s;
    }
    .tier-stack-item:hover { background: rgba(255,255,255,0.06); }
    .tier-stack-live { border-left-color: var(--signal-buy); background: rgba(0,255,136,0.05); }
    .tier-stack-shadow { border-left-color: var(--text-muted); opacity: 0.8; }
    
    .tier-stack-badge {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        text-transform: uppercase;
    }

    /* ── Section Headers ───────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        position: relative;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 140px;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan) 0%, rgba(0, 229, 255, 0.3) 70%, transparent 100%);
        box-shadow: 0 0 8px var(--accent-cyan);
    }
    .section-header-icon-wrap {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.14) 0%, rgba(0, 229, 255, 0.02) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        box-shadow: 0 0 14px rgba(0, 229, 255, 0.15);
        flex-shrink: 0;
    }
    .section-header-text {
        font-family: var(--font-ui);
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    /* ── Logo ──────────────────────────────────────── */
    .logo-container { text-align: center; padding: 0; margin: 0; }
    .logo-text {
        font-family: var(--font-ui);
        font-size: 1.45rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.15;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, #00B8D4 40%, var(--accent-gold) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .logo-sub { font-size: 0.62rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-muted); margin-top: 2px; }
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
    /* ── Sentinel Health Pulse ─────────────────────── */
    .health-badge {
        display: flex; align-items: center; gap: 8px;
        padding: 5px 12px; border-radius: 12px;
        font-size: 0.65rem; font-weight: 700; font-family: var(--font-mono);
        letter-spacing: 0.05em; text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.02);
        margin: 10px auto; width: fit-content;
    }
    .health-dot {
        width: 6px; height: 6px; border-radius: 50%;
        box-shadow: 0 0 8px currentColor;
    }
    .health-online { color: #00FF88; border-color: rgba(0,255,136,0.2); background: rgba(0,255,136,0.05); }
    .health-offline { color: #FF4466; border-color: rgba(255,68,102,0.2); background: rgba(255,68,102,0.05); }
    
    @keyframes heartbeat {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
    }
    .pulse-animation { animation: heartbeat 2s ease-in-out infinite; }

    /* ── Streamlit Container Glass Overrides ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: 
            radial-gradient(circle at 95% 5%, rgba(0, 229, 255, 0.05) 0%, transparent 50%),
            linear-gradient(145deg, rgba(13, 20, 38, 0.76) 0%, rgba(7, 11, 23, 0.88) 100%) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(0, 229, 255, 0.14) !important;
        border-radius: var(--radius-lg) !important;
        padding: 22px 26px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        transition: all 0.25s ease !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 229, 255, 0.32) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 229, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Move Logo to Top of Sidebar ──────────────── */
    [data-testid="stSidebar"] div:has(.logo-container) {
        position: static !important;
    }
    .logo-container {
        position: absolute !important;
        top: 14px !important;
        left: 0 !important;
        right: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        z-index: 99999 !important;
        text-align: center !important;
    }

    /* ── Button Styling Overrides ─────────────────── */
    div.stButton > button, 
    div.stDownloadButton > button, 
    [data-testid^="stBaseButton"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-ui) !important;
        font-weight: 600 !important;
        padding: 9px 20px !important;
        font-size: 0.88rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    }
    div.stButton > button:hover, 
    div.stDownloadButton > button:hover, 
    [data-testid^="stBaseButton"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
        box-shadow: 0 4px 16px rgba(0, 229, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active, 
    [data-testid^="stBaseButton"]:active {
        transform: scale(0.98) !important;
    }
    
    /* Primary buttons */
    div.stButton > button[data-testid="stBaseButton-primary"], 
    div.stButton > button[kind="primary"],
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%) !important;
        color: #030814 !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(0, 229, 255, 0.35) !important;
    }
    div.stButton > button[data-testid="stBaseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #33EBFF 0%, #00C4E4 100%) !important;
        color: #030814 !important;
        box-shadow: 0 6px 24px rgba(0, 229, 255, 0.55) !important;
        transform: translateY(-2px) !important;
    }

    /* Secondary buttons */
    div.stButton > button[data-testid="stBaseButton-secondary"],
    div.stButton > button[kind="secondary"],
    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-glass) !important;
        color: var(--text-primary) !important;
    }
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover,
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(0, 229, 255, 0.06) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }

    /* Form Inputs, Textareas, Steppers */
    div[data-baseweb="input"] {
        background-color: rgba(9, 14, 27, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.25) !important;
    }
    div[data-baseweb="textarea"] {
        background-color: rgba(9, 14, 27, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.25) !important;
    }

    /* Selectboxes & Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: rgba(9, 14, 27, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.2) !important;
    }
    /* Popover menu options */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #0b1122 !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.85) !important;
    }
    li[role="option"] {
        color: #e8ecf4 !important;
        font-family: var(--font-ui) !important;
        font-size: 0.88rem !important;
        transition: background 0.15s ease !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: rgba(0, 229, 255, 0.15) !important;
        color: #00E5FF !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tag"] {
        background: rgba(0, 229, 255, 0.12) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 6px !important;
        color: #00E5FF !important;
    }

    /* ── Accessibility & responsive layer ───────────── */
    a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible,
    [role="tab"]:focus-visible, [role="radio"]:focus-visible, summary:focus-visible,
    [data-testid="stSidebarNavLink"]:focus-visible {
        outline: 2px solid var(--accent-cyan) !important;
        outline-offset: 2px !important;
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }
    }
    @media (max-width: 760px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 { font-size: 1.7rem !important; }
        h2 { font-size: 1.3rem !important; }
        .kpi-card { min-height: 0 !important; padding: 12px 14px !important; }
        .glass-card { padding: 16px !important; }
        /* wide tables/charts scroll inside themselves instead of the page */
        [data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto; }
    }

    /* ── Market Sessions Cockpit ───────────────────── */
    .session-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }
    .session-card {
        background: radial-gradient(circle at 90% 10%, rgba(0, 229, 255, 0.05) 0%, transparent 60%),
                    linear-gradient(145deg, rgba(14, 22, 42, 0.75) 0%, rgba(8, 12, 25, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-md);
        padding: 12px 14px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .session-card.session-open {
        border-top: 2px solid var(--signal-buy);
        box-shadow: 0 4px 18px rgba(0, 255, 136, 0.1);
    }
    .session-card.session-closed {
        border-top: 2px solid rgba(255, 255, 255, 0.1);
        opacity: 0.75;
    }
    .session-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }
    .session-city {
        font-weight: 700;
        font-size: 0.88rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 3px;
    }
    .session-hours {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    .session-badge-open {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 7px;
        border-radius: 10px;
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 700;
        color: var(--signal-buy);
        background: rgba(0, 255, 136, 0.12);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    .session-badge-closed {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 7px;
        border-radius: 10px;
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 600;
        color: var(--text-muted);
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Opportunity Cards ─────────────────────────── */
    .opp-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }
    .opp-card {
        background: radial-gradient(circle at 90% 10%, rgba(0, 229, 255, 0.08) 0%, transparent 60%),
                    linear-gradient(145deg, rgba(14, 22, 44, 0.82) 0%, rgba(8, 12, 26, 0.92) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: var(--radius-lg);
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    }
    .opp-card.opp-buy {
        border-top: 3px solid var(--signal-buy);
    }
    .opp-card.opp-buy:hover {
        border-color: rgba(0, 255, 136, 0.45);
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.55), 0 0 20px rgba(0, 255, 136, 0.18);
        transform: translateY(-2px);
    }
    .opp-card.opp-sell {
        border-top: 3px solid var(--signal-sell);
    }
    .opp-card.opp-sell:hover {
        border-color: rgba(255, 68, 102, 0.45);
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.55), 0 0 20px rgba(255, 68, 102, 0.18);
        transform: translateY(-2px);
    }
    .opp-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .opp-symbol {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .opp-dir-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-family: var(--font-mono);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }
    .opp-dir-buy {
        background: rgba(0, 255, 136, 0.12);
        color: var(--signal-buy);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    .opp-dir-sell {
        background: rgba(255, 68, 102, 0.12);
        color: var(--signal-sell);
        border: 1px solid rgba(255, 68, 102, 0.3);
    }
    .opp-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        background: rgba(0, 0, 0, 0.28);
        border-radius: 8px;
        padding: 10px 12px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .opp-metric-item {
        text-align: center;
    }
    .opp-metric-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .opp-metric-val {
        font-family: var(--font-mono);
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 2px;
        color: #ffffff;
    }

    /* ── Market Breadth / Vitality Pill ────────────────── */
    .vitality-container {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }
    .vitality-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        background: rgba(14, 22, 42, 0.75);
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: var(--radius-md);
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-primary);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    .vitality-pill b {
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SHARED COMPONENTS
# =============================================================================

def kpi_card(label, value, delta="", accent="", link_url=""):
    """Generate glassmorphic KPI card HTML."""
    css = f"kpi-card {accent}" if accent else "kpi-card"
    
    arrow_html = (
        '<span class="kpi-card-arrow" style="font-size: 0.85rem; opacity: 0.4; transition: all 0.2s ease; margin-left: auto;">&#x2197;</span>'
        if link_url else ""
    )
    
    card_html = (
        f'<div class="{css}">'
        f'<div class="kpi-label" style="display: flex; align-items: center;"><span>{label}</span>{arrow_html}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div>'
        f'</div>'
    )
    
    if link_url:
        return f'<a href="{link_url}" target="_self" class="kpi-card-link" style="text-decoration: none; color: inherit; display: block;">{card_html}</a>'
    
    return card_html


def hero_banner(title, subtitle, show_status=False, badge="INSTITUTIONAL AI"):
    """Generate radiant glassmorphic cyber hero banner HTML."""
    status_html = (
        '<div class="hero-status"><div class="status-dot"></div> ALL SYSTEMS OPERATIONAL</div>'
        if show_status else ""
    )
    badge_html = f'<div class="hero-pill-badge">⚡ {badge}</div>' if badge else ""
    st.markdown(
        f'<div class="hero-banner">'
        f'<div class="hero-banner-inner">'
        f'<div>'
        f'{badge_html}'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-subtitle">{subtitle}</div>'
        f'</div>'
        f'{status_html}'
        f'</div>'
        f'</div>', 
        unsafe_allow_html=True
    )


def render_moonx_ticker_ribbon():
    """
    Renders the MoonX-style top live ticker ribbon dock in Streamlit.
    Displays popular asset quotes moving seamlessly to the left in an infinite loop.
    The card contains ONLY the popular pairs quotes, nothing else.
    """
    popular_assets = [
        {"display": "XAU/USD", "symbols": ["XAUUSD", "GOLD"], "tag": "GOLD", "pill": "moonx-pill-gold", "price": "4,195.09", "delta": "+1.45%", "up": True, "decimals": 2},
        {"display": "EUR/USD", "symbols": ["EURUSD"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "1.12018", "delta": "+0.18%", "up": True, "decimals": 5},
        {"display": "BTC/USD", "symbols": ["BTCUSD"], "tag": "CRYPTO", "pill": "moonx-pill-green", "price": "83,022.90", "delta": "+3.20%", "up": True, "decimals": 2},
        {"display": "GBP/USD", "symbols": ["GBPUSD"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "1.32352", "delta": "+0.34%", "up": True, "decimals": 5},
        {"display": "US30", "symbols": ["US30.cash", "US30", "DJ30"], "tag": "INDEX", "pill": "moonx-pill-cyan", "price": "51,688.50", "delta": "+0.65%", "up": True, "decimals": 2},
        {"display": "USD/JPY", "symbols": ["USDJPY"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "158.288", "delta": "-0.22%", "up": False, "decimals": 3},
        {"display": "ETH/USD", "symbols": ["ETHUSD"], "tag": "CRYPTO", "pill": "moonx-pill-green", "price": "2,502.26", "delta": "+2.85%", "up": True, "decimals": 2},
        {"display": "USOIL", "symbols": ["USOIL.cash", "USOIL", "WTI"], "tag": "OIL", "pill": "moonx-pill-gold", "price": "92.58", "delta": "-0.45%", "up": False, "decimals": 2},
        {"display": "NAS100", "symbols": ["US100.cash", "NAS100", "USTEC"], "tag": "INDEX", "pill": "moonx-pill-cyan", "price": "30,903.30", "delta": "+1.12%", "up": True, "decimals": 2},
        {"display": "AUD/USD", "symbols": ["AUDUSD"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "0.69857", "delta": "+0.21%", "up": True, "decimals": 5},
        {"display": "USD/CAD", "symbols": ["USDCAD"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "1.42541", "delta": "-0.15%", "up": False, "decimals": 5},
        {"display": "GBP/JPY", "symbols": ["GBPJPY"], "tag": "CROSS", "pill": "moonx-pill-cyan", "price": "209.497", "delta": "-0.12%", "up": False, "decimals": 3},
        {"display": "USD/CHF", "symbols": ["USDCHF"], "tag": "MAJOR", "pill": "moonx-pill-cyan", "price": "0.83015", "delta": "-0.08%", "up": False, "decimals": 5},
        {"display": "XAG/USD", "symbols": ["XAGUSD", "SILVER"], "tag": "SILVER", "pill": "moonx-pill-gold", "price": "60.81", "delta": "+1.92%", "up": True, "decimals": 2},
    ]

    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            for item in popular_assets:
                for s in item["symbols"]:
                    try:
                        mt5.symbol_select(s, True)
                        tick = mt5.symbol_info_tick(s)
                        if tick and getattr(tick, "bid", 0) > 0:
                            item["price"] = f"{tick.bid:,.{item['decimals']}f}"
                            rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, 1)
                            if rates is not None and len(rates) > 0 and rates[0]["open"] > 0:
                                d = ((tick.bid - rates[0]["open"]) / rates[0]["open"]) * 100
                                item["delta"] = f"{d:+.2f}%"
                                item["up"] = d >= 0
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    # Build Ticker Chips HTML
    items_html = ""
    for item in popular_assets:
        delta_class = "moonx-ticker-delta-up" if item["up"] else "moonx-ticker-delta-down"
        items_html += (
            f'<div class="moonx-ticker-item">'
            f'<span class="moonx-ticker-sym">{item["display"]}</span>'
            f'<span class="moonx-pill {item["pill"]}" style="padding:1px 5px; font-size:0.6rem;">{item["tag"]}</span>'
            f'<span class="moonx-ticker-price">{item["price"]}</span>'
            f'<span class="{delta_class}">{item["delta"]}</span>'
            f'</div>'
        )

    raw_html = f"""<div class="moonx-ribbon-dock">
<div class="moonx-ticker-track">
<div class="moonx-ticker-group">
{items_html}
</div>
<div class="moonx-ticker-group" aria-hidden="true">
{items_html}
</div>
</div>
</div>"""

    clean_html = "".join(line.strip() for line in raw_html.splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


def get_logo_html(alignment="center", is_clickable=True):
    """Generate HTML string for the premium logo."""
    padding = "0"
    justify = "center" if alignment == "center" else "flex-start"
    html = f"""
    <div class="logo-container" style="display:inline-flex; align-items:center; justify-content:{justify}; gap:10px; padding:{padding}; cursor:pointer; text-decoration:none;">
        <div style="width:32px; height:32px; border-radius:8px; background:linear-gradient(135deg, #00E5FF 0%, #0077B6 100%); display:flex; align-items:center; justify-content:center; font-size:1rem; box-shadow:0 2px 10px rgba(0,229,255,0.35); flex-shrink:0;">⚡</div>
        <div style="text-align:left;">
            <div class="logo-text">ForexAlert</div>
            <div class="logo-sub">AI Trading Intelligence</div>
        </div>
    </div>
    """
    if is_clickable:
        # We link to "/" to go to landing/main app
        html = f'<a href="/" target="_self" style="text-decoration:none; color:inherit; display:inline-block;">{html}</a>'
    return html


def sidebar_logo():
    """Render premium sidebar logo."""
    st.markdown(get_logo_html("center", is_clickable=True), unsafe_allow_html=True)


def sidebar_footer():
    """Render sidebar version badge with Sentinel Health Pulse."""
    from datetime import datetime, timedelta
    
    # ── Sentinel Health Check ───────────────────────
    status_class = "health-offline"
    status_text = "SENTINEL STALLED"
    
    try:
        db = get_db()
        latest = db.get_recent_signals(limit=1)
        if latest:
            # Check if updated in last 75 mins (Cooldown blocks save for 60m)
            last_ts = datetime.fromisoformat(latest[0]['timestamp'])
            # Ensure TZ awareness
            if last_ts.tzinfo is None:
                from datetime import timezone
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            
            now = datetime.now(last_ts.tzinfo)
            if (now - last_ts) < timedelta(minutes=75):
                status_class = "health-online"
                status_text = "SENTINEL ACTIVE"
    except:
        pass

    st.markdown(f"""
    <div class="health-badge {status_class}">
        <div class="health-dot pulse-animation" style="background: currentColor;"></div>
        {status_text}
    </div>
    <div class="version-badge">
        <strong>ForexAlert</strong> v2.0.2 · Real-Time Sync Active
    </div>
    """, unsafe_allow_html=True)

def render_system_monitor():
    """Render the diagnostic expander in the sidebar. Call once from main app layout."""
    from datetime import datetime, timezone
    with st.sidebar.expander("🛠️ System Monitor (Debug)"):
        st.caption("Environment Truth")
        st.code(f"CWD: {os.getcwd()}")
        st.code(f"DB: {DB_PATH}")
        
        try:
            db = get_db()  # Already defined in this module - no circular import
            latest = db.get_recent_signals(limit=1, include_hidden=True)
            if latest:
                ts = datetime.fromisoformat(latest[0]['timestamp'])
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
                symbol = latest[0].get('symbol', '?')
                sig = latest[0].get('signal', '?')
                st.write(f"📝 Last Record: **{symbol}** ({sig}) — {age:.1f}m ago")
                st.write(f"📊 Total Signals: {db.get_signal_count()}")
            else:
                st.warning("No signals in database yet.")
        except Exception as e:
            st.error(f"Diagnostic Error: {e}")


def section_header(icon, text):
    """Render section divider header with glowing cyber icon badge and scanline rule."""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-icon-wrap">{icon}</div>
        <div class="section-header-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_market_sessions_cockpit():
    """
    Renders real-time Global Market Trading Sessions cockpit:
    Sydney (21:00-06:00 UTC), Tokyo (00:00-09:00 UTC),
    London (08:00-16:30 UTC), New York (13:00-21:00 UTC).
    Highlights session overlaps and live market open/closed status with pulsing indicators.
    """
    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)
    utc_hour = now_utc.hour + now_utc.minute / 60.0
    weekday = now_utc.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    # FX Market weekend status (closed Friday 22:00 UTC to Sunday 21:00 UTC)
    is_weekend = (weekday == 4 and utc_hour >= 21.9) or (weekday == 5) or (weekday == 6 and utc_hour < 21.0)

    # Session calculations in UTC
    sydney_open = not is_weekend and (utc_hour >= 21.0 or utc_hour < 6.0)
    tokyo_open = not is_weekend and (0.0 <= utc_hour < 9.0)
    london_open = not is_weekend and (8.0 <= utc_hour < 16.5)
    ny_open = not is_weekend and (13.0 <= utc_hour < 21.0)

    # Overlap and vitality message
    if is_weekend:
        overlap_tag = '<span style="color:#FFA726;">🛡️ Weekend Market Pause · Forex Halted (Sunday Buffer Active) · Crypto 24/7 Live</span>'
    elif london_open and ny_open:
        overlap_tag = '<span style="color:#00FF88;">🔥 LONDON / NEW YORK OVERLAP ACTIVE · Peak Global Volume & Tight Spreads</span>'
    elif tokyo_open and london_open:
        overlap_tag = '<span style="color:#00E5FF;">⚡ TOKYO / LONDON OVERLAP ACTIVE · European Opening Expansion</span>'
    elif sydney_open and tokyo_open:
        overlap_tag = '<span style="color:#00E5FF;">🌏 ASIAN SESSION ACTIVE · Cross-Liquidity in JPY & AUD Pairs</span>'
    elif london_open:
        overlap_tag = '<span style="color:#00FF88;">🇬🇧 LONDON SESSION ACTIVE · Institutional Flow in EUR & GBP</span>'
    elif ny_open:
        overlap_tag = '<span style="color:#00FF88;">🇺🇸 NEW YORK SESSION ACTIVE · Wall Street & Dollar Liquidity</span>'
    else:
        overlap_tag = '<span style="color:#8e9bb0;">🌐 INTER-SESSION TRANSITION · Neural Watchdog Scanning M15 / H1</span>'

    sessions = [
        {"city": "SYDNEY", "icon": "🇦🇺", "hours": "21:00 – 06:00 UTC", "is_open": sydney_open},
        {"city": "TOKYO", "icon": "🇯🇵", "hours": "00:00 – 09:00 UTC", "is_open": tokyo_open},
        {"city": "LONDON", "icon": "🇬🇧", "hours": "08:00 – 16:30 UTC", "is_open": london_open},
        {"city": "NEW YORK", "icon": "🇺🇸", "hours": "13:00 – 21:00 UTC", "is_open": ny_open},
    ]

    cards_html = ""
    for s in sessions:
        if s["is_open"]:
            cls = "session-card session-open"
            badge = '<span class="session-badge-open"><span class="moonx-pulse-dot moonx-pulse-green"></span> OPEN</span>'
        else:
            cls = "session-card session-closed"
            badge = '<span class="session-badge-closed">CLOSED</span>'

        cards_html += f"""
        <div class="{cls}">
            <div class="session-city">
                <span>{s['icon']} {s['city']}</span>
                {badge}
            </div>
            <div class="session-hours">{s['hours']}</div>
        </div>
        """

    time_str = now_utc.strftime("%H:%M:%S UTC · %A, %b %d")

    cockpit_html = f"""
    <div style="margin-bottom: 20px;">
        <div class="session-grid">
            {cards_html}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; padding:6px 14px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; font-size:0.75rem; font-family:var(--font-mono);">
            <div>{overlap_tag}</div>
            <div style="color:var(--text-secondary); font-weight:600;">⏱️ {time_str}</div>
        </div>
    </div>
    """
    st.markdown(cockpit_html, unsafe_allow_html=True)

