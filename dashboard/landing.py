"""
ApexForex Premium Landing Page (v2.5 High-Energy Edition).

Features:
  - Ambient Cyber Mesh lighting and glow effects.
  - Top live MoonX ticker ribbon dock streaming live MT5 quotes.
  - Radiant neon typography with cyber gold and electric cyan gradients.
  - Institutional Stats Bar (31 Pairs, 61%+ Conviction Gate, <18ms Execution, 100% Autopilot).
  - High-Octane Live Signal Feed with glowing pill badges and transparent win/loss audit.
  - Visual Terminal & Candlestick Showcase widget.
  - Chronological 3-Step "How It Works" with glowing cyber numbers and micro-badges.
  - Modular glass feature cards with illuminated vector disks.
  - Vibrant 3-Tier Pricing deck with Pro Trader popping out with cyan border glow.
  - Interactive FAQ accordions and lead capture form.
  - Smooth-scrolling anchor links and safe Streamlit callback state management.
"""
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_DASHBOARD_DIR = str(Path(__file__).resolve().parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from theme import inject_css, get_logo_html, render_moonx_ticker_ribbon
except ImportError:
    from dashboard.theme import inject_css, get_logo_html, render_moonx_ticker_ribbon

AUTH_SIGNUP = "Create account"
AUTH_LOGIN = "Log in"
CONTACT_TOPICS = ["General question", "Pro Trader plan", "Institutional plan", "Partnership"]
MIN_PASSWORD_LEN = 8


# ── Helpers ──────────────────────────────────────────────────────────────────
def _html(markup: str) -> str:
    """Strip indentation/blank lines so Markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in markup.splitlines() if line.strip())


def _icon(name: str) -> str:
    paths = {
        "brain": '<path d="M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10.5a3 3 0 0 0 1.5 2.6A3 3 0 0 0 9 18a3 3 0 0 0 3 2V4a3 3 0 0 0-3 0Z"/><path d="M15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 3 3 3 0 0 1-1.5 2.6A3 3 0 0 1 15 18a3 3 0 0 1-3 2"/>',
        "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
        "bell": '<path d="M6 9a6 6 0 1 1 12 0c0 6 2 7 2 7H4s2-1 2-7Z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
        "check": '<path d="m5 12 4.5 4.5L19 7"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    }
    return (
        f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'{paths.get(name, paths["zap"])}</svg>'
    )


def _rel_time(ts) -> str:
    try:
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    except Exception:
        return ""
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


@st.cache_data(ttl=30, show_spinner=False)
def _recent_signals(limit: int = 4):
    """Latest non-WAIT signals from the live database. Cached for 30s."""
    try:
        try:
            from theme import get_db
        except ImportError:
            from dashboard.theme import get_db
        rows = get_db().get_recent_signals(limit=limit * 3)
    except Exception:
        return []
    out = []
    for s in rows:
        if s.get("signal") not in ("BUY", "SELL"):
            continue
        out.append({
            "symbol": s.get("symbol", ""),
            "side": s["signal"],
            "price": s.get("price_at_signal") or 0.0,
            "outcome": s.get("outcome") or "ACTIVE",
            "ts": s.get("timestamp"),
        })
        if len(out) == limit:
            break
    return out


def _feed_html() -> str:
    signals = _recent_signals()
    if not signals:
        body = (
            '<div class="ax-feed-empty">'
            '<span class="pulse-dot" style="display:inline-block; margin-right:6px;"></span>'
            'Surveillance Engine Scanning 31 Pairs · New trade signals will populate automatically.'
            '</div>'
        )
    else:
        status_map = {
            "SUCCESS": ("🎯 Hit TP", "chip-win"),
            "FAIL": ("🛡️ Stopped Out", "chip-loss"),
            "ACTIVE": ("⚡ Live Trade", "chip-open"),
        }
        rows = []
        for s in signals:
            label, chip_cls = status_map.get(s["outcome"], ("⚡ Open", "chip-open"))
            side_cls = "side-buy" if s["side"] == "BUY" else "side-sell"
            dec = 2 if ("XAU" in s["symbol"] or "BTC" in s["symbol"]) else (3 if "JPY" in s["symbol"] else 5)
            rows.append(
                f'<div class="ax-feed-row">'
                f'<span class="ax-side-badge {side_cls}">{s["side"]}</span>'
                f'<span class="ax-sym-name">{html.escape(str(s["symbol"]))}</span>'
                f'<span class="ax-price-val font-mono-numbers">{s["price"]:.{dec}f}</span>'
                f'<span class="ax-chip-badge {chip_cls}">{label}</span>'
                f'<span class="ax-time-val">{_rel_time(s["ts"])}</span>'
                f'</div>'
            )
        body = "".join(rows)

    return _html(
        f"""
        <div class="ax-feed" role="region" aria-label="Latest signals">
          <div class="ax-feed-head">
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:1rem;">📡</span>
              <span style="font-weight:700; color:#ffffff; font-size:0.85rem; letter-spacing:0.04em; text-transform:uppercase;">Live Signal Radar</span>
            </div>
            <div class="ax-live-badge">
              <span class="pulse-dot"></span>
              <span>Scanning 31 Pairs</span>
            </div>
          </div>
          <div class="space-y-1.5 mt-2">
            {body}
          </div>
        </div>
        """
    )


def _stats_strip_html() -> str:
    return _html("""
    <div class="ax-stats-strip">
      <div class="ax-stat-box">
        <div class="ax-stat-num cyan">31 Pairs</div>
        <div class="ax-stat-lbl">Forex Majors, Crosses & Gold</div>
      </div>
      <div class="ax-stat-box">
        <div class="ax-stat-num gold">61%+ Gate</div>
        <div class="ax-stat-lbl">Institutional Conviction Filter</div>
      </div>
      <div class="ax-stat-box">
        <div class="ax-stat-num green">&lt; 18ms</div>
        <div class="ax-stat-lbl">Direct MT5 Broker Bridge</div>
      </div>
      <div class="ax-stat-box">
        <div class="ax-stat-num cyan">100% Auto</div>
        <div class="ax-stat-lbl">Hands-Free Cloud Execution</div>
      </div>
    </div>
    """)


def _terminal_showcase_html() -> str:
    return _html("""
    <div class="ax-showcase-card">
      <div class="ax-showcase-header">
        <div style="display:flex; align-items:center; gap:10px;">
          <div class="ax-showcase-icon">⚡</div>
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-weight:800; font-size:1rem; color:#ffffff;">EURUSD</span>
              <span class="ax-chip-badge chip-open" style="font-size:0.65rem;">M15 Confluence Engine</span>
              <span class="font-mono-numbers" style="color:#00FF88; font-weight:700; font-size:0.85rem;">1.12020</span>
            </div>
            <div style="font-size:0.7rem; color:#7d8aa3;">Autonomous AI Positioning · Auto-SL &amp; Partial Profit Scaled</div>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="ax-pill-micro pill-cyan">GMM: TRENDING</span>
          <span class="ax-pill-micro pill-green">CONVICTION: 74%</span>
        </div>
      </div>

      <!-- Simulated Target Ladder & Visual Levels -->
      <div class="ax-showcase-grid">
        <div class="ax-level-badge level-entry">
          <div class="level-lbl">ENTRY LEVEL</div>
          <div class="level-val font-mono-numbers">1.12020</div>
          <div class="level-sub">Verified Signal Price</div>
        </div>
        <div class="ax-level-badge level-tp">
          <div class="level-lbl">TAKE PROFIT (TP)</div>
          <div class="level-val font-mono-numbers">1.12395</div>
          <div class="level-sub" style="color:#00FF88;">+37.5 Pips Target</div>
        </div>
        <div class="ax-level-badge level-sl">
          <div class="level-lbl">STOP LOSS (SL)</div>
          <div class="level-val font-mono-numbers">1.11770</div>
          <div class="level-sub" style="color:#FF4466;">-25.0 Pips Shield</div>
        </div>
        <div class="ax-level-badge level-rr">
          <div class="level-lbl">PARTIAL BE GUARDIAN</div>
          <div class="level-val font-mono-numbers">+2.0 Pips</div>
          <div class="level-sub" style="color:#FFD700;">25% Closed &amp; BE Moved</div>
        </div>
      </div>
    </div>
    """)


def _go(auth_mode=None, scroll=None, topic=None):
    """Callback: runs before the rerun so setting widget state is legal."""
    if auth_mode:
        st.session_state["auth_mode"] = auth_mode
    if topic:
        st.session_state["c_topic"] = topic
    if scroll:
        st.session_state["_scroll_to"] = scroll


def _run_scroll():
    """Smooth scroll after pricing or contact button click."""
    target = st.session_state.pop("_scroll_to", None)
    if not target:
        return
    js_target = "null" if target == "top" else f'"{target}"'
    script = f"""<script>
    const d = window.parent.document;
    const id = {js_target};
    const el = id ? d.getElementById(id) : null;
    if (el) {{ el.scrollIntoView({{behavior: "smooth", block: "start"}}); }}
    else {{
      const m = d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main');
      if (m) m.scrollTo({{top: 0, behavior: "smooth"}});
    }}
    </script>"""
    try:
        if hasattr(st, "iframe"):
            st.iframe(script, height=1)
        else:
            components.html(script, height=0)
    except Exception:
        pass


# ── High-Energy CSS Design System ────────────────────────────────────────────
_CSS = """
<style>
/* ── Design Tokens & Base Reset ── */
:root {
    --text-muted: #7d8aa3;
    --accent-cyan: #00E5FF;
    --accent-gold: #FFD700;
    --signal-buy: #00FF88;
    --signal-sell: #FF4466;
    --bg-card: rgba(13, 19, 36, 0.75);
    --border-glass: rgba(255, 255, 255, 0.08);
    --radius-lg: 16px;
    --radius-md: 12px;
    --font-ui: 'Inter', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}
html { scroll-behavior: smooth; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    padding-top: 1rem !important;
    max-width: 1140px !important;
}

/* Background Atmosphere Glow */
.stApp {
    background: radial-gradient(ellipse at 50% -10%, #0d1e3d 0%, #060a16 55%, #04060c 100%) !important;
}

/* Focus Rings */
a:focus-visible, button:focus-visible, input:focus-visible,
textarea:focus-visible, [role="radio"]:focus-visible, summary:focus-visible {
    outline: 2px solid var(--accent-cyan) !important;
    outline-offset: 3px !important;
}

/* ── Typography & Gradients ── */
.ax-h1 {
    font-size: clamp(2.3rem, 5vw, 3.4rem) !important;
    line-height: 1.12 !important;
    font-weight: 900 !important;
    margin: 12px 0 16px 0 !important;
    letter-spacing: -0.03em !important;
    color: #ffffff !important;
}
.neon-gradient {
    background: linear-gradient(135deg, #00E5FF 0%, #33EBFF 45%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.ax-lead {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #8e9bb0;
    max-width: 36em;
    margin: 0 0 22px 0;
}

/* Hero Pill Badge */
.ax-pill-hero {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.3);
    border-radius: 30px;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: var(--font-mono);
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.18);
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--signal-buy);
    box-shadow: 0 0 8px var(--signal-buy);
    animation: ax-pulse 2s ease-in-out infinite;
}
@keyframes ax-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

/* Trust Checklist */
.ax-checks {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 22px;
    margin: 0 0 26px 0;
    padding: 0;
    list-style: none;
    font-size: 0.9rem;
    color: #ffffff;
}
.ax-checks li { display: flex; align-items: center; gap: 8px; }
.check-icon {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(0, 255, 136, 0.15);
    border: 1px solid rgba(0, 255, 136, 0.35);
    color: var(--signal-buy);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
}

/* ── Live Signal Feed (The Centerpiece) ── */
.ax-feed {
    background: rgba(10, 15, 30, 0.78);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 229, 255, 0.22);
    border-radius: var(--radius-lg);
    padding: 14px 18px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 24px rgba(0, 229, 255, 0.08);
}
.ax-feed-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.ax-live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 10px;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 12px;
    color: var(--signal-buy);
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--font-mono);
}
.ax-feed-row {
    display: grid;
    grid-template-columns: 54px 1fr auto 100px 70px;
    gap: 12px;
    align-items: center;
    padding: 9px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid transparent;
    transition: all 0.2s ease;
    font-size: 0.88rem;
}
.ax-feed-row:hover {
    background: rgba(0, 229, 255, 0.05);
    border-color: rgba(0, 229, 255, 0.25);
    transform: translateX(3px);
}
.ax-side-badge {
    font-weight: 800;
    font-size: 0.7rem;
    padding: 3px 0;
    border-radius: 5px;
    text-align: center;
    letter-spacing: 0.05em;
    font-family: var(--font-mono);
}
.ax-side-badge.side-buy {
    color: var(--signal-buy);
    background: rgba(0, 255, 136, 0.14);
    border: 1px solid rgba(0, 255, 136, 0.35);
}
.ax-side-badge.side-sell {
    color: var(--signal-sell);
    background: rgba(255, 68, 102, 0.14);
    border: 1px solid rgba(255, 68, 102, 0.35);
}
.ax-sym-name { font-weight: 800; color: #ffffff; }
.ax-price-val { color: #8e9bb0; font-size: 0.82rem; }
.ax-chip-badge {
    font-size: 0.75rem;
    font-weight: 700;
    text-align: right;
    font-family: var(--font-mono);
}
.chip-win  { color: var(--signal-buy); text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }
.chip-loss { color: var(--signal-sell); }
.chip-open { color: var(--accent-cyan); text-shadow: 0 0 10px rgba(0, 229, 255, 0.3); }
.ax-time-val { font-size: 0.75rem; color: var(--text-muted); text-align: right; }
.ax-feed-empty { padding: 24px; text-align: center; color: #8e9bb0; font-size: 0.88rem; }

/* ── Stats Strip Bar ── */
.ax-stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 16px 20px;
    background: rgba(13, 19, 36, 0.72);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    margin: 32px 0 40px 0;
}
.ax-stat-box {
    text-align: center;
    padding: 6px 12px;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
.ax-stat-box:last-child { border-right: none; }
.ax-stat-num {
    font-family: var(--font-mono);
    font-size: 1.55rem;
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.ax-stat-num.cyan  { color: var(--accent-cyan); text-shadow: 0 0 14px rgba(0, 229, 255, 0.4); }
.ax-stat-num.gold  { color: var(--accent-gold); text-shadow: 0 0 14px rgba(255, 215, 0, 0.4); }
.ax-stat-num.green { color: var(--signal-buy);  text-shadow: 0 0 14px rgba(0, 255, 136, 0.4); }
.ax-stat-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Interactive Terminal Showcase Card ── */
.ax-showcase-card {
    background: rgba(13, 19, 36, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 40px;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
}
.ax-showcase-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-wrap: wrap;
    gap: 12px;
}
.ax-showcase-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #00E5FF 0%, #0077B6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 2px 10px rgba(0, 229, 255, 0.35);
}
.ax-pill-micro {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    font-family: var(--font-mono);
}
.pill-cyan { background: rgba(0, 229, 255, 0.1); color: var(--accent-cyan); border: 1px solid rgba(0, 229, 255, 0.3); }
.pill-green { background: rgba(0, 255, 136, 0.1); color: var(--signal-buy); border: 1px solid rgba(0, 255, 136, 0.3); }
.ax-showcase-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 14px;
}
.ax-level-badge {
    padding: 12px 14px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: all 0.2s ease;
}
.ax-level-badge:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 229, 255, 0.3);
}
.level-entry { border-left: 3px solid var(--accent-cyan); }
.level-tp    { border-left: 3px solid var(--signal-buy); }
.level-sl    { border-left: 3px solid var(--signal-sell); }
.level-rr    { border-left: 3px solid var(--accent-gold); }
.level-lbl { font-size: 0.65rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase; }
.level-val { font-size: 1.15rem; font-weight: 800; color: #ffffff; margin: 2px 0; }
.level-sub { font-size: 0.7rem; font-weight: 600; color: #8e9bb0; }

/* ── Cyber Steps Grid (How it works) ── */
.ax-steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 16px;
}
.ax-step-card {
    background: rgba(13, 19, 36, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 3px solid var(--accent-cyan);
    border-radius: var(--radius-lg);
    padding: 24px 22px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
.ax-step-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 229, 255, 0.45);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 229, 255, 0.15);
}
.ax-step-num {
    font-family: var(--font-mono);
    font-size: 1.35rem;
    font-weight: 900;
    color: var(--accent-cyan);
    text-shadow: 0 0 14px rgba(0, 229, 255, 0.4);
    margin-bottom: 6px;
}
.ax-step-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    background: rgba(0, 229, 255, 0.08);
    color: var(--accent-cyan);
    border: 1px solid rgba(0, 229, 255, 0.25);
    margin-bottom: 12px;
}
.ax-step-card h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: #ffffff !important; margin: 0 0 8px 0 !important; }
.ax-step-card p { font-size: 0.88rem !important; color: #8e9bb0 !important; line-height: 1.6 !important; margin: 0 !important; }

/* ── Features Cards ── */
.ax-feature-card {
    background: rgba(13, 19, 36, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 26px 22px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}
.ax-feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 229, 255, 0.4);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 229, 255, 0.15);
}
.ax-feature-icon-disk {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.3);
    color: var(--accent-cyan);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.15);
}
.ax-feature-card h3 { font-size: 1.15rem !important; font-weight: 700 !important; color: #ffffff !important; margin: 0 0 8px 0 !important; }
.ax-feature-card p { font-size: 0.88rem !important; color: #8e9bb0 !important; line-height: 1.65 !important; margin: 0 !important; }

/* ── Pricing Deck ── */
.ax-plan-card {
    background: rgba(13, 19, 36, 0.72);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 26px 24px;
    min-height: 290px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.25s ease;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 14px;
}
.ax-plan-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 229, 255, 0.35);
}
.ax-plan-card.pro {
    border: 2px solid var(--accent-cyan) !important;
    background: linear-gradient(180deg, rgba(0, 229, 255, 0.12) 0%, rgba(10, 15, 30, 0.95) 100%) !important;
    box-shadow: 0 0 35px rgba(0, 229, 255, 0.22), 0 14px 40px rgba(0, 0, 0, 0.5) !important;
}
.ax-plan-badge {
    font-size: 0.65rem;
    font-weight: 800;
    color: #030814;
    background: var(--accent-cyan);
    border-radius: 12px;
    padding: 3px 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.5);
}
.ax-plan-badge.gold {
    background: var(--accent-gold);
    color: #030814;
    box-shadow: 0 0 12px rgba(255, 215, 0, 0.5);
}
.ax-price-tag {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 900;
    color: #ffffff;
    margin: 6px 0 16px 0;
    letter-spacing: -0.02em;
}
.ax-price-tag small {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    font-weight: 500;
    color: #8e9bb0;
}
.ax-plan-card ul {
    margin: 0;
    padding-left: 20px;
    color: #8e9bb0;
    font-size: 0.88rem;
    line-height: 1.9;
}
.ax-plan-card.pro ul { color: #ffffff; }

/* ── Section Titles ── */
.ax-sec-title { font-size: 1.85rem !important; font-weight: 800 !important; margin: 0 0 6px 0 !important; color: #ffffff !important; }
.ax-sec-sub { color: #8e9bb0; margin: 0 0 24px 0; max-width: 42em; line-height: 1.6; font-size: 0.95rem; }
.ax-spacer { height: 60px; }

/* ── Nav Header ── */
.ax-nav { display: flex; justify-content: flex-end; gap: 28px; padding-top: 16px; }
.ax-nav a { color: #8e9bb0; text-decoration: none; font-weight: 600; font-size: 0.9rem; transition: color 0.2s; }
.ax-nav a:hover { color: var(--accent-cyan); }

/* ── Responsive Mobile ── */
@media (max-width: 760px) {
    .ax-nav { display: none; }
    .ax-stats-strip { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .ax-stat-box:nth-child(2) { border-right: none; }
    .ax-showcase-grid { grid-template-columns: 1fr 1fr; }
    .ax-steps-grid { grid-template-columns: 1fr; }
    .ax-feed-row { grid-template-columns: 48px 1fr auto 86px; }
    .ax-time-val { display: none; }
    .ax-spacer { height: 36px; }
}
</style>
"""


def _render_login():
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
          <span style="font-size:1.3rem;">⚡</span>
          <h3 style="margin:0; font-size:1.2rem; font-weight:800; color:#ffffff;">Welcome Back</h3>
        </div>
        <p style="font-size:0.8rem; color:#7d8aa3; margin:0 0 16px 0;">Log in to access your autonomous copy fleet.</p>
        """, unsafe_allow_html=True)

        with st.form("form_login", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@example.com", key="f_login_email")
            pw = st.text_input("Password", type="password", key="f_login_pw")
            submitted = st.form_submit_button("Log in to Terminal 🚀", type="primary", use_container_width=True)

        if submitted:
            if not email or not pw:
                st.error("Enter your email and password.")
                return
            from core.auth import verify_login
            from core.sessions import create_session
            user = verify_login(email.strip(), pw)
            if not user:
                st.error("Email or password is incorrect.")
                return
            st.session_state["_pending_auth"] = {
                "email": user["email"], "name": user["name"], "role": user["role"],
                "id": user["id"], "token": create_session(user),
            }
            st.rerun()


def _render_signup():
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
          <span style="font-size:1.3rem;">🚀</span>
          <h3 style="margin:0; font-size:1.2rem; font-weight:800; color:#ffffff;">Start 14-Day Free Trial</h3>
        </div>
        <p style="font-size:0.8rem; color:#00E5FF; font-weight:600; margin:0 0 16px 0;">Instant setup · Zero credit card required</p>
        """, unsafe_allow_html=True)

        with st.form("form_signup", clear_on_submit=False):
            name = st.text_input("Full name", placeholder="John Smith", key="f_su_name")
            email = st.text_input("Email", placeholder="you@example.com", key="f_su_email")
            pw = st.text_input(
                "Password", type="password", key="f_su_pw",
                help=f"At least {MIN_PASSWORD_LEN} characters.",
            )
            submitted = st.form_submit_button("Create Free Account ⚡", type="primary", use_container_width=True)

        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; gap:6px; font-size:0.75rem; color:#7d8aa3; margin-top:8px;">
          <span>🛡️</span>
          <span>100% Risk Free · Instant Terminal Access</span>
        </div>
        """, unsafe_allow_html=True)

        if not submitted:
            return
        if not (name.strip() and email.strip() and pw):
            st.error("Fill in your name, email and password.")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
            st.error("Enter a valid email address, like you@example.com.")
        elif len(pw) < MIN_PASSWORD_LEN:
            st.error(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
        else:
            from core.auth import register_portal_user, get_portal_user_by_email
            from core.user_accounts import get_user_by_email, add_user
            from core.sessions import create_session

            if get_portal_user_by_email(email):
                st.error("An account with this email already exists. Choose Log in instead.")
                return
            new_user = register_portal_user(name, email, pw)
            if not new_user:
                st.error("We couldn't create your account. Try again in a moment.")
                return
            if not get_user_by_email(email):
                add_user(name=name, email=email, mt5_login="", mt5_password="", mt5_server="")
            st.session_state["_pending_auth"] = {
                "email": new_user["email"], "name": new_user["name"], "role": "subscriber",
                "id": new_user["id"], "token": create_session(new_user),
            }
            st.rerun()


def show_landing():
    inject_css()
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("auth_mode", AUTH_SIGNUP)
    _run_scroll()

    # ── Nav Header ───────────────────────────────────────────────────────────
    n1, n2 = st.columns([2, 3])
    with n1:
        st.markdown(get_logo_html("left", is_clickable=True), unsafe_allow_html=True)
    with n2:
        st.markdown(
            _html("""
            <nav class="ax-nav" aria-label="Page sections">
              <a href="#how-it-works">How it works</a>
              <a href="#terminal-showcase">Terminal</a>
              <a href="#features">Features</a>
              <a href="#pricing">Pricing</a>
              <a href="#faq">FAQ</a>
              <a href="#contact">Contact</a>
            </nav>"""),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Top Live Ticker Ribbon Dock (MoonX Style) ────────────────────────────
    render_moonx_ticker_ribbon()
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Hero Split Section ───────────────────────────────────────────────────
    left, right = st.columns([1.18, 0.82], gap="large")
    with left:
        st.markdown(
            _html(f"""
            <div class="ax-pill-hero">
              <span class="pulse-dot"></span>
              <span>⚡ Institutional AI Surveillance · 31 Forex Pairs</span>
            </div>
            <h1 class="ax-h1">
              Institutional AI Signals.
              <span class="neon-gradient">Auto-Copied to Your MT5.</span>
            </h1>
            <p class="ax-lead">
              Specialist neural models scan 31 forex pairs and spot commodities around the clock. The moment high-conviction confluence triggers, the trade executes directly on your broker account with precision risk and stop protection.
            </p>
            <ul class="ax-checks">
              <li><span class="check-icon">✓</span> <strong>14 Days Free</strong> (Full Access)</li>
              <li><span class="check-icon">✓</span> <strong>No Credit Card</strong> Required</li>
              <li><span class="check-icon">✓</span> <strong>Instant MT5 &amp; Prop Firm</strong> Sync</li>
            </ul>"""),
            unsafe_allow_html=True,
        )
        st.markdown(_feed_html(), unsafe_allow_html=True)

    with right:
        mode = st.radio(
            "Account", [AUTH_SIGNUP, AUTH_LOGIN], horizontal=True,
            label_visibility="collapsed", key="auth_mode",
        )
        if mode == AUTH_SIGNUP:
            _render_signup()
        else:
            _render_login()

    # ── Institutional Stats Strip Bar ────────────────────────────────────────
    st.markdown(_stats_strip_html(), unsafe_allow_html=True)

    # ── Interactive Terminal Showcase ────────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="terminal-showcase"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="ax-sec-title">Execution Architecture</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Real-time level targeting with automated partial profits and prop firm risk guardrails.</p>', unsafe_allow_html=True)
    st.markdown(_terminal_showcase_html(), unsafe_allow_html=True)

    # ── How It Works (Cyber Sequence) ────────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="how-it-works"></div>', unsafe_allow_html=True)
    st.markdown(
        _html("""
        <h2 class="ax-sec-title">How It Works</h2>
        <p class="ax-sec-sub">From instant registration to your first automated trade copy.</p>
        <div class="ax-steps-grid">
          <div class="ax-step-card">
            <div class="ax-step-num">01</div>
            <div class="ax-step-badge">1-Minute Setup</div>
            <h3>Create Your Free Account</h3>
            <p>Sign up in seconds. Your 14-day free trial activates immediately with full access to all 31 currency models—no credit card needed.</p>
          </div>
          <div class="ax-step-card">
            <div class="ax-step-num">02</div>
            <div class="ax-step-badge">Direct IPC Bridge</div>
            <h3>Connect MetaTrader 5</h3>
            <p>Input your MT5 account credentials in the Copy Trading Hub. Set your preferred risk per trade (0.5%–2%) and daily drawdown guardrails.</p>
          </div>
          <div class="ax-step-card">
            <div class="ax-step-num">03</div>
            <div class="ax-step-badge">100% Autopilot</div>
            <h3>Autopilot Execution</h3>
            <p>The neural engine detects high-conviction entries, places orders with automatic SL/TP, locks breakeven at +2p, and alerts you via Telegram.</p>
          </div>
        </div>"""),
        unsafe_allow_html=True,
    )

    # ── Features ("What You Get") ────────────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="features"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="ax-sec-title">What You Get</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Engineered specifically for traders who want precision signals executed, not just listed.</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    cards = [
        (f1, "brain", "31 Autonomous Models",
         "Each currency pair operates on its own dedicated specialist neural network, continuously updated as volatility and market regimes shift."),
        (f2, "copy", "Instant Copy Trading",
         "Trades mirror to your MT5 terminal in under 18ms. Lot sizes and stop protections are calculated dynamically based on your account balance."),
        (f3, "bell", "Real-Time Telegram Pushes",
         "Every entry, breakeven adjustment, and take-profit hit delivers instant Telegram push notifications with full transparency."),
    ]
    for col, icon, title, body in cards:
        with col:
            st.markdown(
                _html(f"""
                <div class="ax-feature-card">
                  <div class="ax-feature-icon-disk">{_icon(icon)}</div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </div>"""),
                unsafe_allow_html=True,
            )

    # ── Pricing (3-Tier High-Octane Deck) ────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="pricing"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="ax-sec-title">Transparent Pricing</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Start 100% free. Upgrade only when you are ready to scale.</p>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.markdown(
            _html("""
            <div class="ax-plan-card">
              <div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <h3 style="font-weight:800; font-size:1.15rem; color:#ffffff; margin:0;">Free Trial</h3>
                  <span class="ax-plan-badge" style="background:#00FF88; color:#030814;">14 Days</span>
                </div>
                <div class="ax-price-tag">$0<small> for 14 days</small></div>
                <ul>
                  <li>Full copy trading enabled</li>
                  <li>All 31 pair models</li>
                  <li>Real-time Telegram alerts</li>
                  <li>Zero credit card required</li>
                </ul>
              </div>
            </div>"""),
            unsafe_allow_html=True,
        )
        st.button("Start Free Trial 🚀", type="primary", use_container_width=True, key="p_btn_trial",
                  on_click=_go, kwargs={"auth_mode": AUTH_SIGNUP, "scroll": "top"})

    with p2:
        st.markdown(
            _html("""
            <div class="ax-plan-card pro">
              <div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <h3 style="font-weight:800; font-size:1.15rem; color:#ffffff; margin:0;">Pro Trader</h3>
                  <span class="ax-plan-badge">Recommended</span>
                </div>
                <div class="ax-price-tag" style="color:#00E5FF;">$49<small style="color:#e0f7fa;"> per month</small></div>
                <ul>
                  <li><strong>All 31 pairs</strong> scanned &amp; copied</li>
                  <li><strong>Instant 18ms</strong> MT5 execution</li>
                  <li><strong>Automated lot sizing</strong> &amp; risk guards</li>
                  <li><strong>Breakeven +2p</strong> partial exit automation</li>
                  <li>Full Telegram signal notifications</li>
                </ul>
              </div>
            </div>"""),
            unsafe_allow_html=True,
        )
        st.button("Get Pro Trader ⚡", type="primary", use_container_width=True, key="p_btn_pro",
                  on_click=_go, kwargs={"auth_mode": AUTH_SIGNUP, "scroll": "top"})

    with p3:
        st.markdown(
            _html("""
            <div class="ax-plan-card">
              <div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <h3 style="font-weight:800; font-size:1.15rem; color:#ffffff; margin:0;">Institutional</h3>
                  <span class="ax-plan-badge gold">Prop Fleet</span>
                </div>
                <div class="ax-price-tag">$199<small> per month</small></div>
                <ul>
                  <li>Dedicated low-latency VPS</li>
                  <li>Multi-account prop firm fleet mirror</li>
                  <li>Custom max daily drawdown safety rules</li>
                  <li>Direct 1-on-1 engineer Telegram support</li>
                </ul>
              </div>
            </div>"""),
            unsafe_allow_html=True,
        )
        st.button("Talk to Us 💬", use_container_width=True, key="p_btn_inst",
                  on_click=_go, kwargs={"topic": "Institutional plan", "scroll": "contact"})

    # ── FAQ Accordion ────────────────────────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="faq"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="ax-sec-title">Frequently Asked Questions</h2>', unsafe_allow_html=True)
    with st.expander("⚡ What do I need to get started?"):
        st.write("Just an email address to sign up and a MetaTrader 5 account (demo, live broker, or prop firm challenge). You can add your MT5 credentials inside the Copy Trading Hub immediately after creating your account.")
    with st.expander("🛡️ Is a credit card required for the trial?"):
        st.write("No. The 14-day trial requires no credit card, no deposit, and no commitment. You receive full access to all 31 pairs and automated copy trading.")
    with st.expander("🎯 Does this work with prop firms like FTMO or FundingPips?"):
        st.write("Yes! The system includes built-in prop firm safety shields: maximum daily drawdown cutoffs (4.5%), Sunday 2-hour opening spread avoidance, and Friday auto-exit (5 minutes before weekend close) to keep your prop challenge safe.")
    with st.expander("📊 Do the signals guarantee profits?"):
        st.write("No. No authentic trading system does. Some trades will hit stop-loss, which is why our Live Radar transparently displays historical losses alongside winners. The system operates on risk-reward asymmetry and statistical conviction.")

    # ── Contact Us ───────────────────────────────────────────────────────────
    st.markdown('<div class="ax-spacer" id="contact"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="ax-sec-title">Contact &amp; Custom Deployments</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Questions about custom prop firm rules, multi-account fleets, or broker setups? We reply within 24 hours.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("form_contact_lead", clear_on_submit=True):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                inq_name = st.text_input("Your name", placeholder="John Smith", key="c_name")
                inq_email = st.text_input("Email", placeholder="you@example.com", key="c_email")
                topic = st.selectbox("I'm asking about", CONTACT_TOPICS, key="c_topic")
            with c2:
                inq_msg = st.text_area(
                    "Message", height=176, key="c_msg_input",
                    placeholder="Tell us about your setup: broker, prop firm, account size, number of accounts.",
                )
            sent = st.form_submit_button("Send Message 🚀", type="primary", use_container_width=True)

        if sent:
            if not (inq_name.strip() and inq_email.strip() and inq_msg.strip()):
                st.error("Fill in your name, email and message.")
            else:
                from core.leads import submit_inquiry
                submit_inquiry(inq_name, inq_email, f"[{topic}] {inq_msg}")
                st.success("Message sent! Our engineering team will reply within 24 hours.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown('<div class="ax-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        _html("""
        <footer style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 24px; padding-bottom: 32px; color: #7d8aa3; font-size: 0.8rem; line-height: 1.7;">
          <p style="margin: 0 0 10px 0; max-width: 56em;">
            <strong style="color:#ffffff;">Risk Warning:</strong> Trading foreign exchange and leveraged derivative products carries substantial risk of capital loss and is not suitable for all investors. Past performance does not guarantee future results. ApexForex provides institutional software tools and copy automation, not financial advisory services.
          </p>
          <div style="font-size: 0.72rem; color: #526079;">
            &copy; 2026 ApexForex · AI Trading Intelligence. All rights reserved.
          </div>
        </footer>
        """),
        unsafe_allow_html=True,
    )
