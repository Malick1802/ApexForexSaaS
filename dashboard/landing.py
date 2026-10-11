"""
ApexForex landing page (v2).

Changes vs. v1:
  - Sign-up first: the hero leads with the free trial and a short form.
  - Live signal feed is the visual centrepiece; it is cached (30s) and shows
    losses as well as wins.
  - Removed unverifiable claims, the "trusted technologies" strip and balloons.
  - Added a real "how it works" sequence, FAQ and a risk disclaimer.
  - Pricing / contact buttons now use callbacks (setting a widget's session
    key after it is rendered raises StreamlitAPIException in v1) and scroll
    the user to the right place.
  - Responsive CSS, visible keyboard focus, reduced-motion support and a
    higher-contrast muted text colour.
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
    from theme import inject_css, get_logo_html
except ImportError:
    from dashboard.theme import inject_css, get_logo_html

AUTH_SIGNUP = "Create account"
AUTH_LOGIN = "Log in"
CONTACT_TOPICS = ["General question", "Pro Trader plan", "Institutional plan", "Partnership"]
MIN_PASSWORD_LEN = 8


# ── helpers ──────────────────────────────────────────────────────────────────
def _html(markup: str) -> str:
    """Strip indentation/blank lines so Markdown never treats the HTML as code."""
    return "".join(line.strip() for line in markup.splitlines())


def _icon(name: str) -> str:
    paths = {
        "brain": '<path d="M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10.5a3 3 0 0 0 1.5 2.6A3 3 0 0 0 9 18a3 3 0 0 0 3 2V4a3 3 0 0 0-3 0Z"/><path d="M15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 3 3 3 0 0 1-1.5 2.6A3 3 0 0 1 15 18a3 3 0 0 1-3 2"/>',
        "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
        "bell": '<path d="M6 9a6 6 0 1 1 12 0c0 6 2 7 2 7H4s2-1 2-7Z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
        "check": '<path d="m5 12 4.5 4.5L19 7"/>',
    }
    return (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{paths[name]}</svg>"
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
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return f"{secs // 86400} d ago"


@st.cache_data(ttl=30, show_spinner=False)
def _recent_signals(limit: int = 4):
    """Latest non-WAIT signals. Cached so every visitor doesn't hit the DB."""
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
            '<div class="ax-feed-empty">No signals yet. New setups appear here '
            "as soon as the models find them.</div>"
        )
    else:
        status = {
            "SUCCESS": ("Hit target", "win"),
            "FAIL": ("Stopped out", "loss"),
        }
        rows = []
        for s in signals:
            label, kind = status.get(s["outcome"], ("Open", "open"))
            side_cls = "buy" if s["side"] == "BUY" else "sell"
            rows.append(
                f'<div class="ax-row">'
                f'<span class="ax-side {side_cls}">{s["side"]}</span>'
                f'<span class="ax-sym">{html.escape(str(s["symbol"]))}</span>'
                f'<span class="ax-price">{s["price"]:.5f}</span>'
                f'<span class="ax-chip {kind}">{label}</span>'
                f'<span class="ax-when">{_rel_time(s["ts"])}</span>'
                f"</div>"
            )
        body = "".join(rows)
    return _html(
        f"""
        <div class="ax-feed" role="region" aria-label="Latest signals">
          <div class="ax-feed-head">
            <span>Latest signals</span>
            <span class="ax-live"><i></i>Scanning 31 pairs</span>
          </div>
          {body}
        </div>
        """
    )


def _go(auth_mode=None, scroll=None, topic=None):
    """Callback: runs *before* the rerun, so setting widget state here is legal."""
    if auth_mode:
        st.session_state["auth_mode"] = auth_mode
    if topic:
        st.session_state["c_topic"] = topic
    if scroll:
        st.session_state["_scroll_to"] = scroll


def _run_scroll():
    """Best-effort smooth scroll after a pricing/contact button click."""
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
        if hasattr(st, "iframe"):          # newer Streamlit (components.html is deprecated)
            st.iframe(script, height=1)
        else:
            components.html(script, height=0)
    except Exception:
        pass  # scrolling is a nicety; never break the page for it


# ── page ─────────────────────────────────────────────────────────────────────
_CSS = """
<style>
:root { --text-muted: #7d8aa3; }
html { scroll-behavior: smooth; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding-top: 1.25rem !important; max-width: 1120px !important; }

a:focus-visible, button:focus-visible, input:focus-visible,
textarea:focus-visible, [role="radio"]:focus-visible, summary:focus-visible {
    outline: 2px solid var(--accent-cyan) !important; outline-offset: 3px !important;
}

/* nav */
.ax-nav { display:flex; justify-content:flex-end; gap:28px; padding-top:20px; }
.ax-nav a { color: var(--text-secondary); text-decoration:none; font-weight:600; font-size:.9rem; }
.ax-nav a:hover { color: var(--text-primary); }

/* hero */
.ax-h1 { font-size: clamp(2rem, 4.6vw, 3.1rem) !important; line-height:1.1 !important;
         font-weight:800 !important; margin:8px 0 18px 0 !important; max-width: 14em;
         color: var(--text-primary) !important; }
.ax-lead { font-size:1.05rem; line-height:1.7; color: var(--text-secondary);
           max-width: 34em; margin: 0 0 20px 0; }
.ax-checks { display:flex; flex-wrap:wrap; gap:8px 22px; margin: 0 0 26px 0; padding:0; list-style:none;
             color: var(--text-primary); font-size:.92rem; font-weight:500; }
.ax-checks li { display:flex; align-items:center; gap:8px; }
.ax-checks svg { width:18px; height:18px; color: var(--signal-buy); }

/* live feed (the one memorable element) */
.ax-feed { background: rgba(255,255,255,.03); border:1px solid var(--border-glass);
           border-radius: var(--radius-lg); padding: 6px 16px 10px; }
.ax-feed-head { display:flex; justify-content:space-between; align-items:center;
                padding: 12px 0; font-size:.85rem; color: var(--text-secondary);
                border-bottom:1px solid var(--border-glass); }
.ax-live { display:flex; align-items:center; gap:8px; color: var(--signal-buy); }
.ax-live i { width:8px; height:8px; border-radius:50%; background: var(--signal-buy);
             animation: ax-pulse 2s ease-in-out infinite; }
@keyframes ax-pulse { 0%,100% { opacity:1; } 50% { opacity:.3; } }
.ax-row { display:grid; grid-template-columns: 52px 1fr auto 96px 76px; gap:12px; align-items:center;
          padding: 12px 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:.9rem; }
.ax-row:last-child { border-bottom:0; }
.ax-side { font-weight:700; font-size:.75rem; padding:3px 0; border-radius:6px; text-align:center; }
.ax-side.buy  { color: var(--signal-buy);  background: var(--signal-buy-bg); }
.ax-side.sell { color: var(--signal-sell); background: var(--signal-sell-bg); }
.ax-sym { font-weight:700; color: var(--text-primary); }
.ax-price { font-family: var(--font-mono); color: var(--text-secondary); font-size:.85rem; }
.ax-chip { font-size:.78rem; font-weight:600; text-align:right; }
.ax-chip.win  { color: var(--signal-buy); }
.ax-chip.loss { color: var(--signal-sell); }
.ax-chip.open { color: var(--accent-cyan); }
.ax-when { font-size:.78rem; color: var(--text-muted); text-align:right; }
.ax-feed-empty { padding: 28px 4px; color: var(--text-secondary); font-size:.92rem; line-height:1.6; }

/* sections */
.ax-sec-title { font-size:1.7rem !important; margin: 0 0 6px 0 !important; }
.ax-sec-sub { color: var(--text-secondary); margin: 0 0 24px 0; max-width: 40em; line-height:1.6; }
.ax-spacer { height: 72px; }

.ax-steps { display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; padding:0; margin:0; list-style:none; counter-reset: s; }
.ax-steps li { counter-increment: s; border-top:2px solid var(--border-glow); padding-top:16px; }
.ax-steps li::before { content: counter(s); font-family: var(--font-mono); color: var(--accent-cyan);
                       font-weight:700; font-size:.95rem; display:block; margin-bottom:8px; }
.ax-steps h3 { margin:0 0 6px 0 !important; font-size:1.05rem !important; }
.ax-steps p { margin:0; color: var(--text-secondary); font-size:.92rem; line-height:1.6; }

.ax-card { background: var(--bg-card); border:1px solid var(--border-glass);
           border-radius: var(--radius-lg); padding: 24px 22px; height:100%; }
.ax-card .ax-ico { color: var(--accent-cyan); margin-bottom:14px; display:block; }
.ax-card h3 { margin:0 0 8px 0 !important; font-size:1.05rem !important; }
.ax-card p  { margin:0; color: var(--text-secondary); font-size:.92rem; line-height:1.6; }

.ax-plan { background: var(--bg-card); border:1px solid var(--border-glass);
           border-radius: var(--radius-lg); padding: 24px 22px; min-height: 250px; margin-bottom:12px; }
.ax-plan.pro { border:1.5px solid rgba(0,229,255,.55); background: linear-gradient(180deg, rgba(0,229,255,.08), rgba(10,15,30,.9)); }
.ax-plan h3 { margin:0 0 6px 0 !important; font-size:1.05rem !important; display:flex; justify-content:space-between; align-items:center; }
.ax-badge { font-size:.7rem; font-weight:700; color: var(--accent-cyan); border:1px solid var(--accent-cyan);
            border-radius:999px; padding:2px 10px; }
.ax-price-tag { font-family: var(--font-mono); font-size:2rem; font-weight:700; color: var(--text-primary); margin: 4px 0 14px 0; }
.ax-price-tag small { font-family: var(--font-ui); font-size:.85rem; font-weight:400; color: var(--text-secondary); }
.ax-plan ul { margin:0; padding-left:18px; color: var(--text-secondary); font-size:.9rem; line-height:1.9; }

.ax-foot { color: var(--text-muted); font-size:.8rem; line-height:1.7; max-width: 52em; padding-bottom: 24px; }

@media (max-width: 760px) {
    .ax-nav { display:none; }
    .ax-steps { grid-template-columns: 1fr; }
    .ax-row { grid-template-columns: 48px 1fr auto 86px; }
    .ax-when { display:none; }
    .ax-spacer { height: 48px; }
}
@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .ax-live i { animation: none; }
}
</style>
"""


def _render_login():
    with st.container(border=True):
        st.markdown("#### Welcome back")
        with st.form("form_login", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@example.com", key="f_login_email")
            pw = st.text_input("Password", type="password", key="f_login_pw")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)

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
        st.markdown("#### Start your 14-day free trial")
        st.caption("No credit card needed.")
        with st.form("form_signup", clear_on_submit=False):
            name = st.text_input("Full name", placeholder="John Smith", key="f_su_name")
            email = st.text_input("Email", placeholder="you@example.com", key="f_su_email")
            pw = st.text_input(
                "Password", type="password", key="f_su_pw",
                help=f"At least {MIN_PASSWORD_LEN} characters. Use the eye icon to check what you typed.",
            )
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        st.caption("Trading involves risk. Signals can lose money.")

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

    # ── Nav ──────────────────────────────────────────────────────────────────
    n1, n2 = st.columns([2, 3])
    with n1:
        st.markdown(get_logo_html("left", is_clickable=True), unsafe_allow_html=True)
    with n2:
        st.markdown(
            _html("""
            <nav class="ax-nav" aria-label="Page sections">
              <a href="#how-it-works">How it works</a>
              <a href="#pricing">Pricing</a>
              <a href="#faq">FAQ</a>
              <a href="#contact">Contact</a>
            </nav>"""),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            _html(f"""
            <h1 class="ax-h1">AI trading signals, copied to your MT5 account</h1>
            <p class="ax-lead">Specialist models scan 31 forex pairs around the clock. When one finds a
            setup, the trade is placed on your account and the entry, stop-loss and take-profit
            arrive in Telegram.</p>
            <ul class="ax-checks">
              <li>{_icon("check")}14 days free</li>
              <li>{_icon("check")}No credit card</li>
              <li>{_icon("check")}Works with MetaTrader 5</li>
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

    st.markdown('<div class="ax-spacer" id="how-it-works"></div>', unsafe_allow_html=True)

    # ── How it works (a real sequence) ───────────────────────────────────────
    st.markdown(
        _html("""
        <h2 class="ax-sec-title">How it works</h2>
        <p class="ax-sec-sub">From sign-up to your first copied trade.</p>
        <ol class="ax-steps">
          <li><h3>Create your account</h3>
              <p>Sign up with your email. The 14-day trial starts immediately and needs no card.</p></li>
          <li><h3>Connect MetaTrader 5</h3>
              <p>Add your MT5 login in the Copy Trading Hub and set how much to risk per trade.</p></li>
          <li><h3>Signals are copied for you</h3>
              <p>New setups are placed on your account automatically, and each one is sent to Telegram.</p></li>
        </ol>"""),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ax-spacer"></div>', unsafe_allow_html=True)

    # ── Features ─────────────────────────────────────────────────────────────
    st.markdown('<h2 class="ax-sec-title" id="features">What you get</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Built for traders who want signals executed, not just listed.</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    cards = [
        (f1, "brain", "A model for every pair",
         "Each of the 31 pairs has its own specialist model, retrained as volatility and trend conditions change."),
        (f2, "copy", "Automatic copy trading",
         "Trades are mirrored to your MT5 account with lot size and risk calculated for you."),
        (f3, "bell", "Telegram alerts",
         "Every signal includes the entry price, stop-loss and take-profit levels."),
    ]
    for col, icon, title, body in cards:
        with col:
            st.markdown(
                _html(f'<div class="ax-card"><span class="ax-ico">{_icon(icon)}</span>'
                      f"<h3>{title}</h3><p>{body}</p></div>"),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="ax-spacer" id="pricing"></div>', unsafe_allow_html=True)

    # ── Pricing ──────────────────────────────────────────────────────────────
    st.markdown('<h2 class="ax-sec-title">Pricing</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Start free. Upgrade when you are ready.</p>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.markdown(
            _html("""<div class="ax-plan"><h3>Free trial</h3>
            <div class="ax-price-tag">$0<small> for 14 days</small></div>
            <ul><li>Full copy trading</li><li>All 31 pair models</li><li>No credit card required</li></ul></div>"""),
            unsafe_allow_html=True,
        )
        st.button("Start free trial", type="primary", use_container_width=True, key="p_btn_trial",
                  on_click=_go, kwargs={"auth_mode": AUTH_SIGNUP, "scroll": "top"})
    with p2:
        st.markdown(
            _html("""<div class="ax-plan pro"><h3>Pro Trader<span class="ax-badge">Recommended</span></h3>
            <div class="ax-price-tag">$49<small> per month</small></div>
            <ul><li>All 31 pairs, scanned and copied</li><li>Instant Telegram alerts</li>
            <li>Automatic lot and risk sizing</li></ul></div>"""),
            unsafe_allow_html=True,
        )
        st.button("Get Pro Trader", type="primary", use_container_width=True, key="p_btn_pro",
                  on_click=_go, kwargs={"auth_mode": AUTH_SIGNUP, "scroll": "top"})
    with p3:
        st.markdown(
            _html("""<div class="ax-plan"><h3>Institutional</h3>
            <div class="ax-price-tag">$199<small> per month</small></div>
            <ul><li>Dedicated VPS</li><li>Direct support</li><li>Multi-account management</li></ul></div>"""),
            unsafe_allow_html=True,
        )
        st.button("Talk to us", use_container_width=True, key="p_btn_inst",
                  on_click=_go, kwargs={"topic": "Institutional plan", "scroll": "contact"})

    st.markdown('<div class="ax-spacer" id="faq"></div>', unsafe_allow_html=True)

    # ── FAQ ──────────────────────────────────────────────────────────────────
    st.markdown('<h2 class="ax-sec-title">Questions</h2>', unsafe_allow_html=True)
    with st.expander("What do I need to get started?"):
        st.write("An email address to sign up, and a MetaTrader 5 account to copy trades to. "
                 "You can add your MT5 details after you create your account.")
    with st.expander("Do I need a credit card for the trial?"):
        st.write("No. The 14-day trial includes full copy trading and all 31 pair models.")
    with st.expander("Do the signals guarantee profit?"):
        st.write("No. No trading system can. Some signals will hit their stop-loss, and the live feed "
                 "above shows those alongside the winners. Only risk money you can afford to lose.")

    st.markdown('<div class="ax-spacer" id="contact"></div>', unsafe_allow_html=True)

    # ── Contact ──────────────────────────────────────────────────────────────
    st.markdown('<h2 class="ax-sec-title">Contact us</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ax-sec-sub">Questions about plans, your broker or a team setup? We reply within 24 hours.</p>', unsafe_allow_html=True)
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
                    placeholder="Tell us about your setup: broker, account size, number of accounts.",
                )
            sent = st.form_submit_button("Send message", type="primary", use_container_width=True)

        if sent:
            if not (inq_name.strip() and inq_email.strip() and inq_msg.strip()):
                st.error("Fill in your name, email and message.")
            else:
                from core.leads import submit_inquiry
                submit_inquiry(inq_name, inq_email, f"[{topic}] {inq_msg}")
                st.success("Message sent. We'll reply within 24 hours.")

    st.markdown('<div class="ax-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        _html("""<p class="ax-foot"><strong>Risk warning:</strong> Trading foreign exchange and leveraged
        products carries a high level of risk and may not be suitable for everyone. You can lose more than
        you invest. Past performance does not guarantee future results. ApexForex provides software and
        signals, not financial advice.<br>&copy; 2026 ApexForex. All rights reserved.</p>"""),
        unsafe_allow_html=True,
    )
