import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import inject_css, get_logo_html

def show_landing():
    inject_css()

    # Hide sidebar on landing page and enable smooth anchor scrolling
    st.markdown("""
    <style>
    html { scroll-behavior: smooth; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }
    .auth-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 32px 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    }
    .landing-card {
        background: rgba(13, 19, 36, 0.72);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 22px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .landing-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 229, 255, 0.1);
    }
    .pricing-card-pro {
        background: linear-gradient(180deg, rgba(0, 229, 255, 0.1) 0%, rgba(10, 15, 30, 0.95) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1.5px solid rgba(0, 229, 255, 0.55);
        border-radius: 18px;
        padding: 24px 22px;
        box-shadow: 0 0 35px rgba(0, 229, 255, 0.16), 0 14px 40px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Nav Header ───────────────────────────────────────────────────────────
    n1, n2 = st.columns([2, 3])
    with n1:
        st.markdown(get_logo_html("left", is_clickable=True), unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div style="text-align:right; padding-top:18px;">
            <a href="#features" style="margin-right:24px;color:var(--text-secondary);text-decoration:none;font-weight:600;font-size:0.9rem;">Features</a>
            <a href="#pricing"  style="margin-right:24px;color:var(--text-secondary);text-decoration:none;font-weight:600;font-size:0.9rem;">Pricing</a>
            <a href="#contact"  style="color:var(--text-secondary);text-decoration:none;font-weight:600;font-size:0.9rem;">Contact</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:0 0 28px 0;'>", unsafe_allow_html=True)

    # ── HERO + AUTH two-column layout ────────────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("""
        <h1 style="font-size:2.9rem;line-height:1.15;margin-bottom:0;
                   background:linear-gradient(90deg,#FFFFFF,#8b95a8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Institutional-Grade
        </h1>
        <h1 style="font-size:2.9rem;line-height:1.15;margin-top:0;
                   color:var(--accent-cyan);-webkit-text-fill-color:var(--accent-cyan);">
            AI Trading Intelligence
        </h1>
        <p style="font-size:1.05rem;color:var(--text-secondary);margin-top:14px;max-width:520px;line-height:1.7;">
            Stop guessing. Start executing. Our AI models analyze 31 forex pairs in real-time
            and copy every signal directly to your MT5 terminal — automatically.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Live signal preview
        signal_html = ""
        try:
            from theme import get_db
            db = get_db()
            recent_signals = db.get_recent_signals(limit=3)
            for s in recent_signals:
                if s.get("signal") == "WAIT":
                    continue
                outcome = s.get("outcome", "ACTIVE")
                color = "var(--accent-green)" if outcome == "SUCCESS" else "var(--accent-red)" if outcome == "FAIL" else "var(--accent-cyan)"
                bg = "rgba(0,255,136,0.05)" if outcome == "SUCCESS" else "rgba(255,68,102,0.05)" if outcome == "FAIL" else "rgba(0,229,255,0.05)"
                status_text = "SUCCESS" if outcome == "SUCCESS" else "FAILED" if outcome == "FAIL" else "LIVE"
                price = s.get("price_at_signal") or 0.0
                signal_html += (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'background:{bg};padding:12px 16px;border-radius:12px;margin-bottom:10px;border-left:3px solid {color};">'
                    f'<div><div style="font-weight:700;color:#FFF;">{s["symbol"]}</div>'
                    f'<div style="font-size:0.8rem;color:var(--text-secondary);">{s["signal"]} @ {price:.5f}</div></div>'
                    f'<div style="text-align:right;"><div style="color:{color};font-weight:700;">{status_text}</div>'
                    f'<div style="font-size:0.7rem;color:var(--text-secondary);">Recent</div></div></div>'
                )
        except Exception:
            pass

        if not signal_html:
            signal_html = '<div style="text-align:center;color:var(--text-muted);padding:28px;">Monitoring live institutional flows...</div>'

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);'
            f'border-radius:16px;padding:18px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:14px;font-size:0.85rem;">'
            f'<span style="color:var(--text-secondary);">Live Intelligence Feed</span>'
            f'<span style="color:var(--accent-green);">&#9679; Active Scan</span></div>'
            f'{signal_html}</div>',
            unsafe_allow_html=True,
        )

    # ── AUTH FORM ────────────────────────────────────────────────────────────
    # NOTE: We avoid st.tabs() here because st.rerun() called inside a tab
    # inside a column can silently fail in Streamlit 1.40+. We use st.form
    # which guarantees atomic state commits before the submit handler fires.
    with right:
        auth_mode = st.radio(
            "auth_mode_selector",
            ["🔑  Log In", "🚀  Sign Up"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )
        st.markdown("")

        # ── LOG IN ───────────────────────────────────────────────────────────
        if auth_mode == "🔑  Log In":
            with st.container(border=True):
                st.markdown("#### Welcome back")
                st.caption("Sign in to your ApexForex account")
                st.markdown("")
                with st.form("form_login", clear_on_submit=False):
                    login_email = st.text_input("Email address", placeholder="you@example.com", key="f_login_email")
                    login_pw    = st.text_input("Password", type="password", key="f_login_pw")
                    st.markdown("")
                    submitted = st.form_submit_button("🔑  Log In", type="primary", use_container_width=True)

                if submitted:
                    if not login_email or not login_pw:
                        st.error("Please enter your email and password.")
                    else:
                        from core.auth import verify_login
                        from core.sessions import create_session
                        user = verify_login(login_email.strip(), login_pw)
                        if user:
                            token = create_session(user)
                            st.session_state["_pending_auth"] = {
                                "email": user["email"],
                                "name":  user["name"],
                                "role":  user["role"],
                                "id":    user["id"],
                                "token": token,
                            }
                            st.rerun()
                        else:
                            st.error("❌ Incorrect email or password.")

        # ── SIGN UP ──────────────────────────────────────────────────────────
        else:
            with st.container(border=True):
                st.markdown("#### Create your account")
                st.caption("14-day free copy trading trial — no credit card needed")
                st.markdown("")
                with st.form("form_signup", clear_on_submit=False):
                    su_name  = st.text_input("Full Name",        placeholder="John Smith",      key="f_su_name")
                    su_email = st.text_input("Email address",    placeholder="you@example.com", key="f_su_email")
                    su_pw    = st.text_input("Password",         type="password",               key="f_su_pw")
                    su_pw2   = st.text_input("Confirm Password", type="password",               key="f_su_pw2")
                    st.markdown("")
                    submitted_su = st.form_submit_button("🚀  Create Account", type="primary", use_container_width=True)

                if submitted_su:
                    if not all([su_name, su_email, su_pw, su_pw2]):
                        st.error("Please fill in all fields.")
                    elif su_pw != su_pw2:
                        st.error("Passwords do not match.")
                    elif len(su_pw) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        from core.auth import register_portal_user, get_portal_user_by_email
                        from core.user_accounts import get_user_by_email, add_user
                        from core.sessions import create_session

                        if get_portal_user_by_email(su_email):
                            st.error("An account with this email already exists. Please log in.")
                        else:
                            new_user = register_portal_user(su_name, su_email, su_pw)
                            if new_user:
                                if not get_user_by_email(su_email):
                                    add_user(
                                        name=su_name, email=su_email,
                                        mt5_login="", mt5_password="", mt5_server="",
                                    )
                                token = create_session(new_user)
                                st.session_state["_pending_auth"] = {
                                    "email": new_user["email"],
                                    "name":  new_user["name"],
                                    "role":  "subscriber",
                                    "id":    new_user["id"],
                                    "token": token,
                                }
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("Registration failed. Please try again.")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # ── FEATURES ─────────────────────────────────────────────────────────────
    st.markdown('<div id="features"></div>', unsafe_allow_html=True)
    st.markdown("## 🧠 Key Platform Features")
    st.markdown("<p style='color:var(--text-secondary);margin-bottom:22px;'>Designed to scale from retail traders to fund operations</p>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="landing-card">
            <div style="font-size:2.2rem;margin-bottom:12px;">🧠</div>
            <h3 style="margin:0 0 8px 0;color:var(--text-primary);">Adaptive AI Models</h3>
            <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.6;margin:0;">
                Specialist models retrain continuously, adapting to volatility regimes and structural trend shifts across 31 global pairs.
            </p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="landing-card">
            <div style="font-size:2.2rem;margin-bottom:12px;">🤖</div>
            <h3 style="margin:0 0 8px 0;color:var(--text-primary);">Fully Automated</h3>
            <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.6;margin:0;">
                Register once — institutional trades mirror 24/7 on your MT5 terminal with millisecond latency even when your PC is offline.
            </p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="landing-card">
            <div style="font-size:2.2rem;margin-bottom:12px;">📲</div>
            <h3 style="margin:0 0 8px 0;color:var(--text-primary);">Instant Alerts</h3>
            <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.6;margin:0;">
                Real-time Telegram notifications with precise entry targets, calibrated stop-loss, and multi-tier take-profit levels.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # ── PRICING ───────────────────────────────────────────────────────────────
    st.markdown('<div id="pricing"></div>', unsafe_allow_html=True)
    st.markdown("## 💳 Flexible Trading Plans")
    st.markdown("<p style='color:var(--text-secondary);margin-bottom:22px;'>Start free — upgrade when you are ready</p>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="landing-card" style="min-height:210px;margin-bottom:10px;">
            <h3 style="margin-top:0;color:var(--text-primary);">Free Trial</h3>
            <div style="font-size:2.2rem;font-weight:800;color:var(--accent-cyan);margin-bottom:12px;font-family:var(--font-mono);">$0<span style="font-size:0.85rem;font-weight:400;color:var(--text-secondary);font-family:var(--font-ui);"> / 14 days</span></div>
            <ul style="color:var(--text-secondary);font-size:0.85rem;padding-left:18px;line-height:1.8;margin:0;">
                <li>Full copy trading execution</li>
                <li>All 31 asset models live</li>
                <li>No credit card required</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        if st.button("🚀 Start Free Trial", type="primary", use_container_width=True, key="p_btn_trial"):
            st.session_state["auth_mode"] = "🚀  Sign Up"
            st.rerun()
    with p2:
        st.markdown("""
        <div class="pricing-card-pro" style="min-height:210px;margin-bottom:10px;">
            <div style="float:right;padding:3px 10px;background:rgba(0,229,255,0.15);border:1px solid #00E5FF;border-radius:12px;font-size:0.65rem;font-weight:800;color:#00E5FF;text-transform:uppercase;letter-spacing:0.08em;box-shadow:0 0 12px rgba(0,229,255,0.3);">⚡ Popular</div>
            <h3 style="margin-top:0;color:#00E5FF;">Pro Trader</h3>
            <div style="font-size:2.2rem;font-weight:800;color:var(--accent-cyan);margin-bottom:12px;font-family:var(--font-mono);">$49<span style="font-size:0.85rem;font-weight:400;color:var(--text-secondary);font-family:var(--font-ui);"> / month</span></div>
            <ul style="color:var(--text-secondary);font-size:0.85rem;padding-left:18px;line-height:1.8;margin:0;">
                <li>All 31 pairs AI scan & copy</li>
                <li>Instant Telegram alerts</li>
                <li>Automated lot & risk sizing</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        if st.button("⚡ Get Pro Access", type="primary", use_container_width=True, key="p_btn_pro"):
            st.session_state["auth_mode"] = "🚀  Sign Up"
            st.rerun()
    with p3:
        st.markdown("""
        <div class="landing-card" style="min-height:210px;margin-bottom:10px;">
            <h3 style="margin-top:0;color:var(--text-primary);">Institutional</h3>
            <div style="font-size:2.2rem;font-weight:800;color:var(--accent-cyan);margin-bottom:12px;font-family:var(--font-mono);">$199<span style="font-size:0.85rem;font-weight:400;color:var(--text-secondary);font-family:var(--font-ui);"> / month</span></div>
            <ul style="color:var(--text-secondary);font-size:0.85rem;padding-left:18px;line-height:1.8;margin:0;">
                <li>Dedicated ultra-low latency VPS</li>
                <li>Direct quant desk support</li>
                <li>Multi-account fleet orchestration</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        if st.button("💼 Contact Enterprise Desk", use_container_width=True, key="p_btn_inst"):
            st.session_state["c_msg_prefill"] = "I am interested in the Institutional Tier ($199/mo) with dedicated VPS hosting."
            st.toast("Institutional inquiry pre-filled! Please complete your message below.", icon="💼")
            st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # ── CONTACT ───────────────────────────────────────────────────────────────
    st.markdown('<div id="contact"></div>', unsafe_allow_html=True)
    section_header_contact = "## 📬 Get In Touch"
    st.markdown(section_header_contact)
    with st.container(border=True):
        with st.form("form_contact_lead"):
            cc1, cc2 = st.columns(2)
            with cc1:
                inq_name = st.text_input("Your Name", placeholder="John Smith", key="c_name")
                inq_email = st.text_input("Email", placeholder="you@example.com", key="c_email")
            with cc2:
                prefill = st.session_state.get("c_msg_prefill", "")
                inq_msg = st.text_area("Message", value=prefill, placeholder="Tell us about your trading setup, prop firm, or broker...", height=112, key="c_msg_input")
            btn_contact = st.form_submit_button("✉️ Send Message", type="primary", use_container_width=True)

        if btn_contact:
            if not inq_name.strip() or not inq_email.strip() or not inq_msg.strip():
                st.error("Please fill in your name, email, and message.")
            else:
                from core.leads import submit_inquiry
                submit_inquiry(inq_name, inq_email, inq_msg)
                st.success("✅ Thank you! Your message has been received. Our team will reach out within 24 hours.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:var(--text-secondary);opacity:0.6;padding-bottom:24px;">
        <p style="text-transform:uppercase;letter-spacing:0.2em;font-size:0.75rem;margin-bottom:16px;">Trusted Technologies</p>
        <div style="display:flex;justify-content:center;gap:40px;font-weight:700;font-size:1.1rem;filter:grayscale(100%);">
            <span>METATRADER 5</span><span>PYTHON</span><span>TENSORFLOW</span><span>OPENAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
