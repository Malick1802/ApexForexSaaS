"""User Profile & Subscription Management — dynamic user profile and plan management."""
import streamlit as st
# ── Auth Guard ─────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Session expired or not logged in. Please log in again.")
    st.page_link("app.py", label="🔑 Go to Login", icon="🔑")
    st.stop()
# ── End Auth Guard ─────────────────────────────────────────────────────────
import sys
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import hero_banner, section_header, kpi_card, inject_css
inject_css()

from core.auth import (
    get_portal_user_by_email, update_portal_user,
    change_password, verify_login,
)
from core.user_accounts import (
    get_user_by_email, update_user, subscription_label,
    is_subscription_active, TRIAL_DAYS,
)

# ── Load Authenticated User Details ──────────────────────────────────────────
user_email = st.session_state.get("user_email", "")
user_name  = st.session_state.get("user_name", "Trader")
user_role  = st.session_state.get("user_role", "subscriber")
is_admin   = (user_role == "admin")

portal_user = get_portal_user_by_email(user_email)
sub_acc     = get_user_by_email(user_email)

hero_banner("User Profile & Subscription", "Manage your personal account, security credentials, and active subscription plan")

# ── Compute Subscription Metrics ─────────────────────────────────────────────
sub_status = "trial"
sub_badge = "🆓 14-Day Free Trial"
sub_color = "#00E5FF"
days_left = 14
trial_pct = 0.0
member_since = "Recent"

if portal_user and portal_user.get("created_at"):
    try:
        dt = datetime.fromisoformat(portal_user["created_at"])
        member_since = dt.strftime("%b %d, %Y")
    except Exception:
        member_since = "Active"

if is_admin:
    sub_badge = "👑 Platform Administrator"
    sub_color = "#FFD700"
    sub_status = "admin"
elif sub_acc:
    sub_status = sub_acc.get("subscription_status", "trial")
    sub_lbl, sub_col = subscription_label(sub_acc)
    sub_badge = sub_lbl
    sub_color = sub_col
    
    if sub_status == "trial" and sub_acc.get("trial_ends_at"):
        try:
            ends = datetime.fromisoformat(sub_acc["trial_ends_at"])
            if ends.tzinfo is None:
                ends = ends.replace(tzinfo=timezone.utc)
            rem = (ends - datetime.now(timezone.utc)).total_seconds() / 86400.0
            days_left = max(0, int(round(rem)))
            trial_pct = max(0.0, min(1.0, (TRIAL_DAYS - days_left) / float(TRIAL_DAYS)))
        except Exception:
            days_left = 14
            trial_pct = 0.1
    elif sub_status == "paid":
        sub_badge = "✅ Pro Trader Member"
        sub_color = "#00FF88"

# ── Two-Column Layout ─────────────────────────────────────────────────────────
col_card, col_main = st.columns([1, 2.2], gap="large")

with col_card:
    avatar_name = urllib.parse.quote(user_name or "Trader")
    avatar_url = f"https://ui-avatars.com/api/?name={avatar_name}&background=0a1628&color=00E5FF&size=200&bold=true&font-size=0.4"

    st.markdown(f"""
    <div class="glass-card" style="text-align: center; padding: 32px 20px;">
        <img style="width:110px;height:110px;border-radius:50%;border:3px solid var(--accent-cyan);box-shadow:0 0 24px rgba(0,229,255,0.25);"
             src="{avatar_url}" />
        <div style="margin-top: 16px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary);">{user_name}</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px;">{user_email}</div>
            <div style="display:inline-flex;align-items:center;gap:6px;padding:5px 14px;background:rgba(255,255,255,0.04);border:1px solid {sub_color}55;border-radius:20px;font-size:0.75rem;font-weight:700;color:{sub_color};margin-top:12px;">
                {sub_badge}
            </div>
            <div style="display:flex;justify-content:center;gap:12px;margin-top:16px;font-size:0.78rem;color:var(--text-muted);border-top:1px solid rgba(255,255,255,0.06);padding-top:14px;">
                <span>Member Since: <strong style="color:var(--text-primary);">{member_since}</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick link to Copy Trading Hub
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("##### ⚡ Quick Navigation")
        if st.button("🔁 Copy Trading Hub", use_container_width=True):
            st.session_state["nav_target"] = "copy_trading"
            st.rerun()
        if is_admin:
            if st.button("🛠️ Admin Panel", use_container_width=True):
                st.session_state["nav_target"] = "admin"
                st.rerun()

with col_main:
    tab_profile, tab_sub = st.tabs(["👤 Profile & Security", "💳 Plan & Subscription"])

    # ── Tab 1: Profile & Security ─────────────────────────────────────────────
    with tab_profile:
        section_header("👤", "Personal Information")
        with st.container(border=True):
            with st.form("form_update_profile"):
                new_name = st.text_input("Full Display Name", value=user_name, placeholder="John Smith")
                st.text_input("Account Email", value=user_email, disabled=True, help="Your email address is your unique login credential and cannot be changed.")
                btn_save_profile = st.form_submit_button("💾 Save Profile Changes", type="primary", use_container_width=True)

            if btn_save_profile:
                if not new_name.strip():
                    st.error("Name cannot be empty.")
                else:
                    try:
                        update_portal_user(user_email, new_name.strip())
                        if sub_acc:
                            update_user(sub_acc["id"], name=new_name.strip())
                        st.session_state["user_name"] = new_name.strip()
                        st.toast("Profile updated successfully!", icon="✅")
                        st.success("✅ Your profile information has been saved.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update profile: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("🔒", "Security & Password")
        with st.container(border=True):
            with st.form("form_change_pw"):
                st.markdown("##### Change Account Password")
                st.caption("Ensure your account uses a strong, unique password with at least 6 characters.")
                curr_pw = st.text_input("Current Password", type="password", key="pw_curr")
                c_np1, c_np2 = st.columns(2)
                with c_np1:
                    new_pw1 = st.text_input("New Password", type="password", key="pw_new1")
                with c_np2:
                    new_pw2 = st.text_input("Confirm New Password", type="password", key="pw_new2")
                
                btn_pw = st.form_submit_button("🔑 Update Password", use_container_width=True)

            if btn_pw:
                if not curr_pw or not new_pw1 or not new_pw2:
                    st.error("Please fill in all password fields.")
                elif new_pw1 != new_pw2:
                    st.error("New passwords do not match.")
                elif len(new_pw1) < 6:
                    st.error("New password must be at least 6 characters long.")
                else:
                    # Verify current password
                    valid = verify_login(user_email, curr_pw)
                    if not valid:
                        st.error("❌ Current password is incorrect.")
                    else:
                        try:
                            change_password(user_email, new_pw1)
                            st.toast("Password updated successfully!", icon="✅")
                            st.success("✅ Password updated successfully. Your new credentials are active.")
                        except Exception as e:
                            st.error(f"Failed to update password: {e}")

    # ── Tab 2: Plan & Subscription ────────────────────────────────────────────
    with tab_sub:
        section_header("💳", "Active Subscription Status")

        if is_admin:
            st.markdown(kpi_card("Current Tier", "Platform Administrator", "Master Access · Unlimited Accounts", "accent-gold"), unsafe_allow_html=True)
            st.info("As an administrator, your account has full root privileges across all trading engines, master broker terminals, and subscriber fleet management.")
        else:
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(kpi_card("Current Plan", "Pro Trader" if sub_status == "paid" else "14-Day Free Trial", sub_badge, "accent-gold" if sub_status == "paid" else "accent-cyan"), unsafe_allow_html=True)
            with k2:
                status_text = "Active Auto-Renewal" if sub_status == "paid" else f"{days_left} Days Remaining"
                st.markdown(kpi_card("Access Status", "Active" if is_subscription_active(sub_acc or {}) else "Inactive", status_text, "accent-green" if is_subscription_active(sub_acc or {}) else "accent-red"), unsafe_allow_html=True)
            with k3:
                st.markdown(kpi_card("Assets Included", "31 Forex & CFDs", "Full MT5 Mirroring", "accent-cyan"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if sub_status == "trial":
                with st.container(border=True):
                    st.markdown("##### ⏳ Trial Timeline")
                    st.caption(f"Your free 14-day copy trading trial has **{days_left} days** remaining.")
                    st.progress(trial_pct, text=f"Day {max(1, TRIAL_DAYS - days_left)} of {TRIAL_DAYS}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Upgrade Card
                with st.container(border=True):
                    st.markdown("""
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:1.15rem; font-weight:800; color:var(--text-primary);">
                                Upgrade to Pro Trader
                            </div>
                            <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:4px;">
                                Keep automated copying running without interruption after your trial concludes.
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.6rem; font-weight:900; color:var(--accent-cyan);">$49</span>
                            <span style="font-size:0.8rem; color:var(--text-secondary);">/month</span>
                        </div>
                    </div>
                    <div style="margin: 16px 0; border-top:1px solid rgba(255,255,255,0.06); padding-top:12px;">
                        <ul style="font-size:0.85rem; color:var(--text-secondary); margin:0; padding-left:18px; line-height:1.8;">
                            <li>Continuous 24/7 MT5 signal execution on any broker</li>
                            <li>Real-time instant Telegram alerts with entry, SL & TP</li>
                            <li>Commodities coverage (Gold & Crude Oil auto-adaptation)</li>
                            <li>Dedicated risk sizing & prop firm drawdown protection</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_upgrade = st.button("🚀 Request Pro Trader Upgrade ($49/mo)", type="primary", use_container_width=True, key="btn_upgrade_pro")
                    if btn_upgrade:
                        try:
                            from core.leads import submit_upgrade_request
                            req_id = submit_upgrade_request(
                                name=user_name,
                                email=user_email,
                                requested_tier="Pro Trader ($49/mo)",
                                user_id=sub_acc["id"] if sub_acc else None
                            )
                            st.session_state["upgrade_requested"] = True
                            st.toast("Upgrade request dispatched to admin!", icon="🚀")
                        except Exception as e:
                            st.error(f"Failed to submit upgrade request: {e}")

                    if st.session_state.get("upgrade_requested"):
                        st.markdown("""
                        <div style="background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);border-radius:10px;padding:12px 16px;margin-top:12px;font-size:0.85rem;color:#00FF88;">
                            ✅ <b>Upgrade Request Submitted!</b> Your request has been queued for immediate activation. You will receive an invoice and confirmation on your account within minutes.
                        </div>
                        """, unsafe_allow_html=True)
            elif sub_status == "paid":
                with st.container(border=True):
                    st.markdown("##### 🛡️ Subscription Management")
                    st.success("✅ Your Pro Trader subscription is currently active. Trades execute continuously 24/7.")
                    if sub_acc and sub_acc.get("paid_until"):
                        st.caption(f"Paid coverage active until: **{sub_acc.get('paid_until')}**")
                    if sub_acc and sub_acc.get("paid_note"):
                        st.caption(f"Reference: `{sub_acc.get('paid_note')}`")
