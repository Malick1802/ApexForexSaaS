"""Admin Panel -- Visible only to admin role accounts."""
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
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import hero_banner, section_header, inject_css
inject_css()

from core.user_accounts import (
    get_all_users, update_user, delete_user,
    mark_paid, extend_trial,
    subscription_label, is_subscription_active,
)
from core.auth import get_all_portal_users, delete_portal_user
from core.leads import (
    get_pending_upgrade_requests, resolve_upgrade_request,
    get_all_inquiries, update_inquiry_status, delete_inquiry,
)

# ── Admin-only guard ──────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.error("Please log in.")
    st.stop()

if st.session_state.get("user_role") != "admin":
    st.error("⛔ Access denied. This page is for admins only.")
    st.stop()

hero_banner(
    "Admin Panel",
    "Manage all subscriber accounts, subscriptions, and copy trading access",
    show_status=False,
)

# ── Load data ─────────────────────────────────────────────────────────────────
users    = get_all_users()          # MT5 / subscription records
portals  = get_all_portal_users()   # Login accounts

# Build a lookup: email -> portal user
portal_map = {p["email"]: p for p in portals}

# ── KPI row ───────────────────────────────────────────────────────────────────
total   = len(users)
trials  = [u for u in users if u.get("subscription_status") == "trial"  and is_subscription_active(u)]
paid    = [u for u in users if u.get("subscription_status") == "paid"   and is_subscription_active(u)]
expired = [u for u in users if not is_subscription_active(u)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Accounts", total)
k2.metric("🆓 Active Trials", len(trials))
k3.metric("✅ Paid Active", len(paid))
k4.metric("⛔ Expired / Paused", len(expired))

st.divider()

# ── Pending Upgrade Requests ──────────────────────────────────────────────────
pending_upgrades = get_pending_upgrade_requests()
if pending_upgrades:
    section_header("⚡", f"Pending Plan Upgrade Requests ({len(pending_upgrades)})")
    for req in pending_upgrades:
        with st.container(border=True):
            rc1, rc2 = st.columns([3.5, 2.5])
            with rc1:
                st.markdown(
                    f"**{req['name']}** &nbsp;·&nbsp; `{req['email']}`<br>"
                    f"<span style='font-size:0.85rem; color:var(--accent-gold); font-weight:700;'>Requested: {req['requested_tier']}</span> "
                    f"&nbsp;·&nbsp; <span style='font-size:0.8rem; color:var(--text-muted);'>Submitted: {req['created_at'][:19]}</span>",
                    unsafe_allow_html=True
                )
            with rc2:
                btn_u1, btn_u2, btn_u3 = st.columns(3)
                target_user = next((u for u in users if u["email"].lower() == req["email"].lower()), None)
                
                with btn_u1:
                    if st.button("✅ Pro 30d", key=f"upg_30d_{req['id']}", use_container_width=True, help="Approve 30-day Pro Trader subscription"):
                        if target_user:
                            paid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
                            mark_paid(target_user["id"], paid_until_iso=paid_until, note=f"Approved: {req['requested_tier']}")
                        resolve_upgrade_request(req["id"], "approved")
                        st.toast(f"✅ Approved Pro Trader for {req['name']}!", icon="💳")
                        st.rerun()
                with btn_u2:
                    if st.button("♾️ Lifetime", key=f"upg_life_{req['id']}", use_container_width=True, help="Approve Lifetime Pro subscription"):
                        if target_user:
                            mark_paid(target_user["id"], paid_until_iso=None, note=f"Approved Lifetime: {req['requested_tier']}")
                        resolve_upgrade_request(req["id"], "approved")
                        st.toast(f"♾️ Granted Lifetime Pro to {req['name']}!", icon="♾️")
                        st.rerun()
                with btn_u3:
                    if st.button("❌ Dismiss", key=f"upg_dism_{req['id']}", use_container_width=True):
                        resolve_upgrade_request(req["id"], "dismissed")
                        st.toast(f"Dismissed request #{req['id']}", icon="❌")
                        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# ── Subscriber table ──────────────────────────────────────────────────────────
section_header("👥", "All Subscribers")

if not users:
    st.info("No subscribers registered yet.")
else:
    for user in users:
        label, color = subscription_label(user)
        active = is_subscription_active(user)
        risk_display = (
            f"{user['risk_value']} lots" if user["risk_type"] == "fixed"
            else f"{user['risk_value']}% account risk"
        )
        portal = portal_map.get(user["email"])
        portal_status = "✅ Has login" if portal else "⚠️ No login account"

        with st.container(border=True):
            hdr_col, btn_col = st.columns([4, 2])

            with hdr_col:
                st.markdown(
                    f"**{user['name']}** &nbsp;·&nbsp; "
                    f"<span style='color: var(--text-muted); font-size:0.85rem;'>{user['email']}</span> "
                    f"&nbsp;<span style='color:{color}; font-weight:700;'>{label}</span> "
                    f"&nbsp;·&nbsp; <span style='font-size:0.8rem; color:var(--text-muted);'>{portal_status}</span>",
                    unsafe_allow_html=True,
                )
                mt5_info = f"`{user['mt5_login']}` on `{user['mt5_server']}`" if user.get("mt5_login") else "_MT5 not configured_"
                tg_info = f"📲 `{user['telegram_chat_id']}`" if user.get("telegram_chat_id") else "📵 No Telegram"
                term_info = f" · Terminal: `{Path(user['terminal_path']).name}`" if user.get("terminal_path") else ""
                arch_tag = "🏢 Prop" if user.get("account_type") == "prop_firm" else ("⚡ Custom" if user.get("account_type") == "custom" else "🏛️ Retail")
                st.caption(
                    f"Arch: **{arch_tag}** · MT5: {mt5_info}{term_info} · Risk: **{risk_display}** · "
                    f"Max daily: **{user['max_daily_trades']}** · "
                    f"Telegram: {tg_info} · "
                    f"Last trade: {user.get('last_trade_at') or 'Never'} · "
                    f"Joined: {user.get('created_at', '')[:10]}"
                )

            with btn_col:
                uid = user["id"]
                new_state = st.toggle(
                    "Copy Active",
                    value=bool(user["enabled"]),
                    key=f"toggle_{uid}",
                )
                if new_state != bool(user["enabled"]):
                    update_user(uid, enabled=int(new_state))
                    st.rerun()

            # ── Action buttons ────────────────────────────────────────────────
            a1, a2, a3, a4, a5 = st.columns(5)

            with a1:
                if st.button("✅ Paid 1 Month", key=f"paid1m_{uid}", use_container_width=True):
                    paid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
                    mark_paid(uid, paid_until_iso=paid_until, note="Admin: 1 month")
                    st.toast(f"✅ {user['name']} paid for 30 days", icon="💳")
                    st.rerun()

            with a2:
                if st.button("✅ Paid 3 Months", key=f"paid3m_{uid}", use_container_width=True):
                    paid_until = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
                    mark_paid(uid, paid_until_iso=paid_until, note="Admin: 3 months")
                    st.toast(f"✅ {user['name']} paid for 90 days", icon="💳")
                    st.rerun()

            with a3:
                if st.button("♾️ Lifetime Paid", key=f"lifetime_{uid}", use_container_width=True):
                    mark_paid(uid, paid_until_iso=None, note="Admin: Lifetime")
                    st.toast(f"♾️ {user['name']} set to Lifetime", icon="♾️")
                    st.rerun()

            with a4:
                if st.button("🕐 Extend Trial 14d", key=f"ext_{uid}", use_container_width=True):
                    extend_trial(uid, extra_days=14)
                    st.toast(f"🕐 {user['name']} trial extended", icon="🕐")
                    st.rerun()

            with a5:
                if st.button("🗑️ Remove", key=f"del_{uid}", use_container_width=True):
                    delete_user(uid)
                    if user.get("email"):
                        delete_portal_user(user["email"])
                    st.toast(f"Removed {user['name']}", icon="🗑️")
                    st.rerun()

st.divider()

# ── Portal accounts with no MT5 record ───────────────────────────────────────
section_header("🔑", "All Portal Login Accounts")

mt5_emails = {u["email"] for u in users}
orphan_portals = [p for p in portals if p["email"] not in mt5_emails]

if not portals:
    st.info("No portal accounts yet.")
else:
    for p in portals:
        has_mt5 = p["email"] in mt5_emails
        role_badge = "🔴 Admin" if p["role"] == "admin" else "👤 Subscriber"
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"**{p['name']}** — `{p['email']}` &nbsp; {role_badge} "
                    f"&nbsp;·&nbsp; MT5 record: {'✅' if has_mt5 else '⚠️ Missing'}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Joined: {p.get('created_at', '')[:10]}")
            with col2:
                if p["role"] != "admin":
                    if st.button("🗑️ Remove Login", key=f"rm_portal_{p['id']}", use_container_width=True):
                        delete_portal_user(p["email"])
                        st.toast(f"Removed portal account for {p['email']}", icon="🗑️")
                        st.rerun()

st.divider()

# ── Inbound Contact Leads ───────────────────────────────────────────────────
inquiries = get_all_inquiries(limit=50)
new_inquiries = [i for i in inquiries if i.get("status") == "new"]
section_header("📬", f"Inbound Contact Leads & Inquiries ({len(inquiries)})")

if not inquiries:
    st.info("No inbound contact messages received yet.")
else:
    for inq in inquiries:
        is_new = (inq.get("status") == "new")
        status_badge = "🟢 New Lead" if is_new else "⚪ Contacted"
        badge_color = "#00FF88" if is_new else "var(--text-muted)"

        with st.container(border=True):
            ic1, ic2 = st.columns([3.8, 1.8])
            with ic1:
                st.markdown(
                    f"**{inq['name']}** &nbsp;·&nbsp; `{inq['email']}` &nbsp;·&nbsp; "
                    f"<span style='color:{badge_color}; font-weight:700; font-size:0.8rem;'>{status_badge}</span><br>"
                    f"<div style='margin-top:6px; font-size:0.88rem; color:var(--text-primary); background:rgba(255,255,255,0.03); padding:10px 14px; border-radius:8px; border-left:3px solid var(--accent-cyan);'>"
                    f"{inq['message']}</div>"
                    f"<span style='font-size:0.75rem; color:var(--text-muted); margin-top:4px; display:inline-block;'>Received: {inq['created_at'][:19]}</span>",
                    unsafe_allow_html=True
                )
            with ic2:
                mailto_url = f"mailto:{inq['email']}?subject=ApexForex%20Inquiry%20Follow-up"
                st.link_button("✉️ Reply via Email", mailto_url, use_container_width=True)
                
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    if is_new:
                        if st.button("✓ Done", key=f"done_inq_{inq['id']}", use_container_width=True, help="Mark lead as contacted"):
                            update_inquiry_status(inq["id"], "contacted")
                            st.rerun()
                    else:
                        if st.button("↺ Reopen", key=f"reopen_inq_{inq['id']}", use_container_width=True):
                            update_inquiry_status(inq["id"], "new")
                            st.rerun()
                with b_c2:
                    if st.button("🗑️ Del", key=f"del_inq_{inq['id']}", use_container_width=True, help="Delete inquiry"):
                        delete_inquiry(inq["id"])
                        st.rerun()
