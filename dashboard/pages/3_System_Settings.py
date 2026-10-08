"""System Settings page — standalone sub-page with shared theme (Admin Only)."""
import streamlit as st
# ── Auth & Admin Guard ─────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Session expired or not logged in. Please log in again.")
    st.page_link("app.py", label="🔑 Go to Login", icon="🔑")
    st.stop()

if st.session_state.get("user_role") != "admin":
    st.error("⛔ Access restricted to platform administrators.")
    st.stop()
# ── End Auth Guard ─────────────────────────────────────────────────────────
import yaml
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from theme import hero_banner, section_header, PROJECT_ROOT, inject_css
inject_css()

hero_banner("System Settings", "Notifications, technical configuration, and data management")

# Quick Navigation Jump Bar
st.markdown("""
<div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center;">
    <span style="font-size:0.8rem;color:var(--text-secondary);font-weight:600;">⚡ Quick Jump:</span>
    <a href="#strategy-models-gatekeeper" style="text-decoration:none;padding:5px 14px;background:rgba(255,214,0,0.12);border:1px solid #ffd60066;border-radius:20px;font-size:0.8rem;font-weight:700;color:#ffd600;">🌟 Dynamic YTD & Strategy Models</a>
    <a href="#apex-connect" style="text-decoration:none;padding:5px 14px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:20px;font-size:0.8rem;font-weight:600;color:var(--text-secondary);">🤖 Apex Connect</a>
    <a href="#neural-brain" style="text-decoration:none;padding:5px 14px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:20px;font-size:0.8rem;font-weight:600;color:var(--text-secondary);">🧠 Neural AI Backbone</a>
    <a href="#asset-classes" style="text-decoration:none;padding:5px 14px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:20px;font-size:0.8rem;font-weight:600;color:var(--text-secondary);">🌐 Asset Classes</a>
</div>
""", unsafe_allow_html=True)

CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

config = load_config()

# Notifications
with st.container(border=True):
    section_header("🔔", "Notifications")

    notif = config.get('notifications', {}).get('telegram', {})
    enable_tg = st.toggle("Enable Telegram Alerts", value=notif.get('enabled', False))
    bot_token = st.text_input("Telegram Bot Token", value=notif.get('bot_token', ''), type="password", key="tg_token")
    chat_id = st.text_input("Telegram Chat ID", value=notif.get('chat_id', ''), key="tg_chat")

    if enable_tg and bot_token and chat_id:
        st.markdown("""
        <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.2);border-radius:20px;font-size:0.75rem;font-weight:600;color:var(--success);">
            <span class="status-dot"></span> Connected
        </span>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;background:rgba(255,82,82,0.08);border:1px solid rgba(255,82,82,0.2);border-radius:20px;font-size:0.75rem;font-weight:600;color:#FF5252;">
            <span style="width:8px;height:8px;border-radius:50%;background:#FF5252;"></span> Not Configured
        </span>
        """, unsafe_allow_html=True)

st.markdown("<br><div id='apex-connect'></div>", unsafe_allow_html=True)

# ── Apex Connect (Auto-Trading) ──────────────────────
with st.container(border=True):
    section_header("🤖", "Apex Connect (Auto-Trading)")

    mt5_config = config.get('mt5', {})
    enable_mt5 = st.toggle("Enable Apex Connect Bridge", value=mt5_config.get('enabled', False), help="Automatically execute new signals on your local MT5 terminal.")

    c1, c2, c3 = st.columns(3)
    with c1:
        risk_options = ["Account %", "Fixed Cash ($)", "Fixed Lot"]
        current_risk = str(mt5_config.get('risk_type', 'percent')).lower()
        if current_risk in ('fixed_cash', 'cash', 'usd', 'fixed_usd', 'dollar'):
            idx = 1
        elif current_risk in ('fixed', 'fixed_lot', 'lot'):
            idx = 2
        else:
            idx = 0
        risk_type_sel = st.selectbox("Risk Model", risk_options, index=idx, key="mt5_risk_type")

    with c2:
        default_val = mt5_config.get('risk_value', 0.5 if idx == 0 else (50.0 if idx == 1 else 0.01))
        if risk_type_sel == "Fixed Lot":
            risk_value = st.number_input("Lot Size", min_value=0.01, max_value=50.0, value=float(default_val), step=0.01, key="mt5_risk_val_lot")
            st.caption("Trades will use this exact lot size.")
        elif risk_type_sel == "Fixed Cash ($)":
            risk_value = st.number_input("Risk $ per Trade", min_value=1.0, max_value=100000.0, value=float(default_val if default_val >= 1.0 else 50.0), step=10.0, key="mt5_risk_val_cash")
            st.caption("Lots auto-calculated from SL distance.")
        else:
            risk_value = st.number_input("Risk % per Trade", min_value=0.1, max_value=5.0, value=float(default_val if default_val <= 5.0 else 0.5), step=0.1, key="mt5_risk_val_pct")
            st.caption("Calculates lots based on SL distance & Equity.")

    with c3:
        max_open_sel = st.number_input(
            "Max Concurrent Trades", min_value=1, max_value=20,
            value=int(mt5_config.get('max_open_trades', 3)), step=1, key="mt5_max_open",
            help="Maximum simultaneous open positions allowed on MT5."
        )
        st.caption("Max open positions concurrency cap.")

    if enable_mt5:
        st.markdown("""
        <div style="margin-top: 12px; padding: 12px; background: rgba(0, 230, 118, 0.05); border-left: 3px solid var(--accent-green); border-radius: 4px;">
            <small><strong>✅ Bridge Active:</strong> The 'apex_connect' service will monitor for new signals and execute them based on these settings.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top: 12px; padding: 12px; background: rgba(255, 255, 255, 0.03); border-left: 3px solid var(--text-tertiary); border-radius: 4px;">
            <small><strong>⏸️ Bridge Paused:</strong> Signals will be generated but NOT executed automatically.</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><div id='neural-brain'></div>", unsafe_allow_html=True)

# ── AI Model Selection ─────────────────────────────────────────────────────
with st.container(border=True):
    section_header("🧠", "AI Model Selection")

    foundation_cfg = config.get('foundation', {})
    current_version = foundation_cfg.get('active_version', 'v3')
    current_routing = config.get('fleet', {}).get('routing_mode', 'truth')

    MODEL_OPTIONS = {
        "Foundation v3 (Active — Recommended)": {
            "version": "v3",
            "description": "57-feature Temporal Fusion Transformer trained on 30 pairs + Gold + macro context (SP500, VIX, DXY). 48-bar sequence window. OOS Accuracy: **41.9%** (3-class with WAIT class). Best overall signal quality.",
            "badge": "🟢 ACTIVE",
            "badge_color": "#00e676",
        },
        "Foundation v1 (Stable — 833k samples)": {
            "version": "v1",
            "description": "34-feature TFT trained on 833,896 samples across all pairs. Includes currency strength, DXY proxy, Gold, VIX proxy, yield curve. Battle-tested on live account since May 2026.",
            "badge": "🔵 STABLE",
            "badge_color": "#29b6f6",
        },
        "Foundation v2 (Extended — 29 pairs)": {
            "version": "v2",
            "description": "Extended TFT covering 29 Forex pairs with longer OOS validation (30 days). OOS Accuracy: **37.4%** (3-class). Experimental — scaler not bundled, uses v1 fallback scaler.",
            "badge": "🟡 EXPERIMENTAL",
            "badge_color": "#ffd600",
        },
    }

    version_to_label = {v["version"]: k for k, v in MODEL_OPTIONS.items()}
    current_label = version_to_label.get(current_version, list(MODEL_OPTIONS.keys())[0])

    st.markdown("#### Foundation Brain")
    st.caption("The Foundation Brain is the global Temporal Fusion Transformer (TFT) model that powers signal generation across all currency pairs.")

    model_sel_label = st.radio(
        "Select Active Foundation Model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(current_label),
        key="model_version_sel",
        label_visibility="collapsed"
    )

    sel_meta = MODEL_OPTIONS[model_sel_label]
    badge_col, desc_col = st.columns([1, 5])
    with badge_col:
        st.markdown(
            f'<div style="margin-top:4px;padding:5px 10px;background:rgba(0,0,0,0.15);border:1px solid {sel_meta["badge_color"]}33;border-radius:8px;text-align:center;font-weight:700;color:{sel_meta["badge_color"]};font-size:0.78rem">{sel_meta["badge"]}</div>',
            unsafe_allow_html=True
        )
    with desc_col:
        st.markdown(f'<small style="color:var(--text-secondary)">{sel_meta["description"]}</small>', unsafe_allow_html=True)

    new_version = sel_meta["version"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Routing Mode")
    st.caption("Controls how signals are generated per symbol — either through the Foundation Brain alone, or via an Ensemble combining Foundation + Expert adapted models.")

    ROUTING_OPTIONS = {
        "truth (Foundation Only)": {
            "value": "truth",
            "description": "Uses the active Foundation Brain for all 31 pairs. Clean, predictable, all signals from one unified source. **Current live configuration.**",
        },
        "ensemble (Foundation + Expert Adapters)": {
            "value": "ensemble",
            "description": "For the 33 pairs with a trained Expert Adapter, uses a blended prediction from Foundation Brain + per-symbol transfer-learned Expert. May improve win rate on well-represented pairs.",
        },
    }

    routing_label_map = {v["value"]: k for k, v in ROUTING_OPTIONS.items()}
    current_routing_label = routing_label_map.get(current_routing, list(ROUTING_OPTIONS.keys())[0])

    routing_sel_label = st.radio(
        "Signal Routing Mode",
        list(ROUTING_OPTIONS.keys()),
        index=list(ROUTING_OPTIONS.keys()).index(current_routing_label),
        key="routing_mode_sel",
        label_visibility="collapsed"
    )
    new_routing = ROUTING_OPTIONS[routing_sel_label]["value"]
    st.markdown(f'<small style="color:var(--text-secondary)">{ROUTING_OPTIONS[routing_sel_label]["description"]}</small>', unsafe_allow_html=True)

    if new_version != current_version or new_routing != current_routing:
        st.warning(
            f"⚠️ **Unsaved Changes**: Foundation model will switch from **{current_version.upper()}** → **{new_version.upper()}** "
            f"and routing from **{current_routing}** → **{new_routing}** on next Save.",
            icon="⚠️"
        )

    # Quick-save model without touching other settings
    if st.button("⚡ Apply Model Change Now", key="quick_apply_model", use_container_width=False):
        if 'foundation' not in config:
            config['foundation'] = {}
        config['foundation']['active_version'] = new_version
        if 'fleet' not in config:
            config['fleet'] = {}
        config['fleet']['routing_mode'] = new_routing
        save_config(config)
        st.toast(f"✅ Model switched to Foundation {new_version.upper()} | Routing: {new_routing}", icon="🧠")
        time.sleep(0.5)
        st.rerun()

    st.markdown("""
    <div style="margin-top: 14px; padding: 10px 14px; background: rgba(255, 214, 0, 0.08); border-left: 3px solid #ffd600; border-radius: 4px;">
        <small>🌟 <strong>Looking for Trading Strategy Models?</strong> The deep learning Foundation Brain acts as the feature encoder. Active execution models — including the <strong>🌟 Dynamic YTD Model (Daily Winning Assets)</strong>, <strong>Confluence ML P60</strong>, <strong>Standard P25</strong>, and <strong>M15 Sniper</strong> — are configured in the <strong>Strategy Models & Live Gatekeeper</strong> section below.</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><div id='strategy-models-gatekeeper'></div>", unsafe_allow_html=True)

# ── Strategy Models & Model Gatekeeper ─────────────────────────────────────
with st.container(border=True):
    section_header("🛡️", "Strategy Models & Live Execution Gatekeeper")
    st.caption(
        "Control which strategy models are authorized to fire real broker orders on MT5 Master. "
        "**Any model not authorized for LIVE execution automatically runs in background SHADOW mode**, "
        "recording simulated paper orders with tick-level TP/SL resolution so you can compare their performance in the Performance Matrix."
    )

    from core.model_gatekeeper import load_gatekeeper_config, save_gatekeeper_config, SUPPORTED_MODELS
    gate_cfg = load_gatekeeper_config()

    # Metrics Summary Row
    live_count = sum(1 for v in gate_cfg.values() if v)
    shadow_count = len(gate_cfg) - live_count

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Strategy Models", len(gate_cfg), help="Engines monitored by the ApexForex runtime")
    with m_col2:
        st.metric("🟢 Live MT5 Models", live_count, help="Models authorized to submit live broker orders")
    with m_col3:
        st.metric("👻 Shadow Paper Models", shadow_count, help="Models generating background paper trades for comparative analysis")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 1. Dynamic YTD Model (Daily Winning Asset Strategy)
    with st.container(border=True):
        c_dyn1, c_dyn2 = st.columns([3.6, 1.4])
        with c_dyn1:
            st.markdown(r"""
            <div style="background: linear-gradient(135deg, rgba(255,214,0,0.12) 0%, rgba(255,171,0,0.04) 100%); border-left: 4px solid #ffd600; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                    <h4 style="margin:0;color:#ffd600;font-size:1.15rem;font-weight:700;">
                        🌟 Model #1: Dynamic YTD Model (Daily Winning Asset Strategy)
                    </h4>
                    <span style="background:rgba(255,214,0,0.2);border:1px solid #ffd60088;color:#ffd600;font-size:0.75rem;padding:3px 10px;border-radius:12px;font-weight:700;">PRIMARY RECOMMENDED ENGINE</span>
                </div>
                <div style="color:var(--text-secondary);font-size:0.83rem;margin-top:4px;">
                    Autonomous 24h cycle · Daily rolling whitelist · Trades exclusively YTD profitable & breakeven assets (Net R &ge; 0.0) across all active models
                </div>
            </div>
            - **Model Key**: `dynamic_ytd_model`
            - **Strategy Function**: Dynamically recalculates Year-To-Date (YTD) profitable/breakeven pairs ($\text{Net } R \ge 0.0$) every day for all activated models.
            - **Live Execution Policy**: Restricts live MT5 execution strictly to winning assets. Chronic underperforming pairs ($\text{Net } R < 0.0$) are diverted to background paper trading (shadow).
            - **Daily Repeating Cycle**: Runs autonomously every day on UTC day rollover (00:01 UTC) and continuously across all cycle ticks.
            - **Multi-Model Orchestration**: Synchronizes winning assets across Confluence ML P60, Confluence Std P25, ML M15, Std M15, and Foundation V1.
            """, unsafe_allow_html=True)
            
            from core.dynamic_model_whitelist import get_dynamic_whitelist_manager
            _dwm = get_dynamic_whitelist_manager()
            _dsum = _dwm.get_all_models_summary()
            _as_of = _dsum.get("as_of_date", "Today")
            st.caption(f"📅 Active Whitelist Date: **{_as_of}** · Autonomous Rollover: **00:01 UTC Daily**")

            st.markdown("<hr style='opacity:0.15;margin:12px 0;'>", unsafe_allow_html=True)
            st.markdown("##### 🎛️ Activate & Deactivate Strategy Models Under YTD Model")
            st.caption(
                r"Configure which strategy models feed into the Dynamic YTD Model. "
                r"The engine dynamically computes and trades winning assets ($\text{Net } R \ge 0.0$) **ONLY** for the activated models below. "
                r"Deactivated models are excluded from YTD live broker execution and diverted to background shadow."
            )

            from core.dynamic_model_whitelist import SUB_MODEL_METADATA, DEFAULT_SUB_MODELS
            _sub_cfg = _dwm.get_sub_models_config()
            _active_sub_cnt = sum(1 for v in _sub_cfg.values() if v)

            # Quick action preset buttons
            q1, q2, q3 = st.columns([1.5, 2.0, 1.5])
            with q1:
                if st.button("✅ Activate All", key="btn_act_all_ytd_sub", use_container_width=True, help="Activate all strategy models under Dynamic YTD"):
                    all_on = {k: True for k in SUB_MODEL_METADATA.keys()}
                    _dwm.save_sub_models_config(all_on)
                    st.toast("✅ All models activated under Dynamic YTD Model!", icon="🌟")
                    st.rerun()
            with q2:
                if st.button("⚡ Recommended (P60 + P25 + V1)", key="btn_rec_ytd_sub", use_container_width=True, help="Activate top performing models: ML P60, Std P25, and Macro V1"):
                    rec = {k: (k in ("confluence_ml_p60", "confluence_std_p25", "foundation_v1")) for k in SUB_MODEL_METADATA.keys()}
                    _dwm.save_sub_models_config(rec)
                    st.toast("⚡ Recommended models activated under Dynamic YTD!", icon="🌟")
                    st.rerun()
            with q3:
                if st.button("🔄 Reset Defaults", key="btn_reset_ytd_sub", use_container_width=True, help="Reset to platform default activated sub-models"):
                    _dwm.save_sub_models_config(DEFAULT_SUB_MODELS)
                    st.toast("Default sub-models restored!", icon="🔄")
                    st.rerun()

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            # Individual Model Toggles
            _models_dict = _dsum.get("models", {})
            for sm_key, sm_meta in SUB_MODEL_METADATA.items():
                is_active = bool(_sub_cfg.get(sm_key, False))
                m_info = _models_dict.get(sm_key, {})
                wp_list = m_info.get("winning_pairs", [])
                tot_trades = m_info.get("total_trades_ytd", 0)

                with st.container(border=True):
                    sm_col1, sm_col2 = st.columns([3.8, 1.2])
                    with sm_col1:
                        badge_html = f'<span style="background:rgba(0,230,118,0.15);border:1px solid #00e67655;color:#00e676;font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;">🟢 ACTIVE UNDER YTD</span>' if is_active else f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);color:var(--text-secondary);font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:600;">⚪ DEACTIVATED</span>'
                        st.markdown(f"**{sm_meta['name']}** &nbsp; {badge_html}", unsafe_allow_html=True)
                        st.caption(f"{sm_meta['description']}")
                        if is_active:
                            if wp_list:
                                p_tags = " ".join([f"`{p}`" for p in wp_list])
                                st.markdown(f"<small>Winning Assets ({len(wp_list)} pairs · {tot_trades} trades): {p_tags}</small>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<small style='color:#ffaa00;'>Active under YTD, but 0 pairs currently meet Net R &ge; 0.0 hurdle.</small>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<small style='color:var(--text-secondary);'>Deactivated &mdash; signals from this model are excluded from Dynamic YTD trading.</small>", unsafe_allow_html=True)
                    with sm_col2:
                        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                        new_toggle = st.toggle("Active under YTD", value=is_active, key=f"sub_ytd_toggle_{sm_key}")
                        if new_toggle != is_active:
                            _dwm.set_sub_model_status(sm_key, new_toggle)
                            action_word = "Activated" if new_toggle else "Deactivated"
                            st.toast(f"{sm_meta['short_name']} {action_word} under Dynamic YTD Model!", icon="🌟" if new_toggle else "⚪")
                            st.rerun()

        with c_dyn2:
            is_live_dyn = bool(gate_cfg.get("dynamic_ytd_model", True))
            badge_dyn = '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.15);border:1px solid #ffd60066;border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">🌟 LIVE ACTIVE</div>' if is_live_dyn else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,82,82,0.1);border:1px solid rgba(255,82,82,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ff5252;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_dyn, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_dyn = st.toggle("Live Execution", value=is_live_dyn, key="gate_toggle_dynamic_ytd_model")
            if t_dyn != is_live_dyn:
                gate_cfg["dynamic_ytd_model"] = t_dyn
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_dyn else "Switched to background SHADOW mode"
                st.toast(f"🌟 Dynamic YTD Model {action}!", icon="🌟" if t_dyn else "👻")
                st.rerun()

            if st.button("🔄 Recompute Whitelists Now", key="btn_settings_recompute_dyn_model", use_container_width=True):
                with st.spinner("Recomputing YTD winning assets across all models..."):
                    _dwm.compute_ytd_whitelists()
                    st.toast("✅ Dynamic Model Whitelists recomputed!", icon="🌟")
                    st.rerun()

            if st.button("🌟 Open Dynamic Terminal", key="btn_open_dyn_terminal", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "🌟 Dynamic YTD Model (Daily Winning Asset Strategy)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # 2. Confluence ML Gate (P60 Model)
    with st.container(border=True):
        c_ml1, c_ml2 = st.columns([3.6, 1.4])
        with c_ml1:
            st.markdown(r"""
            **🧠 Confluence M15 + Deep Learning (AI Gate · P60 & BE+2p)** &nbsp; <span style="background:rgba(168,85,247,0.15);border:1px solid #a855f755;color:#c084fc;font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;">LIVE AUTHORIZED</span>
            - **Model Key**: `confluence_ml_p60`
            - **AI Engine**: LightGBM Meta-Labeling Classifier ($P_{win} \ge 48\%$) with 2.5-pip buffer & 1:1.5 RRR.
            - **Profit Milestone**: Automatically executes **50% Partial Take-Profit at 60% of TP**.
            - **Spread Guard**: Moves Stop Loss to **2.0 pips beyond entry in profit** to lock risk-free spread-compensated breakeven.
            - **Asset Concurrency**: **Permitted** to enter setups even if an active trade already exists for the same asset.
            - **Live Behavior**: Submits pending stops tagged `APEX-ML-P60` to MT5 Master.
            """, unsafe_allow_html=True)
        with c_ml2:
            is_live_ml = bool(gate_cfg.get("confluence_ml_p60", True))
            badge_ml = '<div style="margin-top:4px;padding:5px 10px;background:rgba(168,85,247,0.15);border:1px solid #a855f755;border-radius:8px;text-align:center;font-weight:700;color:#c084fc;font-size:0.78rem">🧠 LIVE ACTIVE</div>' if is_live_ml else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_ml, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_ml = st.toggle("Live Execution", value=is_live_ml, key="gate_toggle_conf_ml_p60")
            if t_ml != is_live_ml:
                gate_cfg["confluence_ml_p60"] = t_ml
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_ml else "Switched to background SHADOW mode"
                st.toast(f"🧠 Confluence AI Gate P60 {action}!", icon="🧠" if t_ml else "👻")
                st.rerun()

            if st.button("🧠 Open P60 AI Terminal", key="btn_open_conf_ml_settings", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "🧠 Confluence ML M15 (P60 · 60% Partial + BE+2p)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # 3. Confluence Standard (P25 Model)
    with st.container(border=True):
        c_std1, c_std2 = st.columns([3.6, 1.4])
        with c_std1:
            st.markdown("""
            **⚡ Confluence M15 Standard (Rule-Based · P25 & BE+2p)** &nbsp; <span style="background:rgba(0,230,118,0.15);border:1px solid #00e67655;color:#00e676;font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;">LIVE AUTHORIZED</span>
            - **Model Key**: `confluence_std_p25`
            - **Logic**: Pure 3-candle state machine (Candle 1 break $\\to$ Candle 2 Magic Candle $\\to$ Candle 3 wick entry) with 2.5-pip buffer & 1:1.5 RRR.
            - **Profit Milestone**: Automatically executes **50% Partial Take-Profit at 25% of TP**.
            - **Spread Guard**: Moves Stop Loss to **2.0 pips beyond entry in profit** to lock risk-free spread-compensated breakeven.
            - **Asset Concurrency**: **Permitted** to enter setups even if an active trade already exists for the same asset.
            - **Live Behavior**: Submits pending stops tagged `APEX-STD-P25` to MT5 Master.
            """)
        with c_std2:
            is_live_std = bool(gate_cfg.get("confluence_std_p25", True))
            badge_std = '<div style="margin-top:4px;padding:5px 10px;background:rgba(0,230,118,0.12);border:1px solid #00e67644;border-radius:8px;text-align:center;font-weight:700;color:#00e676;font-size:0.78rem">🟢 LIVE ACTIVE</div>' if is_live_std else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_std, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_std = st.toggle("Live Execution", value=is_live_std, key="gate_toggle_conf_std_p25")
            if t_std != is_live_std:
                gate_cfg["confluence_std_p25"] = t_std
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_std else "Switched to background SHADOW mode"
                st.toast(f"⚡ Confluence Standard P25 {action}!", icon="⚡" if t_std else "👻")
                st.rerun()

            if st.button("⚡ Open P25 Standard Terminal", key="btn_open_conf_std_settings", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "⚡ Confluence Standard M15 (P25 · 25% Partial + BE+2p)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # 4. Confluence ML Gate (Original Model - Fixed 1.5R)
    with st.container(border=True):
        c_oml1, c_oml2 = st.columns([3.6, 1.4])
        with c_oml1:
            st.markdown(r"""
            **🧠 Confluence M15 + Deep Learning (Original AI Gate · Fixed 1.5R)**
            - **Model Key**: `confluence_ml_m15`
            - **AI Engine**: LightGBM Meta-Labeling Classifier ($P_{win} \ge 48\%$) with 2.5-pip buffer & 1:1.5 RRR.
            - **Profit Milestone**: **Fixed Target** — Runs full position to 1:1.5 Take Profit or Stop Loss (Zero partial profit closures, zero early SL moves).
            - **Asset Concurrency**: **Standard Non-Concurrent** (Blocks new entries if an active order/trade exists for the asset).
            - **Live Behavior**: Submits pending stops tagged `APEX-ML` to MT5 Master.
            """, unsafe_allow_html=True)
        with c_oml2:
            is_live_oml = bool(gate_cfg.get("confluence_ml_m15", False))
            badge_oml = '<div style="margin-top:4px;padding:5px 10px;background:rgba(168,85,247,0.15);border:1px solid #a855f755;border-radius:8px;text-align:center;font-weight:700;color:#c084fc;font-size:0.78rem">🧠 LIVE ACTIVE</div>' if is_live_oml else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_oml, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_oml = st.toggle("Live Execution", value=is_live_oml, key="gate_toggle_conf_ml_m15")
            if t_oml != is_live_oml:
                gate_cfg["confluence_ml_m15"] = t_oml
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_oml else "Switched to background SHADOW mode"
                st.toast(f"🧠 Original Confluence AI Gate {action}!", icon="🧠" if t_oml else "👻")
                st.rerun()

            if st.button("🧠 Open Original AI Terminal", key="btn_open_conf_ml_m15_settings", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "🧠 Confluence ML M15 (Original AI Gate · Fixed 1.5R)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # 5. Confluence Standard (Original Model - Fixed 1.5R)
    with st.container(border=True):
        c_ostd1, c_ostd2 = st.columns([3.6, 1.4])
        with c_ostd1:
            st.markdown("""
            **⚡ Confluence M15 Standard (Original Rule-Based · Fixed 1.5R)**
            - **Model Key**: `confluence_m15`
            - **Logic**: Pure 3-candle state machine (Candle 1 break $\\to$ Candle 2 Magic Candle $\\to$ Candle 3 wick entry) with 2.5-pip buffer & 1:1.5 RRR.
            - **Profit Milestone**: **Fixed Target** — Runs full position to 1:1.5 Take Profit or Stop Loss (Zero partial profit closures, zero early SL moves).
            - **Asset Concurrency**: **Standard Non-Concurrent** (Blocks new entries if an active order/trade exists for the asset).
            - **Live Behavior**: Submits pending stops tagged `APEX-STD` to MT5 Master.
            """)
        with c_ostd2:
            is_live_ostd = bool(gate_cfg.get("confluence_m15", False))
            badge_ostd = '<div style="margin-top:4px;padding:5px 10px;background:rgba(0,230,118,0.12);border:1px solid #00e67644;border-radius:8px;text-align:center;font-weight:700;color:#00e676;font-size:0.78rem">🟢 LIVE ACTIVE</div>' if is_live_ostd else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_ostd, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_ostd = st.toggle("Live Execution", value=is_live_ostd, key="gate_toggle_conf_std_m15")
            if t_ostd != is_live_ostd:
                gate_cfg["confluence_m15"] = t_ostd
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_ostd else "Switched to background SHADOW mode"
                st.toast(f"⚡ Original Confluence Standard {action}!", icon="⚡" if t_ostd else "👻")
                st.rerun()

            if st.button("⚡ Open Original Standard Terminal", key="btn_open_conf_std_m15_settings", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "⚡ Confluence Standard M15 (Original Rule-Based · Fixed 1.5R)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # 6. Foundation V1 Macro AI
    with st.container(border=True):
        c_fnd1, c_fnd2 = st.columns([3.6, 1.4])
        with c_fnd1:
            st.markdown(r"""
            **🌐 Foundation V1 Macro AI (Temporal Fusion Transformer)**
            - **Model Key**: `foundation_v1`
            - **Logic**: Global multi-pair neural network trained on 833,896 samples evaluating yield spreads, DXY proxy, GMM regimes.
            - **Live Behavior**: Executes live market orders when conviction $\ge 61\%$. *(Protected: Kept in shadow mode by default to prevent unintended live orders)*.
            - **Shadow Behavior**: Scans all 31 pairs hourly, recording signals as shadow paper trades to compare long-term macro accuracy.
            """)
        with c_fnd2:
            is_live_fnd = bool(gate_cfg.get("foundation_v1", False))
            badge_fnd = '<div style="margin-top:4px;padding:5px 10px;background:rgba(41,182,246,0.15);border:1px solid #29b6f644;border-radius:8px;text-align:center;font-weight:700;color:#29b6f6;font-size:0.78rem">🔵 LIVE ACTIVE</div>' if is_live_fnd else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_fnd, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_fnd = st.toggle("Live Execution", value=is_live_fnd, key="gate_toggle_fnd_v1")
            if t_fnd != is_live_fnd:
                gate_cfg["foundation_v1"] = t_fnd
                save_gatekeeper_config(gate_cfg)
                action = "Authorized for LIVE MT5 execution" if t_fnd else "Switched to background SHADOW mode"
                st.toast(f"🌐 Foundation Macro AI {action}!", icon="🌐" if t_fnd else "👻")
                st.rerun()

    # 7. Manual M15 Wick Sniper
    with st.container(border=True):
        m15_c1, m15_c2 = st.columns([3.6, 1.4])
        with m15_c1:
            st.markdown("""
            **🎯 Manual M15 Wick Sniper (Semi-Automated Discretionary)**
            - **Model Key**: `manual_m15`
            - **Order Type**: Pending Stop (`BUY_STOP` / `SELL_STOP`) placed at candle wick levels from the Terminal.
            - **Live Behavior**: Armed button submits real orders to MT5 Master with instant copy-trading broadcast.
            - **Shadow Behavior**: Discretionary entries are saved as shadow paper trades for strategy testing.
            """)
        with m15_c2:
            is_live_sniper = bool(gate_cfg.get("manual_m15", True))
            badge_sniper = '<div style="margin-top:4px;padding:5px 10px;background:rgba(0,230,118,0.12);border:1px solid #00e67644;border-radius:8px;text-align:center;font-weight:700;color:#00e676;font-size:0.78rem">🟢 LIVE ARMED</div>' if is_live_sniper else '<div style="margin-top:4px;padding:5px 10px;background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);border-radius:8px;text-align:center;font-weight:700;color:#ffd600;font-size:0.78rem">👻 SHADOW MODE</div>'
            st.markdown(badge_sniper, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            t_sniper = st.toggle("Live Execution", value=is_live_sniper, key="gate_toggle_manual_sniper")
            if t_sniper != is_live_sniper:
                gate_cfg["manual_m15"] = t_sniper
                save_gatekeeper_config(gate_cfg)
                action = "Armed for LIVE MT5 execution" if t_sniper else "Switched to background SHADOW mode"
                st.toast(f"🎯 Manual Sniper {action}!", icon="🎯" if t_sniper else "👻")
                st.rerun()

            if st.button("🎯 Open Sniper Terminal", key="btn_open_sniper_settings", use_container_width=True):
                st.session_state["terminal_active_strategy"] = "🎯 M15 Wick Sniper (Discretionary)"
                st.session_state["nav_to_terminal"] = True
                st.rerun()

    # Informational banner on comparison & aggregate
    st.info(
        "💡 **Automatic Performance Tracking**: All models (both Live and Shadow) are continuously tracked and resolved "
        "by the background watchdog. Head to the **Performance Matrix** on the main dashboard to view individual win rates, "
        "profit factors, and the comprehensive **Aggregate Performance** across all models.",
        icon="📊"
    )

st.markdown("<br><div id='asset-classes'></div>", unsafe_allow_html=True)

# ── Asset Classes & Active Market Instruments ──────────────────────────────
with st.container(border=True):
    section_header("🌐", "Asset Classes & Active Market Instruments")
    st.caption("Enable or disable entire asset classes and view all 35 active instruments monitored across the platform.")

    asset_cfg = config.get('asset_classes', {'forex': True, 'commodities': True, 'crypto': True})

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        enable_fx = st.toggle("Forex Pairs", value=bool(asset_cfg.get('forex', True)), key="toggle_asset_fx")
        st.caption("29 Majors, Minors & Crosses (Sun 5 PM – Fri 5 PM EST)")
    with col_a2:
        enable_comm = st.toggle("Commodities & Metals", value=bool(asset_cfg.get('commodities', True)), key="toggle_asset_comm")
        st.caption("XAUUSD (Gold), XAGUSD (Silver), USOIL.cash (WTI Crude)")
    with col_a3:
        enable_cr = st.toggle("Cryptocurrency (24/7)", value=bool(asset_cfg.get('crypto', True)), key="toggle_asset_crypto")
        st.caption("BTCUSD, ETHUSD, SOLUSD — Trades 24/7 with zero weekend halts")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    tab_crypto, tab_comm, tab_fx = st.tabs(["🪙 Cryptocurrencies (3)", "🏆 Commodities & Metals (3)", "⚡ Forex Pairs (29)"])

    with tab_crypto:
        st.markdown("""
| Symbol | Asset Name | Trading Hours | Pip / Point Scale | Engine Status | Weekend Auto-Exit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BTCUSD** | Bitcoin / USD | **24/7 Continuous** | 1.00 USD / pip | 🟢 ACTIVE | 🛡️ Exempt (Keeps Running) |
| **ETHUSD** | Ethereum / USD | **24/7 Continuous** | 0.10 USD / pip | 🟢 ACTIVE | 🛡️ Exempt (Keeps Running) |
| **SOLUSD** | Solana / USD | **24/7 Continuous** | 0.01 USD / pip | 🟢 ACTIVE | 🛡️ Exempt (Keeps Running) |
""")
        st.info("🪙 **24/7 Crypto Engine Active**: Crypto pairs are fully supported across Confluence M15, Manual Sniper, Copy Trading, and Auto-Execution with weekend halts disabled.", icon="🪙")

    with tab_comm:
        st.markdown("""
| Symbol | Asset Name | Trading Hours | Pip / Point Scale | Engine Status | Weekend Auto-Exit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XAUUSD** | Spot Gold / USD | Mon–Fri (Market Hours) | 0.10 USD / pip | 🟢 ACTIVE | 🔒 Closes Friday 4:55 PM EST |
| **XAGUSD** | Spot Silver / USD | Mon–Fri (Market Hours) | 0.01 USD / pip | 🟢 ACTIVE | 🔒 Closes Friday 4:55 PM EST |
| **USOIL.cash** | WTI Crude Oil Spot | Mon–Fri (Market Hours) | 0.01 USD / pip | 🟢 ACTIVE | 🔒 Closes Friday 4:55 PM EST |
""")

    with tab_fx:
        fx_majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        fx_minors = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "USDSGD"]
        fx_crosses = ["EURAUD", "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD", "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "NZDCAD", "NZDCHF"]

        c_fx1, c_fx2, c_fx3 = st.columns(3)
        with c_fx1:
            st.markdown(f"**⚡ Majors ({len(fx_majors)})**")
            st.caption(", ".join(fx_majors))
        with c_fx2:
            st.markdown(f"**🔷 Minors ({len(fx_minors)})**")
            st.caption(", ".join(fx_minors))
        with c_fx3:
            st.markdown(f"**🔶 Crosses ({len(fx_crosses)})**")
            st.caption(", ".join(fx_crosses))

st.markdown("<br>", unsafe_allow_html=True)

# ── Asset-Adapted Stop Loss & Breakeven Buffers ───────────────────────────
with st.container(border=True):
    section_header("🎯", "Asset-Adapted Stop Loss & Breakeven Buffers")
    st.caption(
        "Instruments such as Gold, Silver, Oil, and Cryptocurrencies have pip scales and volatility orders of magnitude larger than standard Forex. "
        "The Confluence Engine automatically applies scaled buffers so each asset has proper proportional wick clearance (~20% of M15 candle range) "
        "and cannot be prematurely stopped out inside broker spread."
    )

    conf_block = config.get("confluence_model", {})
    saved_bufs = conf_block.get("asset_buffers", {})
    saved_bes = conf_block.get("asset_be_offsets", {})

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.markdown("##### ⚡ Forex Pairs")
        buf_fx = st.number_input(
            "Forex SL Buffer (pips)",
            min_value=0.5, max_value=20.0,
            value=float(saved_bufs.get("forex", 2.5)),
            step=0.5,
            key="cfg_buf_fx",
            help="Stop loss distance beyond magic candle wick for Forex majors and crosses."
        )
        be_fx = st.number_input(
            "Forex BE Offset (pips)",
            min_value=0.5, max_value=10.0,
            value=float(saved_bes.get("forex", 2.0)),
            step=0.5,
            key="cfg_be_fx",
            help="Pips locked in profit beyond entry when partial profit is secured."
        )
        st.caption("Standard 2.5p buffer / 2.0p BE offset")

    with b_col2:
        st.markdown("##### 🏆 Gold (XAUUSD)")
        buf_gold = st.number_input(
            "Gold SL Buffer (pips)",
            min_value=5.0, max_value=100.0,
            value=float(saved_bufs.get("gold", 25.0)),
            step=1.0,
            key="cfg_buf_gold",
            help="25.0 pips = $2.50 in Gold price scale (0.1 pip size). Clears $0.30 broker spread with 8.3x margin."
        )
        be_gold = st.number_input(
            "Gold BE Offset (pips)",
            min_value=2.0, max_value=50.0,
            value=float(saved_bes.get("gold", 10.0)),
            step=1.0,
            key="cfg_be_gold",
            help="10.0 pips = $1.00 locked in profit beyond entry."
        )
        st.caption("25.0p buffer ($2.50) / 10.0p BE ($1.00)")

    with b_col3:
        st.markdown("##### 🛢️ Silver & Crude Oil")
        buf_silver = st.number_input(
            "Silver SL Buffer (pips)",
            min_value=5.0, max_value=50.0,
            value=float(saved_bufs.get("silver", 20.0)),
            step=1.0,
            key="cfg_buf_silver",
            help="20.0 pips = $0.20 in Silver price scale (0.01 pip size)."
        )
        buf_oil = st.number_input(
            "Crude Oil SL Buffer (pips)",
            min_value=5.0, max_value=50.0,
            value=float(saved_bufs.get("oil", 15.0)),
            step=1.0,
            key="cfg_buf_oil",
            help="15.0 pips = $0.15 in USOIL price scale (0.01 pip size)."
        )
        st.caption("Silver: 20p ($0.20) · Oil: 15p ($0.15)")

    with b_col4:
        st.markdown("##### 🪙 Cryptocurrencies")
        buf_btc = st.number_input(
            "Bitcoin SL Buffer (pips)",
            min_value=10.0, max_value=200.0,
            value=float(saved_bufs.get("btc", 60.0)),
            step=5.0,
            key="cfg_buf_btc",
            help="60.0 pips = $60.00 in BTCUSD price scale (1.0 pip size)."
        )
        buf_eth = st.number_input(
            "Ethereum SL Buffer (pips)",
            min_value=5.0, max_value=100.0,
            value=float(saved_bufs.get("eth", 30.0)),
            step=2.0,
            key="cfg_buf_eth",
            help="30.0 pips = $3.00 in ETHUSD price scale (0.1 pip size)."
        )
        st.caption("BTC: 60p ($60) · ETH: 30p ($3)")

    st.markdown("""
    <div style="background:rgba(0,230,118,0.05);border-left:3px solid var(--accent-green);border-radius:4px;padding:10px 14px;margin-top:8px;">
        <small><strong>🛡️ Spread Floor Protection:</strong> Regardless of manual input, all live orders automatically enforce a dynamic broker spread floor (SL Buffer &ge; 1.5&times; live broker spread, BE Offset &ge; Spread + 1.0 pip). Stop losses can never be placed within broker spread.</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Technical Config
with st.container(border=True):
    section_header("🔧", "Technical Configuration")

    tech_config = config.get('technical', {})
    default_cur = tech_config.get('currency', 'USD')
    default_tz = tech_config.get('timezone', 'UTC')
    default_theme = tech_config.get('theme', 'Dark Mode')
    default_loglevel = tech_config.get('log_level', 'INFO')
    active_provider = config.get('data_provider', {}).get('active', 'mt5')

    cur_options = ["USD", "EUR", "GBP", "JPY"]
    tz_options = ["UTC", "EST", "PST", "GMT"]
    theme_options = ["Dark Mode", "Light Mode", "System Default"]
    loglevel_options = ["INFO", "DEBUG", "WARNING", "ERROR"]
    provider_options = ["mt5", "yfinance"]

    def get_idx(val, options, default=0):
        try:
            return options.index(val)
        except ValueError:
            return default

    col1, col2 = st.columns(2)
    with col1:
        cur_sel = st.selectbox("Default Currency", cur_options, index=get_idx(default_cur, cur_options), key="def_currency")
        tz_sel = st.selectbox("Timezone", tz_options, index=get_idx(default_tz, tz_options), key="tz")
        provider_sel = st.selectbox("Active Data Provider", provider_options, index=get_idx(active_provider, provider_options), key="provider")
    with col2:
        theme_sel = st.selectbox("Theme", theme_options, index=get_idx(default_theme, theme_options), key="theme_sel")
        loglevel_sel = st.selectbox("Log Level", loglevel_options, index=get_idx(default_loglevel, loglevel_options), key="log_level")

st.markdown("<br>", unsafe_allow_html=True)

# Data Management
with st.container(border=True):
    section_header("💾", "Data Management & Database Exports")

    def _get_cache_size():
        total_bytes = 0
        try:
            for p in PROJECT_ROOT.rglob("__pycache__"):
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            total_bytes += f.stat().st_size
            streamlit_cache = Path.home() / ".streamlit" / "cache"
            if streamlit_cache.exists():
                for f in streamlit_cache.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size
        except Exception:
            pass
        mb = total_bytes / (1024 * 1024)
        if mb >= 1.0:
            return f"{mb:.1f} MB"
        return f"{max(0.1, total_bytes / 1024):.1f} KB"

    c1, c2 = st.columns(2)
    with c1:
        cache_str = _get_cache_size()
        st.markdown(f"""
        <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
            Cache size: <span style="color: var(--accent-cyan); font-weight: 600;">{cache_str}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Platform cache cleared!")
            time.sleep(0.5)
            st.rerun()
            
    with c2:
        st.markdown("""
        <div style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.85rem;">
            Database Export: <span style="color: var(--accent-cyan); font-weight: 600;">Live CSV — always current</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        import sqlite3
        import io

        def _db_to_csv(db_path: Path, table: str, where_clause: str = "") -> str:
            """Read a full DB table and return as a CSV string."""
            import pandas as pd
            if not db_path.exists():
                return ""
            try:
                conn = sqlite3.connect(str(db_path))
                query = f"SELECT * FROM {table} {where_clause}"
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df.to_csv(index=False)
            except Exception as e:
                return ""

        signals_db  = PROJECT_ROOT / "signals.db"
        users_db    = PROJECT_ROOT / "user_accounts.db"
        portal_db   = PROJECT_ROOT / "portal_users.db"

        # Signals (Exclude SYSTEM heartbeats)
        signals_csv = _db_to_csv(signals_db, "signals", "WHERE symbol != 'SYSTEM'")
        if signals_csv:
            st.download_button(
                label="📥 Download Signals CSV",
                data=signals_csv,
                file_name="signals_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # MT5 Subscriber Accounts
        users_csv = _db_to_csv(users_db, "user_accounts")
        if users_csv:
            st.download_button(
                label="📥 Download MT5 Subscriber Accounts CSV",
                data=users_csv,
                file_name="user_accounts_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Portal logins
        portal_csv = _db_to_csv(portal_db, "portal_users")
        if portal_csv:
            st.download_button(
                label="📥 Download Portal Logins CSV",
                data=portal_csv,
                file_name="portal_users_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Inbound Leads & Upgrade Requests
        inq_csv = _db_to_csv(portal_db, "inquiries")
        if inq_csv:
            st.download_button(
                label="📥 Download Inbound Leads CSV",
                data=inq_csv,
                file_name="leads_export.csv",
                mime="text/csv",
                use_container_width=True
            )



# Save Settings
if st.button("💾 Save System Settings", type="primary", use_container_width=True):
    fresh_config = load_config()
    if 'model_gatekeeper' in fresh_config:
        config['model_gatekeeper'] = fresh_config['model_gatekeeper']
    if 'confluence_model' in fresh_config:
        config['confluence_model'] = fresh_config['confluence_model']
    if 'dynamic_model_whitelist' in fresh_config:
        config['dynamic_model_whitelist'] = fresh_config['dynamic_model_whitelist']

    if 'notifications' not in config: config['notifications'] = {}
    config['notifications']['telegram'] = {
        'enabled': enable_tg,
        'bot_token': bot_token,
        'chat_id': chat_id,
        'channel_id': notif.get('channel_id', ''),
        'notify_shadow_trades': notif.get('notify_shadow_trades', False),
        'alert_threshold': notif.get('alert_threshold', 0.6)
    }
    
    if 'mt5' not in config: config['mt5'] = {}
    config['mt5']['enabled'] = enable_mt5
    if risk_type_sel == 'Account %':
        r_type_key = 'percent'
    elif '$' in risk_type_sel:
        r_type_key = 'fixed_cash'
    else:
        r_type_key = 'fixed'
    config['mt5']['risk_type'] = r_type_key
    config['mt5']['risk_value'] = float(risk_value)
    config['mt5']['max_open_trades'] = int(max_open_sel)
    if 'confluence_model' in config and isinstance(config['confluence_model'], dict):
        config['confluence_model']['risk_type'] = r_type_key
        config['confluence_model']['risk_value'] = float(risk_value)
    else:
        config['confluence_model'] = {}

    if 'asset_buffers' not in config['confluence_model']:
        config['confluence_model']['asset_buffers'] = {}
    if 'asset_be_offsets' not in config['confluence_model']:
        config['confluence_model']['asset_be_offsets'] = {}

    config['confluence_model']['asset_buffers']['forex'] = float(buf_fx)
    config['confluence_model']['asset_buffers']['gold'] = float(buf_gold)
    config['confluence_model']['asset_buffers']['silver'] = float(buf_silver)
    config['confluence_model']['asset_buffers']['oil'] = float(buf_oil)
    config['confluence_model']['asset_buffers']['btc'] = float(buf_btc)
    config['confluence_model']['asset_buffers']['eth'] = float(buf_eth)
    config['confluence_model']['sl_buffer_pips'] = float(buf_fx)

    config['confluence_model']['asset_be_offsets']['forex'] = float(be_fx)
    config['confluence_model']['asset_be_offsets']['gold'] = float(be_gold)
    
    config['technical'] = {
        'currency': cur_sel,
        'timezone': tz_sel,
        'theme': theme_sel,
        'log_level': loglevel_sel
    }
    
    if 'data_provider' not in config: config['data_provider'] = {}
    config['data_provider']['active'] = provider_sel

    # ── Asset Classes ──────────────────────────────────────────────────────
    if 'asset_classes' not in config: config['asset_classes'] = {}
    config['asset_classes']['forex'] = bool(enable_fx)
    config['asset_classes']['commodities'] = bool(enable_comm)
    config['asset_classes']['crypto'] = bool(enable_cr)

    # ── Model Selection ────────────────────────────────────────────────────
    if 'foundation' not in config: config['foundation'] = {}
    config['foundation']['active_version'] = new_version
    if 'fleet' not in config: config['fleet'] = {}
    config['fleet']['routing_mode'] = new_routing
    
    save_config(config)
    st.toast(f"System settings saved! Model: Foundation {new_version.upper()} | Routing: {new_routing}", icon="💾")
    time.sleep(1)
    st.rerun()
