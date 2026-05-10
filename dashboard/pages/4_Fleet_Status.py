import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

# Shared design system
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.theme import (
    inject_css, hero_banner, sidebar_logo, sidebar_footer, section_header, kpi_card
)

st.set_page_config(page_title="Fleet Status · ApexForex", page_icon="📊", layout="wide")
inject_css()

# Auto-refresh check
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# Last execution timestamp
st.session_state.last_ui_update = datetime.now().strftime("%H:%M:%S")

# --- Helpers ---
def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_training_log(log_path, tail_bytes=50000):
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'rb') as f:
        try:
            f.seek(-tail_bytes, os.SEEK_END)
            content = f.read().decode('utf-8', errors='ignore')
        except OSError:
            f.seek(0)
            content = f.read().decode('utf-8', errors='ignore')
            
    lines = content.splitlines()
    
    status = {
        "symbols_processed": [],
        "total_symbols": 26, # Updated to 26 for v2
        "current_symbol": "None",
        "phase": "Preparing Data",
        "keras_progress": None,
        "last_update": None,
        "metrics": {"accuracy": 0.0, "loss": 0.0},
        "history": {"step": [], "accuracy": [], "loss": []},
        "eta": "Calculating...",
        "start_time": None
    }
    
    log_time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    # Keras match: 1/1158 ━━━━━━━━━━━━━━━━━━━━ 50s 40ms/step - accuracy: 0.50 - loss: 0.69
    # Epoch match: Epoch 1/60
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
    keras_steps_pattern = re.compile(r'(\d+)/(\d+).*?accuracy:\s+([\d\.]+)\s+-\s+loss:\s+([\d\.]+)')
    sequence_pattern = re.compile(r'^\d{4}.*INFO\]\s+([A-Z]+):\s+\d+,\d+\s+sequences')
    
    total_epochs = 60
    completed_epochs = 0
    total_steps = 1158

    for line in lines:
        line = strip_ansi(line)
        
        time_match = log_time_pattern.match(line)
        if time_match:
            status["last_update"] = time_match.group(1)

        if "FOUNDATION BRAIN v2 - TRAINING START" in line:
            status["phase"] = "Initializing V2 Brain"
            
        if "Fetching" in line and "from MT5" in line:
            status["phase"] = "Fetching MT5 Historical Data"
            
        if "sequences" in line and "INFO]" in line and "Total" not in line:
            seq_match = sequence_pattern.search(line)
            if seq_match:
                symbol = seq_match.group(1)
                if symbol not in status["symbols_processed"] and len(symbol) <= 7:
                    status["symbols_processed"].append(symbol)
                status["current_symbol"] = symbol
                
        if "Total sequences across all pairs" in line:
            status["phase"] = "Building Data Corpus"

        if "Starting TFT model fit" in line:
            status["phase"] = "Model Training"
            status["current_symbol"] = "Global Brain v2"
            
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            completed_epochs = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            status["phase"] = f"Epoch {completed_epochs}/{total_epochs}"
            status["current_symbol"] = "Global Brain v2"

        keras_match = keras_steps_pattern.search(line)
        if keras_match:
            current_step = int(keras_match.group(1))
            total_steps = int(keras_match.group(2))
            train_acc = float(keras_match.group(3))
            train_loss = float(keras_match.group(4))
            
            status["keras_progress"] = ((completed_epochs - 1) * total_steps + current_step, total_steps * total_epochs)
            status["metrics"]["accuracy"] = train_acc
            status["metrics"]["loss"] = train_loss
            
            global_step = (completed_epochs * total_steps) + current_step
            status["history"]["step"].append(global_step)
            status["history"]["accuracy"].append(train_acc)
            status["history"]["loss"].append(train_loss)

    return status

def parse_specialist_log(log_path, tail_bytes=50000):
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'rb') as f:
        try:
            f.seek(-tail_bytes, os.SEEK_END)
            content = f.read().decode('utf-8', errors='ignore')
        except OSError:
            f.seek(0)
            content = f.read().decode('utf-8', errors='ignore')
            
    lines = content.splitlines()
    
    status = {
        "symbols_processed": [],
        "total_symbols": 31,
        "current_symbol": "None",
        "phase": "Preparing Fleet",
        "keras_progress": None,
        "last_update": None,
        "metrics": {"accuracy": 0.0, "loss": 0.0, "oos_wr": 0.0},
        "history": {"step": [], "accuracy": [], "loss": []},
        "eta": "Calculating...",
        "start_time": None
    }
    
    log_time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    fold_pattern = re.compile(r'Fold\s+(\d+):\s+Acc=([\d\.]+)%,\s+Loss=([\d\.]+).*OOS\s+WR=([\d\.]+)%')
    
    current_step = 0
    
    for line in lines:
        line = strip_ansi(line)
        time_match = log_time_pattern.match(line)
        if time_match:
            status["last_update"] = time_match.group(1)
            
        if "Starting expert training for" in line:
            parts = line.split("Starting expert training for")
            if len(parts) > 1:
                symbol = parts[1].split()[0].strip()
                status["current_symbol"] = symbol
                status["phase"] = f"Training {symbol}"
                
        if "Worker" in line and "Completed" in line:
            # Worker X (Pass 2): Completed EURUSD (1/10)
            parts = line.split("Completed")
            if len(parts) > 1:
                symbol = parts[1].strip().split()[0].strip()
                if symbol not in status["symbols_processed"] and len(symbol) <= 7:
                    status["symbols_processed"].append(symbol)

        if "Training single pair:" in line:
            parts = line.split("Training single pair:")
            if len(parts) > 1:
                symbol = parts[1].strip()
                status["current_symbol"] = symbol
                status["phase"] = f"Training {symbol}"
            
        fold_match = fold_pattern.search(line)
        if fold_match:
            fold_idx = int(fold_match.group(1))
            acc = float(fold_match.group(2)) / 100.0
            loss = float(fold_match.group(3))
            oos = float(fold_match.group(4)) / 100.0
            
            status["metrics"]["accuracy"] = acc
            status["metrics"]["loss"] = loss
            status["metrics"]["oos_wr"] = oos
            
            status["phase"] = f"WFCV Fold {fold_idx}/5"
            status["keras_progress"] = (fold_idx, 5) # 5 folds for specialist
            
            current_step += 1
            status["history"]["step"].append(current_step)
            status["history"]["accuracy"].append(acc)
            status["history"]["loss"].append(loss)
            
    return status

# --- Page Content ---
with st.sidebar:
    sidebar_logo()
    st.markdown("""
    <div style="text-align:center; margin-bottom: 24px;">
        <a href="/" target="_self" style="color: var(--accent-cyan); text-decoration: none; font-weight: 600; font-size: 0.85rem;">← Back to Dashboard</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Monitoring Target")
    monitor_target = st.radio("Log Source", ["Global Foundation", "Specialist Fleet Base"], index=0, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    sidebar_footer()

if monitor_target == "Global Foundation":
    hero_banner("Foundation Brain V2 Training", "Real-time monitor for Global Foundation Intelligence")
    log_dir = PROJECT_ROOT / "logs"
    log_file = log_dir / "foundation_v2_training.log"
    log_files = [log_file] if log_file.exists() else []
    parser_func = parse_training_log
    no_log_msg = f"⏳ No active training log found.\n\nStart training on the VM with:\n```\npython models/foundation_trainer_v2.py\n```\nExpected log at: `{log_dir / 'foundation_v2_training.log'}`"
else:
    hero_banner("Specialist Fleet Retraining", "Real-time monitor for Walk-Forward Cross Validation")
    log_dir = PROJECT_ROOT / "logs"
    log_file = log_dir / "specialist_progressive.log"
    log_files = [log_file] if log_file.exists() else []
    parser_func = parse_specialist_log
    no_log_msg = f"⏳ No active specialist training log found. Expected at: `{log_dir / 'specialist_progressive.log'}`"

if len(log_files) > 20:
    log_files = log_files[-20:]

if not log_files:
    st.info(no_log_msg)
else:
    status = None
    for log_path in log_files:
        current_status = parser_func(log_path)
        if current_status:
            if status is None:
                status = current_status
            else:
                # Aggregate metrics
                if current_status["keras_progress"]:
                    # Keep the maximum progress found across all logs
                    if not status["keras_progress"] or current_status["keras_progress"][0] > status["keras_progress"][0]:
                        status["keras_progress"] = current_status["keras_progress"]
                        status["metrics"] = current_status["metrics"]
                        status["phase"] = current_status["phase"]

                if current_status["last_update"]:
                    # Keep the most recent timestamp
                    if not status["last_update"] or current_status["last_update"] > status["last_update"]:
                        status["last_update"] = current_status["last_update"]
                status["current_symbol"] = current_status["current_symbol"]
                status["symbols_processed"] = list(set(status["symbols_processed"] + current_status["symbols_processed"]))
                status["eta"] = current_status["eta"]
                
                # Append history correctly
                # We only append if there is actual metric data in this log
                if current_status["history"]["step"]:
                    status["history"]["step"].extend(current_status["history"]["step"])
                    status["history"]["accuracy"].extend(current_status["history"]["accuracy"])
                    status["history"]["loss"].extend(current_status["history"]["loss"])

    if status:
        # Determine the Truly Current log for the most accurate KPIs
        # We process the latest log LAST to ensure it has the final say on Phase/Metrics
        current_run_status = parser_func(log_files[-1])
        if current_run_status:
            status["phase"] = current_run_status["phase"]
            status["metrics"] = current_run_status["metrics"]
            status["keras_progress"] = current_run_status["keras_progress"]
            status["last_update"] = current_run_status["last_update"]
            
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Current Phase", status["phase"], accent="accent-cyan"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Fleet Progress", f"{len(status['symbols_processed'])}/32 Symbols"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Intelligence Accuracy", f"{status['metrics']['accuracy']:.1%}", accent="accent-green"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Estimated Time to Completion", status["eta"], accent="accent-gold"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🚀 Summary", "📜 Institutional Audit", "🛠️ System Logs"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="glass-section">', unsafe_allow_html=True)
                section_header("📊", "Global Brain Convergence")
                
                if status["keras_progress"]:
                    curr, total = status["keras_progress"]
                    # Clamp progress to [0, 1] to prevent Streamlit exception
                    progress_pct = min(1.0, max(0.0, curr / total))
                    st.progress(progress_pct)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">
                        <span>Step {curr} / {total}</span>
                        <span>{progress_pct:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini metric chart
                    if status["history"]["accuracy"]:
                        chart_df = pd.DataFrame(status["history"]).set_index("step")
                        st.line_chart(chart_df["accuracy"], use_container_width=True)
                else:
                    st.info("Waiting for first Epoch data...")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="glass-section">', unsafe_allow_html=True)
                section_header("🛰️", "Live Fleet Preparation")
                all_symbols = [
                    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
                    "EURGBP", "EURJPY", "GBPJPY", "EURCAD", "EURAUD", "EURCHF", "GBPCHF",
                    "GBPAUD", "GBPCAD", "AUDJPY", "CADJPY", "CHFJPY", "AUDCAD", "NZDJPY",
                    "AUDNZD", "EURNZD", "GBPNZD", "AUDCHF", "NZDCAD", "CADCHF", "NZDCHF",
                    "GOLD", "CrudeOIL", "USDSGD"
                ]
                rows = [all_symbols[i:i + 6] for i in range(0, len(all_symbols), 6)]
                for row in rows:
                    cols = st.columns(6)
                    for i, sym in enumerate(row):
                        is_done = sym in status["symbols_processed"]
                        is_current = sym == status["current_symbol"]
                        bg = "rgba(0, 255, 136, 0.1)" if is_done else "rgba(255, 255, 255, 0.03)"
                        border = "1px solid var(--success)" if is_done else "1px solid var(--border-glass)"
                        if is_current:
                            border = "2px solid var(--accent-cyan)"
                            bg = "rgba(0, 229, 255, 0.05)"
                        cols[i].markdown(f"""
                        <div style="padding: 10px; border-radius: 8px; background: {bg}; border: {border}; text-align: center; font-size: 0.75rem; font-weight: 600;">
                            {sym}
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-section">', unsafe_allow_html=True)
                section_header("🧠", "Intelligence Status")
                st.write(f"**Current Task:** Learning {status['current_symbol']}")
                st.write(f"**Accuracy:** {status['metrics']['accuracy']:.4f}")
                st.write(f"**Loss:** {status['metrics']['loss']:.4f}")
                st.write(f"**Log Sync:** {status['last_update']}")
                st.write(f"**UI Refresh:** {st.session_state.last_ui_update}")
                
                if st.button("🔄 Force Refresh", type="primary", use_container_width=True):
                    with st.spinner("Re-scanning logs..."):
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        # Force a reset of the session state sync to be sure
                        st.session_state.last_ui_update = datetime.now().strftime("%H:%M:%S")
                        time.sleep(1) # Visual feedback
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="glass-section">', unsafe_allow_html=True)
            section_header("🏦", "Institutional Performance Audit")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Convergence Trend")
                if status["history"]["accuracy"]:
                    chart_df = pd.DataFrame(status["history"]).set_index("step")
                    st.line_chart(chart_df[["accuracy", "loss"]], use_container_width=True)
                else:
                    st.info("Insufficient history for convergence audit.")
            
            with c2:
                st.subheader("System Architecture")
                st.markdown("""
                - **Model Architecture**: Temporal Fusion Transformer (TFT)
                - **Input Sequence**: 60 hours (2.5 days of market history)
                - **Global Context**: Currency Strength Matrix (CSM) + DXY
                - **Intelligence Tier**: Global Foundation (3-Class)
                - **Intelligence Logic**: BUY / SELL / WAIT
                """)
            
            st.divider()
            
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Risk Performance Targets")
                st.table({
                    "Parameter": ["Target Win Rate", "Risk/Reward Ratio", "Min Stop Loss", "Max Hold Time"],
                    "Value": ["~80% (Expert)", "1 : 1.5", "25 Pips (Dynamic ATR)", "120 Hours"]
                })
            with c4:
                st.subheader("Dataset Volume")
                st.markdown(f"""
                - **Total Samples**: 329,140 candle-hours
                - **Fleet Coverage**: 31 Pairs + Gold
                - **Training Endurance**: 15 Epochs
                - **Memory Management**: Chunked In-Place Scaling
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="glass-section">', unsafe_allow_html=True)
            section_header("📜", "Full System Log Stream")
            latest_log = log_files[-1]
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                log_tail = f.readlines()[-100:]
            log_content = "".join([strip_ansi(l) for l in log_tail])
            st.code(log_content, language="text")
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"Last Log Update: {status['last_update']} | Log: {latest_log.name}")
    else:
        st.error("Could not parse log file.")

st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: var(--accent-cyan);
    }
</style>
""", unsafe_allow_html=True)

# --- Auto Refresh System ---
st.sidebar.divider()
st.session_state.auto_refresh = st.sidebar.toggle("🔄 Auto Refresh (30s)", value=st.session_state.auto_refresh)
if st.session_state.auto_refresh:
    # Use a placeholder to show countdown
    placeholder = st.sidebar.empty()
    for i in range(30, 0, -1):
        placeholder.caption(f"Refreshing in {i}s...")
        time.sleep(1)
    st.rerun()
