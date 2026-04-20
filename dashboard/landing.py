import streamlit as st
from theme import inject_css, PROJECT_ROOT

def show_landing():
    # minimalist header
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown("## ⚡ **ApexForex**")
    with c2:
        st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="margin-right: 20px; color: var(--text-secondary); cursor: pointer;">Features</span>
            <span style="margin-right: 20px; color: var(--text-secondary); cursor: pointer;">Pricing</span>
            <span style="margin-right: 20px; color: var(--text-secondary); cursor: pointer;">Contact</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)

    # HERO SECTION
    cols = st.columns([1.2, 0.8])
    with cols[0]:
        st.markdown("""
        <h1 style="font-size: 3.5rem; line-height: 1.1; background: linear-gradient(90deg, #FFFFFF, #8b95a8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Institutional-Grade<br>
            <span style="color: var(--accent-cyan); -webkit-text-fill-color: var(--accent-cyan);">AI Trading Intelligence</span>
        </h1>
        <p style="font-size: 1.2rem; color: var(--text-secondary); margin-top: 20px; max-width: 600px;">
            Stop guessing. Start executing. Our AI models analyze 31 forex pairs in real-time to deliver 
            <span style="color: var(--accent-green);">Institutional-Grade Conviction</span> directly to your terminal.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # CTA Buttons
        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("🚀 Get Started", type="primary", use_container_width=True):
                st.session_state['authenticated'] = True
                st.rerun()
        with b2:
            st.markdown("""
            <div style="display: flex; align-items: center; height: 100%; padding-left: 10px;">
                <span style="color: var(--text-muted); font-size: 0.9rem;">No credit card required for demo.</span>
            </div>
            """, unsafe_allow_html=True)

    with cols[1]:
        # LIVE PREVIEW DATA
        from theme import get_db
        db = get_db()
        recent_signals = db.get_recent_signals(limit=3)
        
        signal_html = ""
        for s in recent_signals:
            if s.get('signal') == 'WAIT': continue
            
            outcome = s.get('outcome', 'ACTIVE')
            color = "var(--accent-green)" if outcome == 'SUCCESS' else "var(--accent-red)" if outcome == 'FAIL' else "var(--accent-cyan)"
            bg_color = "rgba(0, 255, 136, 0.05)" if outcome == 'SUCCESS' else "rgba(255, 68, 102, 0.05)" if outcome == 'FAIL' else "rgba(0, 229, 255, 0.05)"
            
            status_text = "SUCCESS" if outcome == 'SUCCESS' else "FAILED" if outcome == 'FAIL' else "LIVE"
            time_str = "Recent" # Simple for landing
            
            price = s.get('price_at_signal')
            if price is None:
                price = 0.0
                
            signal_html += f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: {bg_color}; padding: 12px; border-radius: 12px; margin-bottom: 10px; border-left: 3px solid {color};">
                <div>
                    <div style="font-weight: 700; color: #FFFFFF;">{s['symbol']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">{s['signal']} @ {price:.5f}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {color}; font-weight: 700;">{status_text}</div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary);">{time_str}</div>
                </div>
            </div>
            """
        
        if not signal_html:
            signal_html = '<div style="text-align: center; color: var(--text-muted); padding: 40px;">Monitoring live institutional flows...</div>'

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)); 
            border: 1px solid rgba(255,255,255,0.1); 
            border-radius: 20px; 
            padding: 2px; 
            transform: perspective(1000px) rotateY(-10deg) rotateX(5deg);
            box-shadow: -20px 20px 60px rgba(0,0,0,0.5);
        ">
            <div style="background: #0a0e1a; border-radius: 18px; padding: 20px; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <span style="color: var(--text-secondary);">Real-Time Intelligence</span>
                    <span style="color: var(--accent-green);">● Active Scan</span>
                </div>
                {signal_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # FEATURES GRID
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 24px; border-radius: 16px; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 16px;">🧠</div>
            <h3 style="margin-bottom: 8px;">Adaptive AI Models</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Our Specialist Models retrain daily on new market data, ensuring they adapt to changing volatility and trends.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 24px; border-radius: 16px; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 16px;">🤖</div>
            <h3 style="margin-bottom: 8px;">Fully Automated</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Connect "Apex Connect" to your MT5 terminal and let the AI execute trades 24/7 while you sleep.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 24px; border-radius: 16px; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 16px;">📲</div>
            <h3 style="margin-bottom: 8px;">Instant Alerts</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Receive real-time notifications via Telegram with entry, stop-loss, and take-profit levels clearly defined.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # TRUST BANNER
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); opacity: 0.7;">
        <p style="text-transform: uppercase; letter-spacing: 0.2em; font-size: 0.8rem; margin-bottom: 20px;">Trusted Technoloiges</p>
        <div style="display: flex; justify-content: center; gap: 40px; font-weight: 700; font-size: 1.2rem; filter: grayscale(100%);">
            <span>METATRADER 5</span>
            <span>PYTHON</span>
            <span>TENSORFLOW</span>
            <span>OPENAI</span>
            <span>TWELVE DATA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
