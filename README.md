# ⚡ ApexForex SaaS

Institutional-grade AI Trading Intelligence platform for Forex markets.

## 🚀 Features

- **AI Sentinel Surveillance**: Real-time scanning across 31 forex pairs with 90%+ precision targeting.
- **Premium Glassmorphic Dashboard**: A high-performance UI/UX built with modern Streamlit, featuring real-time telemetry and advanced analytics.
- **AI Specialist Models**: Custom-trained models for each currency pair, optimized for high win rates.
- **Signal Lifecycle Management**: Automatic signal deduplication, TP/SL resolution, and automated Telegram notifications.
- **Trading Terminal**: Advanced candlestick charting with AI verdict indicators and trading level overlays.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Modern Multi-Page Architecture)
- **Backend**: Python, SQLite
- **AI/ML**: TensorFlow/Keras, Scikit-learn, XGBoost
- **Data**: TwelveData API Integration
- **DevOps**: Git, Python Virtual Environments

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/ApexForexSaaS.git
   cd ApexForexSaaS
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   - Create a `.env` file based on `.env.example`.
   - Add your TwelveData and Telegram API keys.

4. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 🛡️ License

© 2026 ApexForex. All rights reserved.
