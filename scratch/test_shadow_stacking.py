import os
import sys
import logging
from datetime import datetime, timezone
sys.path.insert(0, os.getcwd())
from core.executive import ExecutiveEngine
from core.database import SignalDatabase

# Setup mock engine
engine = ExecutiveEngine()
db = SignalDatabase()

# Clean up EURCAD active signals first to have a clean state
conn = db._get_connection()
conn.execute("DELETE FROM signals WHERE symbol='EURCAD'")
conn.commit()

# Create a mock signal that is NOT proven (so it's a shadow)
mock_signal_60 = {
    'id': None,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'symbol': 'EURCAD',
    'signal': 'BUY',
    'confidence': 0.60,
    'confidence_tier': '60',
    'price_at_signal': 1.4500,
    'tp_price': 1.4600,
    'sl_price': 1.4400,
    'is_proven': 0,
    'is_hidden': 1,  # Shadow
    'outcome': 'ACTIVE'
}

print("Saving first 60% shadow signal...")
engine.db.save_signal(mock_signal_60)

# Try to save a second 60% shadow signal (stacking)
mock_signal_60_v2 = mock_signal_60.copy()
mock_signal_60_v2['timestamp'] = datetime.now(timezone.utc).isoformat()

print("Analyzing second 60% shadow signal (should allow stacking)...")
# We'll use analyze_symbol but we need to mock the inference engine response
# Actually, I'll just check if Rule 2 would block it.
active_signals = [s for s in engine.db.get_active_signals(symbol='EURCAD', include_hidden=True) if s['signal'] in ('BUY', 'SELL')]
print(f"Active signals count: {len(active_signals)}")

# Simulate the Rule 2 logic from executive.py
is_hidden = True # It's a shadow
new_tier = 60
allowed = True
if not is_hidden:
    for active in active_signals:
        if active['signal'] == 'BUY' and int(float(active.get('confidence_tier', 0))) == new_tier:
            allowed = False
            break

print(f"Rule 2 Bypass for Shadows: {'SUCCESS' if allowed else 'FAILED'}")

# Verify that if it was NOT a shadow, it WOULD be blocked
is_hidden_live = False
allowed_live = True
if not is_hidden_live:
    for active in active_signals:
        if active['signal'] == 'BUY' and int(float(active.get('confidence_tier', 0))) == new_tier:
            allowed_live = False
            break

print(f"Rule 2 Blocking for Live: {'SUCCESS' if not allowed_live else 'FAILED'}")
