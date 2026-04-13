import sys
import os
from pathlib import Path

# Fix module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5

print("🔍 Checking MT5 Bridge connection (mt5linux)...")
_mt5 = get_mt5()

if _mt5 is None:
    print("❌ FAILED to link to MT5 Bridge.")
    print("Ensure the bridge is running at 127.0.0.1:18812")
    sys.exit(1)

print("✅ SUCCESS: MT5 Bridge Linked.")
account = _mt5.account_info()

if account:
    print(f"📊 Connected Server: {account.server}")
    print(f"👤 Account: #{account.login}")
    print(f"💰 Balance: {account.balance} {account.currency}")
else:
    print("⚠️ Linked to bridge, but no MT5 login session active.")
    print("Check your credentials in config.yaml.")

# No shutdown call needed, handled by singleton eventually
