import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5
from scripts.apex_connect import load_config

def debug():
    # 2. Connect to MT5
    mt5 = get_mt5()
    if not mt5:
        print("❌ Failed to connect to MT5 Bridge.")
        return

    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info('GOLD')
    
    print(f"Balance: {account_info.balance}")
    print(f"Risk: 0.5% -> ${account_info.balance * 0.005}")
    print(f"Tick Size: {symbol_info.trade_tick_size}")
    print(f"Tick Value: {symbol_info.trade_tick_value}")
    print(f"Contract Size: {symbol_info.trade_contract_size}")
    
    entry = 4726.1
    sl = 4625.7
    price_dist = abs(entry - sl)
    dist_ticks = price_dist / symbol_info.trade_tick_size
    loss_lot = dist_ticks * symbol_info.trade_tick_value
    
    print(f"Price Dist: {price_dist}")
    print(f"Dist Ticks: {dist_ticks}")
    print(f"Loss per Lot: ${loss_lot}")
    
    risk_lots = (account_info.balance * 0.005) / loss_lot
    print(f"Calculated Risk Lots: {risk_lots}")
    print(f"Volume Min: {symbol_info.volume_min}")
    print(f"Volume Step: {symbol_info.volume_step}")
    

if __name__ == "__main__":
    debug()
