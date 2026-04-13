import yaml
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mt5_connector import get_mt5

def print_table(data_list):
    if not data_list: return
    keys = data_list[0].keys()
    header = " | ".join([f"{str(k):<12}" for k in keys])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in data_list:
        print(" | ".join([f"{str(v):<12}" for v in row.values()]))

def run_risk_audit():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mt5_conf = config.get('mt5', {})
    trading_conf = config.get('trading', {})
    max_leverage = mt5_conf.get('max_trade_leverage', 30)
    risk_pct = mt5_conf.get('risk_value', 0.5)

    print(f"🛡️ Starting Multi-Pair Risk Audit ({risk_pct}% Balance Risk, {max_leverage}:1 Max Leverage)...")
    
    _mt5 = get_mt5()
    if not _mt5:
        print("❌ Failed to connect to MT5 Bridge.")
        return

    acc = _mt5.account_info()
    if not acc:
        print("⚠️ Connected to bridge, but no MT5 account active.")
        return

    balance = acc.balance
    risk_usd = balance * (risk_pct / 100.0)
    
    print(f"💰 Account Balance: ${balance:,.2f} | Risk Per Trade: ${risk_usd:,.2f}")
    
    results = []
    symbols = []
    for cat in ['majors', 'minors', 'crosses']:
        symbols.extend([p['symbol'] for p in config['currency_pairs'].get(cat, [])])

    print(f"📊 Auditing {len(symbols)} symbols...")

    for sym in symbols:
        info = mt5.symbol_info(sym)
        if not info:
             for s in [f"{sym}m", f"{sym}.", f"{sym}+", f"#{sym}"]:
                 info = mt5.symbol_info(s)
                 if info: break
        
        if not info:
            results.append({'Symbol': sym, 'Status': '❌ NOT FOUND'})
            continue

        # 1. Determine SL distance (Gold floor vs Forex standard)
        is_gold = 'XAU' in sym or 'GOLD' in sym
        pip_size = 0.01 if (is_gold or 'JPY' in sym) else 0.0001
        
        if is_gold:
            sl_pips = trading_conf.get('gold_min_sl_pips', 100)
        else:
            sl_pips = trading_conf.get('stop_loss_pips', 25)
            
        sl_dist = sl_pips * pip_size
        
        # 2. Risk-Based Lots
        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value
        dist_in_ticks = sl_dist / tick_size
        loss_per_lot = dist_in_ticks * tick_value
        
        if loss_per_lot > 0:
            risk_lots = risk_usd / loss_per_lot
            
            # Use current price (ask/bid mid)
            tick = _mt5.symbol_info_tick(info.name)
            price = (tick.ask + tick.bid) / 2 if tick else 1.0
            
            # 3. Margin-Limited Lots (Robust Check)
            margin_per_lot = _mt5.order_calc_margin(_mt5.ORDER_TYPE_BUY, info.name, 1.0, price)
            
            if not margin_per_lot:
                # Fallback to estimate if MT5 fails
                margin_per_lot = (price * info.trade_contract_size) / max_leverage
            
            # Max safe lots (using 90% of buying power)
            max_margin_lots = (balance * 0.9) / margin_per_lot
            
            # 4. Final Final Lots (Rule of Smallest)
            final_lots = min(risk_lots, max_margin_lots)
            
            # Normalize
            step = info.volume_step
            final_lots = max(info.volume_min, min(info.volume_max, round(final_lots / step) * step))
            
            # Leverage Calculation (Corrected):
            # Since Margin = Notional / Leverage, 
            # then Effective_Leverage = Notional / Balance = (Margin * Account_Leverage) / Balance
            # Wait, easier: Leverage = Notional_USD / Balance.
            # Notional_USD = Margin_USD * Account_Leverage
            
            actual_account_leverage = acc.leverage
            trade_margin_usd = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, info.name, final_lots, price)
            trade_notional_usd = trade_margin_usd * actual_account_leverage
            trade_leverage = trade_notional_usd / balance
            
            results.append({
                'Symbol': sym,
                'Lots': round(final_lots, 2),
                'Risk %': round((final_lots * loss_per_lot / balance) * 100, 2),
                'Leverage': round(trade_leverage, 1),
                'Margin $': round(trade_margin_usd, 2),
                'Status': 'SAFE' if trade_leverage <= max_leverage else 'MARGIN'
            })
        else:
            results.append({'Symbol': sym, 'Status': '⚠️ MATH ERROR'})

    
    print_table(results)
    
    import csv
    with open('risk_audit_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
    print("\n✅ Audit complete. Results saved to 'risk_audit_results.csv'.")

if __name__ == "__main__":
    run_risk_audit()
