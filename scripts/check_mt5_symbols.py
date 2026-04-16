import MetaTrader5 as mt5
import sys

def check_symbols():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    # 1. Search for GOLD and XAU patterns
    print("--- Detailed XAU/GOLD Search ---")
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        print("No symbols found.")
        mt5.shutdown()
        return

    found = []
    for s in all_symbols:
        name = s.name.upper()
        if "GOLD" in name or "XAU" in name or "SPX" in name or "USA500" in name:
            tick = mt5.symbol_info_tick(s.name)
            price = tick.bid if tick else "N/A"
            found.append((s.name, price, s.path))

    # Sort by name length to find the "purest" symbol
    found.sort(key=lambda x: len(x[0]))
    
    for name, price, path in found:
        print(f"  {name:15} | Bid: {str(price):10} | Path: {path}")

    mt5.shutdown()

if __name__ == "__main__":
    check_symbols()
