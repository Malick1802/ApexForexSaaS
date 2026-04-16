import MetaTrader5 as mt5

def check():
    if not mt5.initialize():
        print("Init failed")
        return
    
    for s in ["EURUSD", "GBPUSD", "GOLD", "XAUUSD", "SPX500", "USA500"]:
        t = mt5.symbol_info_tick(s)
        if t:
            print(f"{s:10} | Bid: {t.bid}")
        else:
            print(f"{s:10} | Not Found")
    
    mt5.shutdown()

if __name__ == "__main__":
    check()
