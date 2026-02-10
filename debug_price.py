
import yfinance as yf
from datetime import datetime

symbol = "AUDUSD=X"
ticker = yf.Ticker(symbol)

print(f"--- Debugging {symbol} at {datetime.now()} ---")

# 1. Fetch History (1h)
df = ticker.history(period="1d", interval="1h")
print(f"\n[History 1h] Last Rows:")
print(df.tail(3)[['Open', 'High', 'Low', 'Close', 'Volume']])

if not df.empty:
    last_close = df['Close'].iloc[-1]
    last_time = df.index[-1]
    print(f"\nDataFrame Last Close: {last_close} at {last_time}")

# 2. Check Fast Info
try:
    fast_price = ticker.fast_info['last_price']
    print(f"\n[Fast Info] Last Price: {fast_price}")
except Exception as e:
    print(f"\n[Fast Info] Error: {e}")

# 3. Check Regular Info
try:
    # forced fetch
    info = ticker.info 
    regular_price = info.get('regularMarketPrice')
    bid = info.get('bid')
    ask = info.get('ask')
    print(f"\n[Info] RegularPrice: {regular_price}")
    print(f"[Info] Bid: {bid}, Ask: {ask}")
except Exception as e:
    print(f"\n[Info] Error: {e}")
