"""Verify TwelveData API key."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TWELVEDATA_API_KEY")
print(f"Testing API Key: {api_key}")

if not api_key:
    print("No API Key found!")
    exit(1)

symbol = "EUR/USD"
interval = "1h"
url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=5"

try:
    response = requests.get(url)
    data = response.json()
    
    if "code" in data and data["code"] != 200:
        print(f"Error: {data}")
    elif "values" in data:
        print(f"Success! Retrieved {len(data['values'])} candles.")
        print(f"Last candle: {data['values'][0]}")
    else:
        print(f"Unexpected response: {data}")

except Exception as e:
    print(f"Request failed: {e}")
