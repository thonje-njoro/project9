"""Pre-fetch all data for the backtest (builds cache)."""
import os, sys, time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

api_key = os.getenv("ALPACA_API_KEY", "")
secret = os.getenv("ALPACA_SECRET_KEY", "")

from config import INSTRUMENTS, BACKTEST_CONFIG
from data.fetcher import DataFetcher

stock_syms = [s for s, v in INSTRUMENTS.items() if v["asset_class"] == "stock"]
crypto_syms = [s for s, v in INSTRUMENTS.items() if v["asset_class"] == "crypto"]
forex_syms = [s for s, v in INSTRUMENTS.items() if v["asset_class"] == "forex"]

print(f"Stocks ({len(stock_syms)}): {stock_syms}")
print(f"Crypto ({len(crypto_syms)}): {crypto_syms}")
print(f"Forex ({len(forex_syms)}): {forex_syms}")
sys.stdout.flush()

fetcher = DataFetcher(api_key, secret)
for sym in stock_syms + crypto_syms:
    ac = "crypto" if sym in crypto_syms else "stock"
    print(f"Fetching {sym} ({ac})...", end=" ")
    sys.stdout.flush()
    try:
        df = fetcher.fetch(sym, ac, BACKTEST_CONFIG["start_date"], BACKTEST_CONFIG["end_date"])
        print(f"{len(df)} bars")
    except Exception as e:
        print(f"FAILED: {e}")
    sys.stdout.flush()

# Forex via yfinance
import yfinance as yf
mapping = {"XAU/USD": "GC=F"}
for sym in forex_syms:
    yf_sym = mapping.get(sym, sym)
    print(f"Fetching {sym} via yfinance ({yf_sym})...", end=" ")
    sys.stdout.flush()
    try:
        df = yf.download(yf_sym, start=BACKTEST_CONFIG["start_date"],
                         end=BACKTEST_CONFIG["end_date"], progress=False)
        print(f"{len(df)} days" if df is not None and not df.empty else "NO DATA")
    except Exception as e:
        print(f"FAILED: {e}")
    sys.stdout.flush()

print("\nData fetch complete")
