"""
DataFetcher — Fetches OHLCV data from Alpaca free tier.

Known limitations of free Alpaca tier:
1. IEX feed (free tier) has lower liquidity than SIP (paid).
   For SPY and QQQ, IEX data is reliable. For GLD and USO,
   IEX coverage is thinner — volume data may be incomplete.

2. Free tier historical data limit: 5 years for stocks,
   unlimited for crypto. Do not request data before 2020-01-01
   for stocks to be safe.

3. Rate limits: 200 requests/minute. With 1-minute bars over
   12 months, each symbol requires roughly 100k rows — typically
   2–3 API requests per symbol. Well within limits.

4. IEX does not include pre-market or after-hours data.
   All bars are regular session only (09:30–16:00 ET).
   This is fine for these strategies.

5. Crypto data (BTC/USD) from Alpaca free tier is sourced
   from multiple venues and is reliable for backtesting purposes.

6. GLD and USO are ETFs, not futures. They do not perfectly
   track spot gold and crude oil. Slippage assumptions may
   underestimate real-world costs for these instruments.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


class DataFetcher:
    """Fetches OHLCV data from Alpaca free tier with caching."""

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_client = CryptoHistoricalDataClient()
        self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        safe_sym = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_sym}_{start}_{end}.parquet"

    def _fetch_stock(self, symbol: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
        cache = self._cache_path(symbol, start, end)
        if cache.exists() and not force_refresh:
            print(f"Loading {symbol} from cache...")
            df = pd.read_parquet(cache)
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            return df

        print(f"Fetching {symbol} from Alpaca (IEX feed)...")
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1),
            feed="iex",
            adjustment="all",
        )
        bars = self.stock_client.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC")
        df.to_parquet(cache)
        return df

    def _fetch_crypto(self, symbol: str, start: str, end: str, force_refresh: bool = False, timeframe: str = "1Min") -> pd.DataFrame:
        cache = self._cache_path(symbol, start, end)
        # Check if a cached file with this exact symbol/start/end exists
        # (we append timeframe to avoid collisions)
        cache_tf = self._cache_path(f"{symbol}_{timeframe}", start, end)
        if cache_tf.exists() and not force_refresh:
            print(f"Loading {symbol} from cache...")
            df = pd.read_parquet(cache_tf)
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            return df

        print(f"Fetching {symbol} from Alpaca (crypto, {timeframe})...")
        tf_map = {"1Min": TimeFrame.Minute, "1h": TimeFrame.Hour, "1D": TimeFrame.Day}
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, TimeFrame.Minute),
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1),
        )
        bars = self.crypto_client.get_crypto_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC")
        df.to_parquet(cache_tf)
        return df

    def fetch(self, symbol: str, asset_class: str, start: str, end: str, force_refresh: bool = False,
              timeframe: str = "1Min") -> pd.DataFrame:
        for attempt in range(2):
            try:
                if asset_class == "crypto":
                    return self._fetch_crypto(symbol, start, end, force_refresh, timeframe)
                else:
                    return self._fetch_stock(symbol, start, end, force_refresh)
            except Exception as e:
                if attempt == 0:
                    print(f"  Attempt 1 failed for {symbol}: {e}. Retrying in 5s...")
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to fetch {symbol} after 2 attempts: {e}")

    def fetch_all(self, instruments: dict, config: dict, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for i, (symbol, info) in enumerate(instruments.items()):
            # Use the base_tf from config to determine fetch granularity
            base_tf = info.get("base_tf", "1Min")
            df = self.fetch(
                symbol,
                info["asset_class"],
                config["start_date"],
                config["end_date"],
                force_refresh,
                timeframe=base_tf,
            )
            data[symbol] = df
            print(f"  {symbol}: {len(df)} bars loaded")
            if i < len(instruments) - 1:
                time.sleep(0.3)
        return data
