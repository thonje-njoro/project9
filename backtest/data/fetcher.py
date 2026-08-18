"""
data/fetcher.py — Alpaca + LSE fetch with Parquet cache + synthetic fallback.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, LSE_API_KEY,
    DATA_CONFIG, INSTRUMENTS,
)

logger = logging.getLogger(__name__)

# ─── Parquet Cache ───────────────────────────────────────────────────────────

def _cache_path(symbol: str, timeframe: str) -> Path:
    """Return cache file path: data/cache/<SYMBOL>_<tf>.parquet"""
    cache_dir = Path(DATA_CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{timeframe}.parquet"


def _cache_is_fresh(path: Path, hours: int = 24) -> bool:
    """Check if cache file exists and is fresher than `hours`."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=hours)


def _load_cache(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load cached Parquet if fresh enough."""
    path = _cache_path(symbol, timeframe)
    if _cache_is_fresh(path, DATA_CONFIG["cache_freshness_hours"]):
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                logger.info(f"Cache hit: {symbol} {timeframe} ({len(df)} bars)")
                return df
        except Exception as e:
            logger.warning(f"Cache read failed for {path}: {e}")
    return None


def _save_cache(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save DataFrame to Parquet cache."""
    if df.empty:
        return
    path = _cache_path(symbol, timeframe)
    try:
        df.to_parquet(path, index=True)
        logger.info(f"Cached {len(df)} bars to {path}")
    except Exception as e:
        logger.warning(f"Cache write failed for {path}: {e}")


# ─── Alpaca Fetcher ──────────────────────────────────────────────────────────

def _get_alpaca_client():
    """Get Alpaca historical data client. Returns None if keys missing."""
    if not ALPACA_API_KEY or ALPACA_API_KEY == "your_alpaca_api_key_here":
        logger.warning("Alpaca API keys not configured")
        return None
    try:
        from alpaca.data import StockHistoricalDataClient
        return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except ImportError:
        logger.warning("alpaca-py not installed")
        return None
    except Exception as e:
        logger.warning(f"Alpaca client init failed: {e}")
        return None


def _alpaca_timeframe(tf: str):
    """Convert string timeframe to Alpaca TimeFrame enum."""
    from alpaca.data import TimeFrame, TimeFrameUnit
    mapping = {
        "1min": TimeFrame.Minute,
        "5min": TimeFrame(5, TimeFrameUnit.Minute),
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
        "1h": TimeFrame.Hour,
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame.Day,
    }
    return mapping.get(tf, TimeFrame.Minute)


def fetch_alpaca(symbol: str, timeframe: str,
                 start: str = "2022-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca IEX feed.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC).
    """
    cached = _load_cache(symbol, timeframe)
    if cached is not None:
        return cached

    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data import TimeFrame

        tf = _alpaca_timeframe(timeframe)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
            feed="iex",
        )
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning(f"No Alpaca data for {symbol} {timeframe}")
            return df

        # Ensure standard columns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        _save_cache(df, symbol, timeframe)
        return df

    except Exception as e:
        logger.error(f"Alpaca fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── LSE Fetcher ─────────────────────────────────────────────────────────────

_last_lse_call = 0.0


def fetch_lse(symbol: str, timeframe: str,
              start: str = "2022-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch OHLCV bars from London Strategic Edge API.
    Rate-limited to 1 call per 7 seconds.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC).
    """
    global _last_lse_call

    cached = _load_cache(symbol, timeframe)
    if cached is not None:
        return cached

    if not LSE_API_KEY or LSE_API_KEY == "your_lse_api_key_here":
        logger.warning("LSE API key not configured")
        return pd.DataFrame()

    # Rate limit: 7 seconds between calls
    elapsed = time.time() - _last_lse_call
    if elapsed < DATA_CONFIG["lse_rate_limit_seconds"]:
        wait = DATA_CONFIG["lse_rate_limit_seconds"] - elapsed
        logger.info(f"LSE rate limit: waiting {wait:.1f}s")
        time.sleep(wait)

    try:
        import requests

        # Map symbol to LSE format (e.g., XAUUSD_MR -> XAUUSD)
        lse_symbol = symbol.replace("_MR", "").replace("_", "")

        tf_map = {"1min": "1min", "5min": "5min", "15min": "15min",
                   "1h": "1h", "4h": "4h", "1d": "1d"}
        lse_tf = tf_map.get(timeframe, "1h")

        url = DATA_CONFIG["lse_base_url"]
        params = {
            "symbol": lse_symbol,
            "from": start,
            "to": end,
            "timeframe": lse_tf,
        }
        headers = {"Authorization": f"Bearer {LSE_API_KEY}"}

        _last_lse_call = time.time()
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data or "data" not in data:
            logger.warning(f"No LSE data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        _save_cache(df, symbol, timeframe)
        return df

    except Exception as e:
        logger.error(f"LSE fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── yfinance Fallback ───────────────────────────────────────────────────────

def fetch_yfinance(symbol: str, start: str = "2022-01-01",
                   end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch daily bars from yfinance (fallback only)."""
    cached = _load_cache(symbol, "1d")
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty:
            return df

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()

        _save_cache(df, symbol, "1d")
        return df

    except Exception as e:
        logger.error(f"yfinance fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── Main Fetch Dispatcher ───────────────────────────────────────────────────

def fetch(symbol: str, timeframe: str,
          start: str = "2022-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch data for a symbol using the appropriate source.
    Falls back to synthetic data if all sources fail.
    """
    instrument = INSTRUMENTS.get(symbol, {})
    source = instrument.get("data_source", "alpaca")

    df = pd.DataFrame()

    if source == "lse":
        df = fetch_lse(symbol, timeframe, start, end)
    elif source == "alpaca":
        # For CPER_GLD, we need both CPER and GLD
        if symbol == "CPER_GLD":
            df_cper = fetch_alpaca("CPER", timeframe, start, end)
            df_gld = fetch_alpaca("GLD", timeframe, start, end)
            if not df_cper.empty and not df_gld.empty:
                # Align on common index
                common_idx = df_cper.index.intersection(df_gld.index)
                if len(common_idx) > 0:
                    df = pd.DataFrame(index=common_idx)
                    df["open"] = df_cper.loc[common_idx, "open"] / df_gld.loc[common_idx, "open"]
                    # Max ratio = CPER_high / GLD_low, Min ratio = CPER_low / GLD_high
                    df["high"] = df_cper.loc[common_idx, "high"] / df_gld.loc[common_idx, "low"]
                    df["low"] = df_cper.loc[common_idx, "low"] / df_gld.loc[common_idx, "high"]
                    df["close"] = df_cper.loc[common_idx, "close"] / df_gld.loc[common_idx, "close"]
                    df["volume"] = df_cper.loc[common_idx, "volume"] + df_gld.loc[common_idx, "volume"]
                    _save_cache(df, symbol, timeframe)
        else:
            df = fetch_alpaca(symbol, timeframe, start, end)

    # Fallback: yfinance for daily data
    if df.empty and timeframe == "1d":
        df = fetch_yfinance(symbol, start, end)

    # Final fallback: synthetic data
    if df.empty:
        from backtest.data.synthetic import generate_synthetic
        logger.warning(f"Using synthetic data for {symbol} — do not use for live trading")
        df = generate_synthetic(symbol)

    return df
