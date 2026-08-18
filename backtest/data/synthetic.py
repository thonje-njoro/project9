"""
data/synthetic.py — Fallback synthetic OHLCV data via geometric Brownian motion.
Used when Alpaca/LSE API keys are missing or fetch fails.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Realistic starting prices for known instruments
SYNTHETIC_PRICES = {
    "GLD": 175.0, "TLT": 100.0, "IWM": 190.0, "CPER": 25.0,
    "CPER_GLD": 0.14, "SPY": 400.0, "IWM_VWAP": 190.0,
    "SPY_ORB": 400.0, "QQQ_ORB": 350.0,
    "NVDA_MORB": 450.0, "AMD_MORB": 120.0, "PLTR_MORB": 20.0, "MRVL_MORB": 60.0,
    "XAUUSD_MR": 1950.0,
}

# Volatility scaling by asset class
SYNTHETIC_VOL = {
    "equity_etf": 0.015,
    "bond_etf": 0.008,
    "equity": 0.025,
    "forex": 0.008,
    "synthetic": 0.012,
}


def generate_synthetic(symbol: str, n_bars: int = 10_000,
                       mu: float = 0.0002, sigma: float | None = None,
                       initial_price: float | None = None,
                       timeframe: str = "15min") -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data using geometric Brownian motion.

    Args:
        symbol: Instrument symbol (used for price/vol lookup).
        n_bars: Number of bars to generate.
        mu: Drift per bar (annualized ~5% for daily, scaled by bar frequency).
        sigma: Volatility per bar. If None, uses asset-class default.
        initial_price: Starting price. If None, uses known price or 100.
        timeframe: Bar frequency for date index generation.

    Returns:
        DataFrame with columns: open, high, low, close, volume
        DatetimeIndex starting 2022-01-01.
    """
    from backtest.config import INSTRUMENTS

    if initial_price is None:
        initial_price = SYNTHETIC_PRICES.get(symbol, 100.0)

    if sigma is None:
        asset_class = INSTRUMENTS.get(symbol, {}).get("asset_class", "equity_etf")
        sigma = SYNTHETIC_VOL.get(asset_class, 0.015)

    logger.warning(f"Generating synthetic data for {symbol} ({n_bars} bars) — NOT for live trading")

    np.random.seed(hash(symbol) % (2**31))

    # Generate log returns
    log_returns = np.random.normal(mu, sigma, n_bars)

    # Add mean-reversion component for realism
    prices = np.zeros(n_bars)
    prices[0] = initial_price
    for i in range(1, n_bars):
        # Mild mean reversion toward initial price
        mean_rev = -0.001 * (prices[i-1] / initial_price - 1)
        ret = log_returns[i] + mean_rev
        prices[i] = prices[i-1] * np.exp(ret)

    # Generate OHLC from close prices
    opens = prices * (1 + np.random.normal(0, sigma * 0.1, n_bars))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, sigma * 0.3, n_bars)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, sigma * 0.3, n_bars)))

    # Volume: lognormal with autocorrelation
    base_vol = 1_000_000 if "SY" in symbol or "QQQ" in symbol else 500_000
    log_vol = np.random.normal(np.log(base_vol), 0.5, n_bars)
    # Add volume autocorrelation
    for i in range(1, n_bars):
        log_vol[i] = 0.9 * log_vol[i-1] + 0.1 * log_vol[i]
    volumes = np.exp(log_vol)

    # Build date index
    tf_minutes = {"1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240, "1d": 1440}
    freq_min = tf_minutes.get(timeframe, 15)
    dates = pd.date_range(
        start="2022-01-01", periods=n_bars,
        freq=f"{freq_min}min", tz="UTC",
    )

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df
