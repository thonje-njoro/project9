"""Generate realistic synthetic OHLCV data for testing when API keys are unavailable."""

import numpy as np
import pandas as pd


def generate_synthetic_data(symbols: dict, config: dict) -> dict[str, pd.DataFrame]:
    """Generate synthetic 1-minute OHLCV data for each symbol."""
    np.random.seed(42)
    start = pd.Timestamp(config["start_date"], tz="UTC")
    end = pd.Timestamp(config["end_date"], tz="UTC")

    stock_base_prices = {"SPY": 470.0, "QQQ": 390.0, "GLD": 190.0, "USO": 75.0}
    crypto_base_prices = {"BTC/USD": 42000.0}

    annual_vol = {"SPY": 0.16, "QQQ": 0.22, "GLD": 0.15, "USO": 0.30, "BTC/USD": 0.65}
    drift = {"SPY": 0.12, "QQQ": 0.15, "GLD": 0.05, "USO": 0.03, "BTC/USD": 0.30}

    data = {}
    for symbol, info in symbols.items():
        print(f"Generating synthetic data for {symbol}...")
        is_crypto = info["asset_class"] == "crypto"

        if is_crypto:
            idx = pd.date_range(start, end, freq="1min", tz="UTC")
        else:
            idx = pd.bdate_range(start, end, tz="UTC")
            idx = idx.intersection(
                pd.date_range(start, end, freq="min", tz="UTC")
            )
            hours = idx.hour + idx.minute / 60.0
            mask = (hours >= 9.5) & (hours < 16.0)
            idx = idx[mask]
            if len(idx) == 0:
                idx = pd.date_range(start, end, freq="15min", tz="UTC")
                hours = idx.hour + idx.minute / 60.0
                mask = (hours >= 9.5) & (hours < 16.0)
                idx = idx[mask]

        n = len(idx)
        base = stock_base_prices.get(symbol, crypto_base_prices.get(symbol, 100.0))
        vol = annual_vol.get(symbol, 0.20)
        dr = drift.get(symbol, 0.10)

        dt = 1.0 / (252 * 390)
        vol_min = vol * np.sqrt(dt)
        dr_min = (dr - 0.5 * vol**2) * dt

        log_returns = np.random.normal(dr_min, vol_min, n)
        log_prices = np.log(base) + np.cumsum(log_returns)
        close = np.exp(log_prices)

        noise_h = np.abs(np.random.normal(0, vol_min * 0.3, n))
        noise_l = np.abs(np.random.normal(0, vol_min * 0.3, n))
        high = close * (1 + noise_h)
        low = close * (1 - noise_l)
        open_ = np.roll(close, 1)
        open_[0] = base

        if is_crypto:
            base_vol = 500.0
        elif symbol in ("SPY", "QQQ"):
            base_vol = 80_000_000.0 / 390.0
        else:
            base_vol = 5_000_000.0 / 390.0

        volume = np.random.lognormal(np.log(base_vol), 0.5, n)

        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": volume,
        }, index=idx[:n])

        data[symbol] = df

    return data
