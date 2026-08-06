"""Market regime classification."""

import numpy as np
import pandas as pd


def detect_regime(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high, low = df["high"], df["low"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / 14
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    sma_200 = close.rolling(200).mean()
    ret_5d = close.pct_change(5)
    vol_5d = close.pct_change().rolling(5).std()
    vol_60d_avg = close.pct_change().rolling(60).std().rolling(60).mean()

    regimes = pd.Series("ranging", index=df.index, dtype="object")

    high_vol_chaos = vol_5d > vol_60d_avg * 2
    trending_up = (close > sma_200) & (adx > 25) & (ret_5d > 0)
    trending_down = (close < sma_200) & (adx > 25) & (ret_5d < 0)

    regimes[high_vol_chaos] = "high_vol_chaos"
    regimes[trending_up & ~high_vol_chaos] = "trending_up"
    regimes[trending_down & ~high_vol_chaos] = "trending_down"

    return regimes


def regime_performance_report(
    entries: pd.Series, exits: pd.Series, df: pd.DataFrame, regimes: pd.Series
) -> pd.DataFrame:
    entry_bars = entries[entries].index
    if len(entry_bars) == 0:
        return pd.DataFrame()

    results = []
    for regime in ["trending_up", "trending_down", "ranging", "high_vol_chaos"]:
        regime_mask = regimes == regime
        regime_entries = entries & regime_mask
        regime_exits = exits & regime_mask

        n_entries = regime_entries.sum()
        if n_entries == 0:
            results.append({
                "regime": regime,
                "trades": 0,
                "win_rate": 0.0,
                "rr_ratio": 0.0,
                "return_pct": 0.0,
            })
            continue

        entry_prices = df["close"].where(regime_entries).dropna()
        exit_prices = df["close"].where(regime_exits).reindex(entry_prices.index, method="ffill")

        wins = (exit_prices > entry_prices).sum()
        win_rate = wins / len(entry_prices) if len(entry_prices) > 0 else 0

        pnl = (exit_prices - entry_prices)
        avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
        avg_loss = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else 1
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        total_return = (pnl.sum() / entry_prices.sum()) * 100

        results.append({
            "regime": regime,
            "trades": int(n_entries),
            "win_rate": round(win_rate, 3),
            "rr_ratio": round(rr_ratio, 2),
            "return_pct": round(total_return, 2),
        })

    return pd.DataFrame(results)
