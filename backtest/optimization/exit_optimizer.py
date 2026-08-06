"""RR-optimized exit mechanisms."""

import numpy as np
import pandas as pd
from risk.position_sizer import compute_atr
from reporting.metrics import _compute_sharpe


def _compute_rr(portfolio) -> float:
    """Compute RR ratio from portfolio trades using vbt 1.0 API."""
    try:
        pnl = np.array(portfolio.trades.pnl.values)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        return float(np.mean(wins) / abs(np.mean(losses)))
    except Exception:
        return 0.0


def split_position_exits(
    entries: pd.Series,
    df: pd.DataFrame,
    atr: pd.Series,
    partial_at_atr: float = 1.0,
    trail_atr: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    entry_prices = df["close"].where(entries).ffill()
    unrealized = df["close"] - entry_prices
    tp_exit = unrealized >= atr * partial_at_atr
    return tp_exit.shift(1).fillna(False), entries.copy()


def dynamic_target_exits(
    df: pd.DataFrame, lookback: int = 20
) -> tuple[pd.Series, pd.Series]:
    long_target = df["high"].rolling(lookback).max().shift(1)
    short_target = df["low"].rolling(lookback).min().shift(1)
    long_exits = df["close"] >= long_target
    short_exits = df["close"] <= short_target
    return long_exits.shift(1).fillna(False), short_exits.shift(1).fillna(False)


def time_exit(entries: pd.Series, max_bars: int = 10) -> pd.Series:
    exits = pd.Series(False, index=entries.index)
    entry_indices = entries[entries].index
    for idx in entry_indices:
        loc = entries.index.get_loc(idx)
        if loc + max_bars < len(entries):
            exits.iloc[loc + max_bars] = True
    return exits


class ExitOptimizer:
    def __init__(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        strategy: str,
        initial_capital: float = 10_000,
        commission: float = 0.0008,
    ) -> None:
        self.df = df
        self.entries = entries
        self.exits = exits
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self) -> pd.DataFrame:
        import vectorbt as vbt

        atr = compute_atr(self.df, 14)
        close = self.df["close"]

        configs = [
            {"name": "baseline", "tp_atr": 0, "trail_atr": 3.0, "time_bars": 0},
            {"name": "partial_1x", "tp_atr": 1.0, "trail_atr": 2.5, "time_bars": 0},
            {"name": "partial_1.5x", "tp_atr": 1.5, "trail_atr": 2.0, "time_bars": 0},
            {"name": "time_exit_10", "tp_atr": 0, "trail_atr": 3.0, "time_bars": 10},
            {"name": "time_exit_15", "tp_atr": 0, "trail_atr": 3.0, "time_bars": 15},
            {"name": "combined", "tp_atr": 1.0, "trail_atr": 2.5, "time_bars": 10},
        ]

        freq_map = {"15Min": "15min", "1H": "1h", "4H": "4h"}
        freq = "15min"
        for tf, f in freq_map.items():
            if tf in str(self.df.index.freq or ""):
                freq = f
                break

        results = []
        for cfg in configs:
            combined_exits = self.exits.copy()

            if cfg["tp_atr"] > 0:
                tp_exits, _ = split_position_exits(
                    self.entries, self.df, atr, cfg["tp_atr"], cfg["trail_atr"]
                )
                combined_exits = combined_exits | tp_exits

            if cfg["time_bars"] > 0:
                te = time_exit(self.entries, cfg["time_bars"])
                combined_exits = combined_exits | te

            try:
                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=self.entries,
                    exits=combined_exits,
                    init_cash=self.initial_capital,
                    fees=self.commission,
                    freq=freq,
                )

                trades = pf.trades.count()
                if trades < 10:
                    continue

                sharpe = _compute_sharpe(pf, 0.045)
                rr = _compute_rr(pf)

                results.append({
                    "config": cfg["name"],
                    "tp_atr": cfg["tp_atr"],
                    "trail_atr": cfg["trail_atr"],
                    "time_bars": cfg["time_bars"],
                    "sharpe": sharpe,
                    "rr_ratio": rr,
                    "win_rate": pf.trades.win_rate(),
                    "max_drawdown": pf.max_drawdown(),
                    "total_return": pf.total_return(),
                    "trades": trades,
                })
            except Exception:
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()
