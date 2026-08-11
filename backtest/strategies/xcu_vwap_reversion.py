"""
XCU VWAP Reversion Strategy — Copper
=====================================
The ONLY commodity strategy that survived walk-forward validation.

Results:
- Win Rate: 73.08% (≥65% ✅)
- Profit Factor: 1.589 (≥1.5 ✅)
- Sharpe Ratio: 2.674 (≥0.5 ✅)
- Max Drawdown: 3.96% (≤15% ✅)
- Walk-Forward: ROBUST (2/3 windows)

Parameters: VWAP(20), entry at 1σ, exit at VWAP
Logic: Buy when price < VWAP - 1σ, sell when price > VWAP

Critical Risks:
1. Slippage kills edge — breakeven at just 5 bps
2. 2-year losing period (2023-2024)
3. Bear market weakness (PF drops to 1.19)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_vwap(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Calculate VWAP (Volume Weighted Average Price) with bands.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback : int
        VWAP lookback period (default: 20)
    
    Returns
    -------
    DataFrame with VWAP and bands added
    """
    df = df.copy()
    
    # Calculate typical price
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    
    # Use volume if available, otherwise use price range as proxy
    if "volume" in df.columns:
        volume = df["volume"]
    else:
        volume = (df["high"] - df["low"]) / df["close"] * 100  # Range as volume proxy
    
    # Calculate VWAP
    cum_tp_vol = (typical_price * volume).rolling(lookback).sum()
    cum_vol = volume.rolling(lookback).sum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    
    # Calculate VWAP bands (standard deviation)
    deviation = df["close"] - df["vwap"]
    df["vwap_std"] = deviation.rolling(lookback).std()
    df["vwap_upper"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower"] = df["vwap"] - df["vwap_std"]
    
    # Calculate z-score
    df["vwap_zscore"] = np.where(
        df["vwap_std"] > 0,
        deviation / df["vwap_std"],
        0
    )
    
    return df


def generate_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_mult: float = 1.0,
    exit_at_vwap: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate XCU VWAP Reversion signals.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback : int
        VWAP lookback period (default: 20)
    entry_mult : float
        Entry multiplier for VWAP bands (default: 1.0)
    exit_at_vwap : bool
        Exit when price returns to VWAP (default: True)
    
    Returns
    -------
    Tuple of (long_entries, long_exits, short_entries, short_exits)
    """
    # Calculate VWAP
    df = calculate_vwap(df, lookback)
    
    close = df["close"].values
    vwap = df["vwap"].values
    upper = df["vwap"].values + entry_mult * df["vwap_std"].values
    lower = df["vwap"].values - entry_mult * df["vwap_std"].values
    zscore = df["vwap_zscore"].values
    
    n = len(close)
    long_entries = np.zeros(n, dtype=bool)
    long_exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    
    in_trade = False
    direction = 0
    
    for i in range(1, n):
        if np.isnan(vwap[i]) or np.isnan(zscore[i]):
            continue
        
        if not in_trade:
            # Long entry: price below lower band (oversold)
            if close[i] < lower[i]:
                long_entries[i] = True
                in_trade = True
                direction = 1
            # Short entry: price above upper band (overbought)
            elif close[i] > upper[i]:
                short_entries[i] = True
                in_trade = True
                direction = -1
        else:
            # Exit when price returns to VWAP
            if exit_at_vwap:
                if direction == 1 and close[i] >= vwap[i]:
                    long_exits[i] = True
                    in_trade = False
                    direction = 0
                elif direction == -1 and close[i] <= vwap[i]:
                    short_exits[i] = True
                    in_trade = False
                    direction = 0
    
    return (
        pd.Series(long_entries, index=df.index),
        pd.Series(long_exits, index=df.index),
        pd.Series(short_entries, index=df.index),
        pd.Series(short_exits, index=df.index),
    )


def backtest_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_mult: float = 1.0,
    exit_at_vwap: bool = True,
    commission: float = 0.0005,
    slippage: float = 0.0005,
) -> dict:
    """
    Run backtest for XCU VWAP Reversion strategy.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback : int
        VWAP lookback period
    entry_mult : float
        Entry multiplier for VWAP bands
    exit_at_vwap : bool
        Exit when price returns to VWAP
    commission : float
        Commission per trade (default: 0.05%)
    slippage : float
        Slippage per trade (default: 0.05%)
    
    Returns
    -------
    dict with backtest metrics
    """
    # Generate signals
    long_entries, long_exits, short_entries, short_exits = generate_signals(
        df, lookback, entry_mult, exit_at_vwap
    )
    
    close = df["close"].values
    trades = []
    in_trade = False
    direction = 0
    entry_price = 0
    entry_idx = 0
    
    for i in range(len(close)):
        if not in_trade:
            if long_entries.iloc[i]:
                in_trade = True
                direction = 1
                entry_price = close[i] * (1 + slippage)  # Slippage on entry
                entry_idx = i
            elif short_entries.iloc[i]:
                in_trade = True
                direction = -1
                entry_price = close[i] * (1 - slippage)  # Slippage on entry
                entry_idx = i
        else:
            if long_exits.iloc[i] or short_exits.iloc[i]:
                exit_price = close[i] * (1 - slippage)  # Slippage on exit
                pnl = (exit_price - entry_price) / entry_price * direction
                pnl -= commission * 2  # Commission on both sides
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "bars_held": i - entry_idx,
                })
                in_trade = False
                direction = 0
    
    if not trades:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0, "sharpe": 0, "max_dd": 0, "total_return": 0}
    
    # Calculate metrics
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    win_rate = len(wins) / len(pnls) * 100
    total_return = sum(pnls) * 100
    
    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss
    
    # Sharpe ratio
    pnl_std = np.std(pnls) if len(pnls) > 1 else 0.001
    sharpe = np.mean(pnls) / pnl_std * np.sqrt(252) if pnl_std > 0 else 0
    
    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = (cum_pnl - peak)
    max_dd = abs(min(drawdown)) * 100 if len(drawdown) > 0 else 0
    
    return {
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "total_return": round(total_return, 2),
        "avg_pnl": round(np.mean(pnls) * 100, 3),
        "avg_win": round(np.mean(wins) * 100, 3) if wins else 0,
        "avg_loss": round(np.mean(losses) * 100, 3) if losses else 0,
    }


# Preset parameters from MiMo Claw optimization
XCU_VWAP_PARAMS = {
    "lookback": 20,
    "entry_mult": 1.0,
    "exit_at_vwap": True,
    "commission": 0.0005,
    "slippage": 0.0005,
}


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/admin1/project9/backtest")
    
    # Load copper data
    DATA = "/mnt/c/Users/Admin/project9/data/commodities"
    df = pd.read_parquet(f"{DATA}/XCUUSD_1d.parquet")
    
    print("=" * 70)
    print("XCU VWAP REVERSION STRATEGY — BACKTEST")
    print("=" * 70)
    
    # Run backtest with optimal parameters
    result = backtest_strategy(df, **XCU_VWAP_PARAMS)
    
    print(f"\nResults:")
    print(f"  Trades: {result['trades']}")
    print(f"  Win Rate: {result['win_rate']}%")
    print(f"  Profit Factor: {result['profit_factor']}")
    print(f"  Sharpe Ratio: {result['sharpe']}")
    print(f"  Max Drawdown: {result['max_dd']}%")
    print(f"  Total Return: {result['total_return']}%")
    print(f"  Avg PnL/Trade: {result['avg_pnl']}%")
    
    # Check acceptance criteria
    checks = {
        "WR >= 65%": result["win_rate"] >= 65,
        "PF >= 1.5": result["profit_factor"] >= 1.5,
        "Sharpe >= 0.5": result["sharpe"] >= 0.5,
        "DD <= 15%": result["max_dd"] <= 15,
    }
    
    print(f"\nAcceptance Criteria:")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    all_pass = all(checks.values())
    print(f"\nOverall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    
    print("\n" + "=" * 70)
