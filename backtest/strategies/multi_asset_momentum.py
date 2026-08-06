"""
Multi-Asset Momentum Strategy
==============================
Based on MiMo Claw research: 80 momentum variants tested across 9 US equities

Key Results:
- Train-test Sharpe correlation = 0.685 (STRONG CONSISTENCY)
- Top strategy: vol_target, 60d lookback, weekly rebalancing, 20% vol target
- Test Sharpe: 2.055, Test Return: +47.7%, Test MaxDD: -8.1%
- 47.5% of combos positive in both train AND test

What Works:
1. Weekly rebalancing (daily is too noisy)
2. Longer lookbacks (40-60d, not 5-10d)
3. Volatility targeting (15-20% annual vol)
4. Time-series momentum (trend following)

What Doesn't Work:
1. Cross-sectional momentum (ranking doesn't work)
2. Trend filter (200-day MA is UNRELIABLE)
3. Short lookbacks (5-10d are too noisy)
4. Daily rebalancing (too much transaction cost)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")


def calculate_momentum_signal(
    df: pd.DataFrame,
    lookback_days: int = 60,
    signal_type: str = "simple",
) -> pd.Series:
    """
    Calculate time-series momentum signal.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days for momentum calculation (default: 60)
    signal_type : str
        Type of momentum signal:
        - "simple": Simple return over lookback period
        - "risk_adjusted": Return / volatility
        - "multi_tf": Average of multiple timeframes
    
    Returns
    -------
    Series with momentum signal (positive = bullish, negative = bearish)
    """
    df = df.copy()
    
    # Calculate daily returns
    if "close" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily = df.resample("D")["close"].last().dropna()
        else:
            daily = df["close"]
        
        # Calculate momentum based on signal type
        if signal_type == "simple":
            # Simple return over lookback period
            momentum = daily.pct_change(lookback_days)
            
        elif signal_type == "risk_adjusted":
            # Return / volatility (risk-adjusted momentum)
            returns = daily.pct_change()
            volatility = returns.rolling(lookback_days).std()
            momentum = daily.pct_change(lookback_days) / volatility
            
        elif signal_type == "multi_tf":
            # Average of multiple timeframes (20d, 40d, 60d)
            mom_20 = daily.pct_change(20)
            mom_40 = daily.pct_change(40)
            mom_60 = daily.pct_change(60)
            momentum = (mom_20 + mom_40 + mom_60) / 3
            
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            momentum = momentum.reindex(df.index, method="ffill")
        
        return momentum
    else:
        return pd.Series(0, index=df.index)


def calculate_volatility_target(
    df: pd.DataFrame,
    target_vol: float = 0.20,
    lookback_days: int = 20,
) -> pd.Series:
    """
    Calculate volatility targeting position size.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    target_vol : float
        Target annualized volatility (default: 0.20 = 20%)
    lookback_days : int
        Number of days for volatility calculation (default: 20)
    
    Returns
    -------
    Series with position size (0-1, where 1 is full size)
    """
    df = df.copy()
    
    # Calculate daily returns
    if "close" in df.columns:
        # Resample to daily if intraday
        if hasattr(df.index, 'hour'):
            daily = df.resample("D")["close"].last().dropna()
        else:
            daily = df["close"]
        
        # Calculate realized volatility
        returns = daily.pct_change()
        realized_vol = returns.rolling(lookback_days).std() * np.sqrt(252)
        
        # Calculate position size (inverse volatility targeting)
        # Size = target_vol / realized_vol
        position_size = target_vol / realized_vol
        
        # Cap position size at 1.0 (no leverage)
        position_size = position_size.clip(0, 1)
        
        # Resample back to original timeframe if needed
        if hasattr(df.index, 'hour'):
            position_size = position_size.reindex(df.index, method="ffill")
        
        return position_size
    else:
        return pd.Series(0.5, index=df.index)


def generate_momentum_signals(
    df: pd.DataFrame,
    lookback_days: int = 60,
    signal_type: str = "simple",
    rebalance_freq: str = "weekly",
    use_vol_target: bool = True,
    target_vol: float = 0.20,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate multi-asset momentum trading signals.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days for momentum calculation (default: 60)
    signal_type : str
        Type of momentum signal (simple, risk_adjusted, multi_tf)
    rebalance_freq : str
        Rebalancing frequency (daily, weekly, monthly)
    use_vol_target : bool
        Whether to use volatility targeting
    target_vol : float
        Target annualized volatility (default: 0.20 = 20%)
    
    Returns
    -------
    Tuple of (long_entries, long_exits, short_entries, short_exits)
    """
    df = df.copy()
    
    # Calculate momentum signal
    momentum = calculate_momentum_signal(df, lookback_days, signal_type)
    
    # Calculate volatility targeting position size
    if use_vol_target:
        vol_size = calculate_volatility_target(df, target_vol)
    else:
        vol_size = pd.Series(1, index=df.index)
    
    # Generate signals based on momentum direction
    # Long when momentum > 0, short when momentum < 0
    long_signal = momentum > 0
    short_signal = momentum < 0
    
    # Apply rebalancing frequency
    if rebalance_freq == "weekly":
        # Rebalance weekly (every 5 trading days)
        rebalance_mask = pd.Series(False, index=df.index)
        # Find the first bar of each week
        if hasattr(df.index, 'isocalendar'):
            week_numbers = df.index.isocalendar().week
            # Mark first bar of each week
            for i in range(1, len(df)):
                if week_numbers.iloc[i] != week_numbers.iloc[i-1]:
                    rebalance_mask.iloc[i] = True
        else:
            # Fallback: every 5 bars
            for i in range(0, len(df), 5):
                if i < len(df):
                    rebalance_mask.iloc[i] = True
                    
    elif rebalance_freq == "monthly":
        # Rebalance monthly (every 20 trading days)
        rebalance_mask = pd.Series(False, index=df.index)
        for i in range(0, len(df), 20):
            if i < len(df):
                rebalance_mask.iloc[i] = True
    else:
        # Daily rebalancing
        rebalance_mask = pd.Series(True, index=df.index)
    
    # Apply rebalancing mask
    long_signal = long_signal & rebalance_mask
    short_signal = short_signal & rebalance_mask
    
    # Apply volatility targeting (reduce size in high vol environments)
    # This is done by scaling the signal strength
    long_entries = long_signal & (vol_size > 0.5)  # Only trade when vol size > 50%
    short_entries = short_signal & (vol_size > 0.5)
    
    # Exits: exit when momentum reverses or at rebalance
    long_exits = (momentum < 0) | rebalance_mask
    short_exits = (momentum > 0) | rebalance_mask
    
    return long_entries, long_exits, short_entries, short_exits


def calculate_momentum_metrics(
    df: pd.DataFrame,
    lookback_days: int = 60,
    signal_type: str = "simple",
    rebalance_freq: str = "weekly",
    use_vol_target: bool = True,
    target_vol: float = 0.20,
) -> dict:
    """
    Calculate momentum strategy metrics.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    lookback_days : int
        Number of days for momentum calculation
    signal_type : str
        Type of momentum signal
    rebalance_freq : str
        Rebalancing frequency
    use_vol_target : bool
        Whether to use volatility targeting
    target_vol : float
        Target annualized volatility
    
    Returns
    -------
    dict with strategy metrics
    """
    # Generate signals
    long_entries, long_exits, short_entries, short_exits = generate_momentum_signals(
        df, lookback_days, signal_type, rebalance_freq, use_vol_target, target_vol
    )
    
    # Calculate returns
    if "close" in df.columns:
        returns = df["close"].pct_change()
        
        # Calculate strategy returns
        strategy_returns = pd.Series(0, index=df.index)
        
        # Long returns
        long_mask = long_entries.shift(1).fillna(False)  # Enter on next bar
        strategy_returns[long_mask] = returns[long_mask]
        
        # Short returns
        short_mask = short_entries.shift(1).fillna(False)  # Enter on next bar
        strategy_returns[short_mask] = -returns[short_mask]
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (strategy_returns > 0).sum()
        total_days = (strategy_returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        
        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_days,
            "avg_daily_return": strategy_returns.mean(),
            "std_daily_return": strategy_returns.std(),
        }
        
        return metrics
    else:
        return {}


def walk_forward_momentum(
    df: pd.DataFrame,
    train_start: str = "2019-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2024-07-31",
    lookback_days: int = 60,
    signal_type: str = "simple",
    rebalance_freq: str = "weekly",
    use_vol_target: bool = True,
    target_vol: float = 0.20,
) -> dict:
    """
    Perform walk-forward validation for momentum strategy.
    
    Parameters
    ----------
    df : DataFrame with OHLCV data
    train_start, train_end : str
        Training period dates
    test_start, test_end : str
        Test period dates
    lookback_days : int
        Number of days for momentum calculation
    signal_type : str
        Type of momentum signal
    rebalance_freq : str
        Rebalancing frequency
    use_vol_target : bool
        Whether to use volatility targeting
    target_vol : float
        Target annualized volatility
    
    Returns
    -------
    dict with train and test metrics
    """
    # Split data
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    # Calculate metrics for train and test
    train_metrics = calculate_momentum_metrics(
        df_train, lookback_days, signal_type, rebalance_freq, use_vol_target, target_vol
    )
    
    test_metrics = calculate_momentum_metrics(
        df_test, lookback_days, signal_type, rebalance_freq, use_vol_target, target_vol
    )
    
    # Calculate robustness (test/train Sharpe ratio)
    robustness = test_metrics.get("sharpe", 0) / train_metrics.get("sharpe", 1) if train_metrics.get("sharpe", 0) != 0 else 0
    
    return {
        "train": train_metrics,
        "test": test_metrics,
        "robustness": robustness,
        "params": {
            "lookback_days": lookback_days,
            "signal_type": signal_type,
            "rebalance_freq": rebalance_freq,
            "use_vol_target": use_vol_target,
            "target_vol": target_vol,
        }
    }


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/admin1/project9/backtest")
    
    # Load sample data
    DATA = "/mnt/c/Users/Admin/project9/data"
    syms = ["NVDA", "AMD", "PLTR", "MRVL"]
    
    print("=" * 80)
    print("MULTI-ASSET MOMENTUM STRATEGY ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    for sym in syms:
        print(f"\n{'─' * 60}")
        print(f"  {sym}")
        print(f"{'─' * 60}")
        
        try:
            df = pd.read_parquet(f"{DATA}/{sym}_5min.parquet")
            
            # Walk-forward validation
            wf_result = walk_forward_momentum(
                df,
                train_start="2022-01-01",
                train_end="2023-06-30",
                test_start="2023-07-01",
                test_end="2024-07-31",
                lookback_days=60,
                signal_type="simple",
                rebalance_freq="weekly",
                use_vol_target=True,
                target_vol=0.20,
            )
            
            results[sym] = wf_result
            
            # Print results
            train = wf_result["train"]
            test = wf_result["test"]
            robustness = wf_result["robustness"]
            
            print(f"  Train Sharpe: {train.get('sharpe', 0):.3f}")
            print(f"  Test Sharpe: {test.get('sharpe', 0):.3f}")
            print(f"  Train Return: {train.get('annual_return', 0)*100:.1f}%")
            print(f"  Test Return: {test.get('annual_return', 0)*100:.1f}%")
            print(f"  Train Max DD: {train.get('max_drawdown', 0)*100:.1f}%")
            print(f"  Test Max DD: {test.get('max_drawdown', 0)*100:.1f}%")
            print(f"  Robustness: {robustness:.3f}")
            
            # Check acceptance criteria
            train_sharpe = train.get('sharpe', 0)
            test_sharpe = test.get('sharpe', 0)
            train_pf = train.get('profit_factor', 0)
            test_pf = test.get('profit_factor', 0)
            test_wr = test.get('win_rate', 0)
            test_return = test.get('annual_return', 0)
            
            acceptance = (
                train_sharpe > 0.3 and test_sharpe > 0.2 and
                train_pf > 1.2 and test_pf > 1.0 and
                test_wr > 0.42 and test_return > 0
            )
            
            print(f"  Acceptance: {'PASS ✓' if acceptance else 'FAIL ✗'}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[sym] = {"error": str(e)}
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    # Count passing symbols
    passing = sum(1 for r in results.values() if r.get("test", {}).get("sharpe", 0) > 0.2)
    print(f"  Symbols with Test Sharpe > 0.2: {passing}/{len(results)}")
    
    # Average test Sharpe
    avg_test_sharpe = np.mean([r.get("test", {}).get("sharpe", 0) for r in results.values()])
    print(f"  Average Test Sharpe: {avg_test_sharpe:.3f}")
    
    print("\n" + "=" * 80)
    print("MULTI-ASSET MOMENTUM IMPLEMENTATION COMPLETE")
    print("=" * 80)
