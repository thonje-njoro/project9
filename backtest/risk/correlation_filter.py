"""
risk/correlation_filter.py — Block new trades when portfolio correlation > 0.7.
"""

import logging

import numpy as np
import pandas as pd

from backtest.config import RISK_CONFIG

logger = logging.getLogger(__name__)


def compute_correlation_matrix(returns_dict: dict[str, pd.Series],
                               lookback: int = 20) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix from return series.

    Args:
        returns_dict: {symbol: pd.Series of returns}.
        lookback: Rolling window for correlation (days).

    Returns:
        Correlation DataFrame (symbols × symbols).
    """
    if not returns_dict:
        return pd.DataFrame()

    df = pd.DataFrame(returns_dict)
    # Use the last `lookback` periods
    if len(df) > lookback:
        df = df.iloc[-lookback:]
    return df.corr()


def would_exceed_correlation(returns_dict: dict[str, pd.Series],
                             new_symbol: str,
                             new_returns: pd.Series,
                             max_corr: float | None = None) -> bool:
    """
    Check if adding a new position would exceed the correlation threshold
    with any existing open position.

    Args:
        returns_dict: {symbol: returns} for currently open positions.
        new_symbol: Symbol of the proposed new trade.
        new_returns: Return series for the proposed new trade.
        max_corr: Maximum allowed correlation. Default from config.

    Returns:
        True if the new trade should be BLOCKED (correlation too high).
    """
    if max_corr is None:
        max_corr = RISK_CONFIG["max_correlation"]

    if not returns_dict:
        return False

    for existing_symbol, existing_returns in returns_dict.items():
        # Align series
        common = new_returns.index.intersection(existing_returns.index)
        if len(common) < 10:
            continue  # Not enough data to compute correlation

        corr = new_returns.loc[common].corr(existing_returns.loc[common])

        if abs(corr) > max_corr:
            logger.info(
                f"Correlation block: {new_symbol} vs {existing_symbol} "
                f"corr={corr:.3f} > {max_corr}"
            )
            return True

    return False


def get_open_position_returns(open_positions: dict,
                              all_returns: dict[str, pd.Series],
                              lookback: int = 20) -> dict[str, pd.Series]:
    """
    Extract return series for currently open positions.

    Args:
        open_positions: Dict of {symbol: position_info} from state.
        all_returns: Dict of {symbol: full return series}.
        lookback: Number of recent periods to use.

    Returns:
        Dict of {symbol: recent return series} for open positions.
    """
    result = {}
    for symbol in open_positions:
        if symbol in all_returns:
            series = all_returns[symbol]
            result[symbol] = series.iloc[-lookback:] if len(series) > lookback else series
    return result
