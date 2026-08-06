"""Correlation-based entry filter to prevent risk-on pile-up."""

import pandas as pd


def apply_correlation_filter(
    spy_long_entries: pd.Series,
    qqq_long_entries: pd.Series,
    btc_long_entries: pd.Series,
) -> pd.Series:
    """
    Block BTC long entries on any bar where both SPY and QQQ
    already have active long entries (risk-on pile-up).
    """
    both_long = spy_long_entries & qqq_long_entries
    filtered_btc = btc_long_entries & ~both_long
    return filtered_btc
