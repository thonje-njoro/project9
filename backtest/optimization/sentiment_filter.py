"""Sentiment-based entry filter using multi-source data.

Blocks entries when sentiment is too negative or fundamentals are weak.
Uses cached daily data (not per-bar) for backtest compatibility.
"""

import numpy as np
import pandas as pd
from typing import Optional


def sentiment_filter(
    df: pd.DataFrame,
    ticker: str = "SPY",
    min_score: float = -0.5,
) -> pd.Series:
    """Block entries when sentiment is below threshold.

    Args:
        df: OHLCV DataFrame.
        ticker: Ticker symbol for data fetch.
        min_score: Minimum sentiment score (-1 to 1) to allow entries.

    Returns:
        Boolean Series (True = allow entry).
    """
    try:
        from data.llm_data_fetcher import fetch_sentiment
        sentiment = fetch_sentiment(ticker)
        score = sentiment.get("score", 0.0)
    except Exception:
        return pd.Series(True, index=df.index)

    allow = score >= min_score
    return pd.Series(allow, index=df.index).shift(1).fillna(True)


def fundamentals_filter(
    df: pd.DataFrame,
    ticker: str = "SPY",
    min_fcf_yield: float = 0.0,
    max_debt_ratio: float = 1.0,
) -> pd.Series:
    """Block entries when fundamentals are weak.

    Args:
        df: OHLCV DataFrame.
        ticker: Ticker symbol for data fetch.
        min_fcf_yield: Minimum free cash flow yield.
        max_debt_ratio: Maximum debt-to-asset ratio.

    Returns:
        Boolean Series (True = allow entry).
    """
    try:
        from data.llm_data_fetcher import fetch_fundamentals
        fundamentals = fetch_fundamentals(ticker)
        fcf_yield = fundamentals.get("fcf_yield")
        debt_ratio = fundamentals.get("debt_ratio")
    except Exception:
        return pd.Series(True, index=df.index)

    allow = True
    if fcf_yield is not None and fcf_yield < min_fcf_yield:
        allow = False
    if debt_ratio is not None and debt_ratio > max_debt_ratio:
        allow = False

    return pd.Series(allow, index=df.index).shift(1).fillna(True)


def insider_filter(
    df: pd.DataFrame,
    ticker: str = "SPY",
    min_net_sentiment: float = -0.5,
) -> pd.Series:
    """Block entries when insider selling is heavy.

    Args:
        df: OHLCV DataFrame.
        ticker: Ticker symbol for data fetch.
        min_net_sentiment: Minimum insider net sentiment (-1 to 1).

    Returns:
        Boolean Series (True = allow entry).
    """
    try:
        from data.llm_data_fetcher import fetch_insider_activity
        insiders = fetch_insider_activity(ticker)
        net = insiders.get("net_sentiment", 0.0)
    except Exception:
        return pd.Series(True, index=df.index)

    allow = net >= min_net_sentiment
    return pd.Series(allow, index=df.index).shift(1).fillna(True)
