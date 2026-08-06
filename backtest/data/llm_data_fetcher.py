"""Multi-source data enrichment using TradingAgents' dataflows.

Fetches sentiment, fundamentals, macro data, and insider activity.
All results are cached daily to avoid repeated API calls.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np


CACHE_DIR = Path(__file__).parent.parent / "data" / "llm_cache"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    _ensure_cache_dir()
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, max_age_hours: int = 24) -> Optional[dict]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(hours=max_age_hours):
            return None
        return data.get("value")
    except Exception:
        return None


def _save_cache(key: str, value: dict):
    path = _cache_path(key)
    path.write_text(json.dumps({
        "cached_at": datetime.now().isoformat(),
        "value": value,
    }, default=str))


def fetch_sentiment(ticker: str) -> dict:
    """Fetch aggregated sentiment score from Reddit + StockTwits.

    Returns:
        dict with keys: score (-1 to 1), volume, sources
    """
    cache_key = f"sentiment_{ticker}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {"score": 0.0, "volume": 0, "sources": []}

    try:
        from tradingagents.dataflows.reddit import get_reddit_posts
        posts = get_reddit_posts(ticker)
        if posts and len(posts) > 0:
            bullish = sum(1 for p in posts if "bull" in str(p).lower() or "buy" in str(p).lower())
            bearish = sum(1 for p in posts if "bear" in str(p).lower() or "sell" in str(p).lower())
            total = max(len(posts), 1)
            reddit_score = (bullish - bearish) / total
            result["score"] += reddit_score * 0.5
            result["volume"] += len(posts)
            result["sources"].append("reddit")
    except Exception:
        pass

    try:
        from tradingagents.dataflows.stocktwits import get_stocktwits_posts
        posts = get_stocktwits_posts(ticker)
        if posts and len(posts) > 0:
            bullish = sum(1 for p in posts if str(p).get("sentiment") == "bullish")
            bearish = sum(1 for p in posts if str(p).get("sentiment") == "bearish")
            total = max(len(posts), 1)
            st_score = (bullish - bearish) / total
            result["score"] += st_score * 0.5
            result["volume"] += len(posts)
            result["sources"].append("stocktwits")
    except Exception:
        pass

    result["score"] = max(-1.0, min(1.0, result["score"]))
    _save_cache(cache_key, result)
    return result


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental valuation metrics.

    Returns:
        dict with keys: pe_ratio, pb_ratio, fcf_yield, debt_ratio, revenue_growth
    """
    cache_key = f"fundamentals_{ticker}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {
        "pe_ratio": None, "pb_ratio": None, "fcf_yield": None,
        "debt_ratio": None, "revenue_growth": None,
    }

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        result["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        result["pb_ratio"] = info.get("priceToBook")
        market_cap = info.get("marketCap", 0)
        fcf = info.get("freeCashflow", 0)
        if market_cap and fcf and market_cap > 0:
            result["fcf_yield"] = fcf / market_cap
        total_debt = info.get("totalDebt", 0)
        total_assets = info.get("totalAssets", 1)
        if total_assets and total_assets > 0:
            result["debt_ratio"] = total_debt / total_assets
        result["revenue_growth"] = info.get("revenueGrowth")
    except Exception:
        pass

    _save_cache(cache_key, result)
    return result


def fetch_macro_context() -> dict:
    """Fetch current macro regime indicators.

    Returns:
        dict with keys: vix, fed_rate, yield_curve, dxy_change
    """
    cache_key = "macro_context"
    cached = _load_cache(cache_key, max_age_hours=6)
    if cached:
        return cached

    result = {"vix": None, "fed_rate": None, "yield_curve": None, "dxy_change": None}

    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if len(hist) > 0:
            result["vix"] = float(hist["Close"].iloc[-1])
    except Exception:
        pass

    try:
        from tradingagents.dataflows.fred import get_macro_indicators
        macro = get_macro_indicators()
        if macro:
            result["fed_rate"] = macro.get("fed_rate")
            result["yield_curve"] = macro.get("yield_curve")
    except Exception:
        pass

    try:
        import yfinance as yf
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        if len(hist) >= 2:
            result["dxy_change"] = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
    except Exception:
        pass

    _save_cache(cache_key, result)
    return result


def fetch_insider_activity(ticker: str) -> dict:
    """Fetch recent insider buy/sell activity.

    Returns:
        dict with keys: buy_count, sell_count, net_sentiment
    """
    cache_key = f"insider_{ticker}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {"buy_count": 0, "sell_count": 0, "net_sentiment": 0.0}

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        insider = stock.insider_txns
        if insider is not None and len(insider) > 0:
            recent = insider.head(20)
            buys = len(recent[recent["Text"].str.contains("Buy", case=False, na=False)])
            sells = len(recent[recent["Text"].str.contains("Sale", case=False, na=False)])
            result["buy_count"] = buys
            result["sell_count"] = sells
            total = max(buys + sells, 1)
            result["net_sentiment"] = (buys - sells) / total
    except Exception:
        pass

    _save_cache(cache_key, result)
    return result
