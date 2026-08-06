"""Alpha Vantage data client for OHLCV, fundamentals, and technical indicators.

Uses the free-tier API key (5 calls/min, 500 calls/day).  Designed as a
fallback/alternative vendor in the vendor_router chain, not a primary source.

Capabilities:
  - Daily OHLCV (full-history or trimmed)
  - Fundamental data: income statement, balance sheet, cash flow, company overview
  - Technical indicators: RSI, MACD, SMA, EMA (pre-computed by AV)
  - Sector performance

Free-tier limits:
  - 5 API calls per minute
  - 500 API calls per day
  - Some endpoints (intraday) are premium-only; daily is free

All functions return dicts or DataFrames.  Functions that can fail due to
rate limits or missing data return clear sentinel values instead of raising.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────

BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_API_KEY = "QYLL5W42OLQBBNSY"
_MIN_CALL_INTERVAL = 12.0  # 5 calls/min → 12s between calls minimum
_MAX_RETRIES = 2

_last_call_time: float = 0.0


def _rate_limit() -> None:
    """Throttle to stay within free-tier limit of 5 calls/min."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


def _get(api_key: str | None = None, **params: Any) -> dict | None:
    """Make a rate-limited GET to Alpha Vantage and return JSON or None."""
    key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY") or DEFAULT_API_KEY
    params["apikey"] = key

    for attempt in range(_MAX_RETRIES):
        _rate_limit()
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("AV request failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            continue
        except ValueError as e:
            logger.warning("AV JSON decode failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            continue

        # Alpha Vantage returns {"Information": "... rate limit ..."} on 5-call
        # overage even with HTTP 200. Detect and retry after a longer pause.
        if isinstance(data, dict):
            info = data.get("Information") or data.get("Note") or ""
            if "rate" in info.lower() or "limit" in info.lower():
                logger.warning("AV rate-limited on attempt %d/%d; waiting 60s", attempt + 1, _MAX_RETRIES)
                time.sleep(60.0)
                continue
            if "Error Message" in data:
                logger.warning("AV error for %s: %s", params.get("function", "?"), data["Error Message"])
                return None

        return data

    logger.error("AV request failed after %d attempts for function=%s", _MAX_RETRIES, params.get("function", "?"))
    return None


# ── OHLCV ──────────────────────────────────────────────────────────────────


def fetch_daily_ohlcv(
    symbol: str,
    outputsize: str = "compact",
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """Fetch daily OHLCV from Alpha Vantage.

    Args:
        symbol: Ticker symbol (canonical Yahoo form).
        outputsize: "compact" (last 100 days) or "full" (20+ years).
        api_key: API key; defaults to env or hardcoded key.

    Returns:
        DataFrame with columns [open, high, low, close, volume], tz-aware UTC index.
        None on any failure.
    """
    data = _get(function="TIME_SERIES_DAILY", symbol=symbol, outputsize=outputsize, api_key=api_key)
    if not data:
        return None

    series = data.get("Time Series (Daily)")
    if not series:
        logger.warning("AV: no daily series for %s", symbol)
        return None

    records = []
    for date_str, values in series.items():
        try:
            records.append({
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": float(values.get("5. volume", 0)),
            })
        except (TypeError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records, index=pd.to_datetime(list(series.keys())))
    df.index.name = "timestamp"
    df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    return df


# ── Fundamentals ───────────────────────────────────────────────────────────


def fetch_company_overview(symbol: str, api_key: str | None = None) -> dict | None:
    """Fetch company overview: sector, industry, market cap, ratios.

    Returns a dict with keys like:
        Sector, Industry, MarketCapitalization, PERatio, PEGRatio,
        BookValue, DividendYield, EPS, RevenueTTM, FreeCashFlowTTM,
        DebtToEquityRatio, ReturnOnEquityTTM, etc.
    """
    data = _get(function="OVERVIEW", symbol=symbol, api_key=api_key)
    if not data or "Symbol" not in data:
        return None
    return data


def fetch_income_statement(symbol: str, api_key: str | None = None) -> list[dict] | None:
    """Fetch annual income statements.  Returns list of dicts, most recent first."""
    data = _get(function="INCOME_STATEMENT", symbol=symbol, api_key=api_key)
    if not data:
        return None
    reports = data.get("annualReports")
    if not reports:
        return None
    return [
        {
            "fiscal_date": r.get("fiscalDateEnding", ""),
            "reported_currency": r.get("reportedCurrency", ""),
            "total_revenue": _safe_float(r.get("totalRevenue")),
            "gross_profit": _safe_float(r.get("grossProfit")),
            "operating_income": _safe_float(r.get("operatingIncome")),
            "net_income": _safe_float(r.get("netIncome")),
            "ebitda": _safe_float(r.get("EBITDA")),
        }
        for r in reports
    ]


def fetch_balance_sheet(symbol: str, api_key: str | None = None) -> list[dict] | None:
    """Fetch annual balance sheets.  Returns list of dicts, most recent first."""
    data = _get(function="BALANCE_SHEET", symbol=symbol, api_key=api_key)
    if not data:
        return None
    reports = data.get("annualReports")
    if not reports:
        return None
    return [
        {
            "fiscal_date": r.get("fiscalDateEnding", ""),
            "total_assets": _safe_float(r.get("totalAssets")),
            "total_liabilities": _safe_float(r.get("totalLiabilities")),
            "total_shareholder_equity": _safe_float(r.get("totalShareholderEquity")),
            "cash_and_equivalents": _safe_float(r.get("cashAndCashEquivalentsAtCarryingValue")),
            "long_term_debt": _safe_float(r.get("longTermDebt")),
        }
        for r in reports
    ]


def fetch_cash_flow(symbol: str, api_key: str | None = None) -> list[dict] | None:
    """Fetch annual cash flow statements.  Returns list of dicts, most recent first."""
    data = _get(function="CASH_FLOW", symbol=symbol, api_key=api_key)
    if not data:
        return None
    reports = data.get("annualReports")
    if not reports:
        return None
    return [
        {
            "fiscal_date": r.get("fiscalDateEnding", ""),
            "operating_cashflow": _safe_float(r.get("operatingCashflow")),
            "capital_expenditures": _safe_float(r.get("capitalExpenditures")),
            "free_cash_flow": (
                _safe_float(r.get("operatingCashflow")) - _safe_float(r.get("capitalExpenditures"))
                if r.get("operatingCashflow") and r.get("capitalExpenditures")
                else None
            ),
        }
        for r in reports
    ]


def fetch_earnings(symbol: str, api_key: str | None = None) -> list[dict] | None:
    """Fetch quarterly earnings.  Returns list of dicts, most recent first."""
    data = _get(function="EARNINGS", symbol=symbol, api_key=api_key)
    if not data:
        return None
    reports = data.get("quarterlyEarnings")
    if not reports:
        return None
    return [
        {
            "fiscal_date": r.get("fiscalDateEnding", ""),
            "reported_eps": _safe_float(r.get("reportedEPS")),
            "estimated_eps": _safe_float(r.get("estimatedEPS")),
            "surprise_pct": _safe_float(r.get("surprisePercentage")),
        }
        for r in reports
    ]


# ── Technical Indicators ───────────────────────────────────────────────────


def fetch_rsi(symbol: str, period: int = 14, series: str = "close", api_key: str | None = None) -> pd.Series | None:
    """Fetch RSI values from Alpha Vantage.  Returns Series indexed by date."""
    data = _get(function="RSI", symbol=symbol, interval="daily", time_period=period, series_type=series, api_key=api_key)
    if not data:
        return None
    series_data = data.get("Technical Analysis: RSI")
    if not series_data:
        return None
    records = {k: float(v["RSI"]) for k, v in sorted(series_data.items())}
    return pd.Series(records, name="rsi").rename_axis("timestamp").tz_localize("UTC")


def fetch_macd(symbol: str, api_key: str | None = None) -> pd.DataFrame | None:
    """Fetch MACD values (MACD, signal, histogram).  Returns DataFrame."""
    data = _get(function="MACD", symbol=symbol, interval="daily", series_type="close", api_key=api_key)
    if not data:
        return None
    series_data = data.get("Technical Analysis: MACD")
    if not series_data:
        return None
    records = {
        k: {
            "macd": float(v.get("MACD", 0)),
            "macd_signal": float(v.get("MACD_Signal", 0)),
            "macd_hist": float(v.get("MACD_Hist", 0)),
        }
        for k, v in sorted(series_data.items())
    }
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df.index.name = "timestamp"
    return df


def fetch_sma(symbol: str, period: int = 50, api_key: str | None = None) -> pd.Series | None:
    """Fetch SMA values.  Returns Series indexed by date."""
    data = _get(function="SMA", symbol=symbol, interval="daily", time_period=period, series_type="close", api_key=api_key)
    if not data:
        return None
    series_data = data.get("Technical Analysis: SMA")
    if not series_data:
        return None
    records = {k: float(v["SMA"]) for k, v in sorted(series_data.items())}
    return pd.Series(records, name=f"sma_{period}").rename_axis("timestamp").tz_localize("UTC")


# ── Sector ─────────────────────────────────────────────────────────────────


def fetch_sector_performance(api_key: str | None = None) -> dict | None:
    """Fetch real-time sector performance percentages.

    Returns dict like: {"Information Technology": 0.42, "Energy": -0.15, ...}
    Rate-limited per sector call.
    """
    data = _get(function="SECTOR", api_key=api_key)
    if not data:
        return None
    perf = data.get("Rank A: Real-Time Performance")
    if not perf:
        return None
    return {k: float(v) for k, v in perf.items()}


# ── helpers ────────────────────────────────────────────────────────────────


def _safe_float(val: Any) -> float | None:
    """Convert AV string-number to float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def remaining_daily_calls(api_key: str | None = None) -> int | None:
    """Estimate remaining daily calls by reading the last response header.

    Alpha Vantage doesn't expose remaining quota in the API response, so this
    is best-effort from a lightweight quote call.
    """
    # AV doesn't provide a usage header; the free tier is 500/day.
    # This is a placeholder for future enhancement.
    return None
