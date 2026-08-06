"""Vendor router — multi-source data fetching with typed error handling and fallback.

Ported pattern from TradingAgents (TauricResearch) v0.3.0 interface.py.

Routes data requests through a configurable vendor chain per category:

    core_stock_apis:  Alpaca → yfinance → Alpha Vantage → synthetic
    forex_data:       yfinance → Alpha Vantage
    fundamental_data: Alpha Vantage → yfinance
    crypto_data:      Alpaca → yfinance

Each vendor implements the same interface (fetch_daily_ohlcv, etc.).  The router
tries vendors in configured order.  Typed errors let callers distinguish "no
data for this symbol" from "rate-limited" from "network error".

Usage:
    from data.vendor_router import get_ohlcv
    df = get_ohlcv("GLD", "2022-01-01", "2024-12-31")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ── typed error hierarchy ──────────────────────────────────────────────────


class VendorError(Exception):
    """Base for all vendor-router errors."""


class VendorRateLimitError(VendorError):
    """Vendor returned a rate-limit response.  Router can retry after delay."""


class VendorNotConfiguredError(VendorError):
    """Vendor is not available (no API key, not installed, etc.)."""


class NoMarketDataError(VendorError):
    """Vendor returned a successful response but no usable data for the symbol.

    Attributes:
        symbol: The requested symbol.
        canonical: The resolved Yahoo symbol (may differ from input).
        detail: Human-readable reason.
    """
    def __init__(self, symbol: str, canonical: str = "", detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        super().__init__(f"No market data for '{symbol}'" + (f" ({detail})" if detail else ""))


# ── vendor registry ────────────────────────────────────────────────────────

# Each vendor registers one or more fetch functions.
# Signature: fetch_ohlcv(symbol: str, start: str, end: str) → pd.DataFrame | None
# Return None to signal "not my symbol, try next vendor".

_vendor_registry: dict[str, dict[str, Callable]] = {
    "ohlcv": {},
    "fundamentals": {},
}


def register_vendor(category: str, name: str, func: Callable) -> None:
    """Register a fetch function under a category + vendor name."""
    if category not in _vendor_registry:
        _vendor_registry[category] = {}
    _vendor_registry[category][name] = func


def _get_vendors(category: str, preferred_chain: list[str] | None = None) -> list[tuple[str, Callable]]:
    """Return ordered list of (name, func) for the category."""
    available = _vendor_registry.get(category, {})
    if preferred_chain:
        ordered = [(n, available[n]) for n in preferred_chain if n in available]
        # Append any unlisted vendors at the end
        for name, func in available.items():
            if name not in preferred_chain:
                ordered.append((name, func))
        return ordered
    return list(available.items())


# ── router ─────────────────────────────────────────────────────────────────


def get_ohlcv(
    symbol: str,
    start: str,
    end: str,
    vendor_chain: list[str] | None = None,
) -> pd.DataFrame | None:
    """Fetch OHLCV data, trying vendors in order.

    Args:
        symbol: Ticker symbol.
        start: Start date "YYYY-MM-DD".
        end: End date "YYYY-MM-DD".
        vendor_chain: Ordered vendor names to try.  None = all registered.

    Returns:
        DataFrame or None if all vendors fail.
    """
    vendors = _get_vendors("ohlcv", vendor_chain)
    if not vendors:
        logger.warning("No OHLCV vendors registered")
        return None

    last_error: Exception | None = None
    for name, func in vendors:
        try:
            logger.debug("Trying vendor %r for %s", name, symbol)
            df = func(symbol, start, end)
            if df is not None and not df.empty:
                logger.info("Got OHLCV for %s from %s (%d bars)", symbol, name, len(df))
                return df
            logger.debug("Vendor %r returned no data for %s", name, symbol)
        except NoMarketDataError:
            logger.debug("Vendor %r: no market data for %s", name, symbol)
            continue
        except VendorRateLimitError:
            logger.warning("Vendor %r rate-limited for %s, trying next", name, symbol)
            continue
        except VendorNotConfiguredError:
            logger.debug("Vendor %r not configured for %s", name, symbol)
            continue
        except Exception as e:
            logger.warning("Vendor %r failed for %s: %s", name, symbol, e)
            last_error = e
            continue

    if last_error:
        logger.error("All OHLCV vendors failed for %s; last error: %s", symbol, last_error)
    else:
        logger.warning("No OHLCV vendor returned data for %s", symbol)
    return None


def get_fundamentals(
    symbol: str,
    vendor_chain: list[str] | None = None,
) -> dict | None:
    """Fetch company fundamentals, trying vendors in order.

    Returns:
        Dict with keys like sector, industry, market_cap, pe_ratio, etc.
    """
    # This calls alpha_vantage.fetch_company_overview if registered
    vendors = _get_vendors("fundamentals", vendor_chain)
    if not vendors:
        return None

    for name, func in vendors:
        try:
            result = func(symbol)
            if result:
                return result
        except Exception as e:
            logger.debug("Vendor %r fundamentals failed: %s", name, e)
            continue
    return None


# ── built-in vendor wrappers ───────────────────────────────────────────────


def _yfinance_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch OHLCV from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise VendorNotConfiguredError("yfinance not installed")

    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    except Exception as e:
        raise VendorRateLimitError(str(e)) from e

    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        price_types = {"open", "high", "low", "close", "adj close", "volume"}
        col_names = []
        for col in df.columns:
            name = None
            for level in col:
                if isinstance(level, str) and level.lower() in price_types:
                    name = level.lower()
                    break
            col_names.append(name or str(col[-1]).lower())
        df.columns = col_names
    else:
        df.columns = [c.lower() for c in df.columns]

    if "adj close" in df.columns and "close" in df.columns:
        df = df.drop(columns=["adj close"])
    elif "adj close" in df.columns:
        df = df.rename(columns={"adj close": "close"})

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return None

    df = df[["open", "high", "low", "close", "volume"]]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df.sort_index()


def _alpha_vantage_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch OHLCV from Alpha Vantage (daily only)."""
    try:
        from data.alpha_vantage import fetch_daily_ohlcv
    except ImportError:
        raise VendorNotConfiguredError("alpha_vantage module not available")

    # AV only has daily data on free tier; outputsize="full" for history
    df = fetch_daily_ohlcv(symbol, outputsize="full")
    if df is None or df.empty:
        return None

    # Trim to requested range
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    if df.empty:
        return None

    # Standardize columns
    if "adjusted_close" in df.columns:
        # Use adjusted close as close for continuity
        df = df.rename(columns={"adjusted_close": "close"})
    df = df[["open", "high", "low", "close", "volume"]]
    return df


# ── Alpaca wrapper (delegates to existing DataFetcher) ─────────────────────


def _alpaca_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch OHLCV from Alpaca via the existing DataFetcher."""
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or api_key == "your_free_tier_key" or not secret_key:
        raise VendorNotConfiguredError("no valid Alpaca API keys")

    try:
        from data.fetcher import DataFetcher
        fetcher = DataFetcher(api_key, secret_key)
        # Alpaca uses stock/crypto distinction
        info = {"asset_class": "stock"}  # assume stock; caller overrides
        data = fetcher.fetch_all({symbol: info}, {"start_date": start, "end_date": end})
        return data.get(symbol)
    except Exception as e:
        raise VendorRateLimitError(str(e)) from e


# ── register built-in vendors ──────────────────────────────────────────────

register_vendor("ohlcv", "yfinance", _yfinance_ohlcv)
register_vendor("ohlcv", "alpha_vantage", _alpha_vantage_ohlcv)
register_vendor("ohlcv", "alpaca", _alpaca_ohlcv)


def _alpha_vantage_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamentals from Alpha Vantage."""
    try:
        from data.alpha_vantage import fetch_company_overview
    except ImportError:
        return None
    overview = fetch_company_overview(symbol)
    if not overview:
        return None
    return {
        "sector": overview.get("Sector", ""),
        "industry": overview.get("Industry", ""),
        "market_cap": overview.get("MarketCapitalization"),
        "pe_ratio": overview.get("PERatio"),
        "dividend_yield": overview.get("DividendYield"),
        "eps": overview.get("EPS"),
        "book_value": overview.get("BookValue"),
        "debt_to_equity": overview.get("DebtToEquityTTM"),
        "free_cash_flow": overview.get("FreeCashFlowTTM"),
        "revenue_ttm": overview.get("RevenueTTM"),
        "profit_margin": overview.get("ProfitMargin"),
        "return_on_equity": overview.get("ReturnOnEquityTTM"),
        "beta": overview.get("Beta"),
    }


register_vendor("fundamentals", "alpha_vantage", _alpha_vantage_fundamentals)


# ── convenience: default vendor chains ─────────────────────────────────────

# Matches the config.py DATA_VENDORS structure.  These are the recommended
# chains — override per call if needed.

DEFAULT_VENDOR_CHAINS = {
    "core_stock_apis": ["alpaca", "yfinance", "alpha_vantage"],
    "forex_data": ["yfinance", "alpha_vantage"],
    "fundamental_data": ["alpha_vantage"],
    "crypto_data": ["alpaca", "yfinance"],
}
