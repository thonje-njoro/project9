"""Symbol normalization: user/broker symbols → canonical Yahoo Finance tickers.

Ported from TradingAgents (TauricResearch) v0.3.0 symbol_utils.py.
Resolves metals, energy, forex, crypto, and index CFD symbols to the format
yfinance actually understands, so data fetches never return empty frames from
a symbol mismatch.

Resolution order (first match wins):
  1. Explicit alias table (metals, energy, index CFDs)
  2. Crypto rule: known base quoted in USD/USDT/USDC → BASE-USD
  3. Forex rule: six letters, two ISO currency codes → PAIR=X
  4. Return as-is (equities, ETFs, native Yahoo symbols like GC=F, ^GSPC)

Usage:
    from data.symbol_map import normalize_symbol
    yf_ticker = normalize_symbol("XAUUSD")  # → "GC=F"
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs.
_FOREX_CURRENCIES = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
    "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
})

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset({
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK",
})

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule.  Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol.
_ALIASES = {
    # Precious metals (spot names → COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs → Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-^=]+$")

# Crypto quote currencies: Yahoo lists only <BASE>-USD, so USDT/USDC quotes
# resolve to -USD (#982).  Longest-first so USDT/USDC match before USD.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` if ``s`` is a known crypto quoted in USD/USDT/USDC.

    Accepts dashed or undashed forms: ``BTCUSD``, ``BTCUSDT``, ``BTC-USDT``,
    ``BTC-USDC`` all resolve to ``BTC-USD``.  Returns None otherwise.
    """
    compact = s.replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            if base in _CRYPTO_BASES:
                return f"{base}-USD"
            break
    return None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Safe to call on every request — no network calls, pure string ops + cache.
    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    s = s.rstrip("+")  # broker CFD/qualifier suffixes
    s = s.replace("/", "")  # XAU/USD → XAUUSD for alias matching

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r → Yahoo %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and bool(_YAHOO_SAFE.fullmatch(symbol))


# ── convenience for the existing main.py data-fetch path ──

def yfinance_symbol(symbol: str) -> str:
    """Map our project instrument name to a yfinance ticker.

    Drop-in replacement for the old ``_yfinance_symbol()`` in main.py.
    """
    return normalize_symbol(symbol)
