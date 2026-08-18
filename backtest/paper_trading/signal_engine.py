"""
paper_trading/signal_engine.py — Runs live strategies and outputs signals.
"""

import importlib
import inspect
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.config import INSTRUMENTS, STRATEGY_PARAMS
from backtest.data.fetcher import fetch

logger = logging.getLogger(__name__)


class LiveSignalEngine:
    """
    Generates live trading signals for all configured instruments.
    Runs each strategy on the latest available data.
    """

    def __init__(self):
        self.strategy_cache = {}

    def _import_strategy(self, strategy_name: str):
        """Import and cache strategy module."""
        if strategy_name not in self.strategy_cache:
            module_map = {
                "kalman_trend": "backtest.strategies.kalman_trend",
                "orb_strategy": "backtest.strategies.orb_strategy",
                "momentum_orb": "backtest.strategies.momentum_orb",
                "vwap_mean_reversion": "backtest.strategies.vwap_mean_reversion",
                "xauusd_session_mr": "backtest.strategies.xauusd_session_mr",
                "cper_gld_ratio": "backtest.strategies.cper_gld_ratio",
            }
            module_path = module_map.get(strategy_name)
            if module_path:
                self.strategy_cache[strategy_name] = importlib.import_module(module_path)
        return self.strategy_cache.get(strategy_name)

    def generate(self, lookback_bars: int = 200) -> dict:
        """
        Generate signals for all instruments.

        Args:
            lookback_bars: Number of recent bars to fetch for signal generation.

        Returns:
            dict: {symbol: signal_dict} where signal_dict has:
                'side': 'long' | 'short' | None
                'strength': float (0-1)
                'timestamp': datetime
                'strategy': str
        """
        signals = {}

        for symbol, instrument in INSTRUMENTS.items():
            strategy_name = instrument["strategy"]
            timeframe = instrument["timeframe"]

            try:
                signal = self._generate_single(
                    symbol, strategy_name, timeframe, lookback_bars
                )
                if signal is not None:
                    signals[symbol] = signal
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")

        return signals

    def _generate_single(self, symbol: str, strategy_name: str,
                         timeframe: str, lookback_bars: int) -> dict | None:
        """Generate signal for a single instrument."""
        # Fetch recent data
        df = fetch(symbol, timeframe)
        if df.empty:
            return None

        # Use last N bars
        if len(df) > lookback_bars:
            df = df.iloc[-lookback_bars:]

        # Import strategy
        strategy_mod = self._import_strategy(strategy_name)
        if strategy_mod is None:
            return None

        # Get params
        params = STRATEGY_PARAMS.get(strategy_name, {}).get(symbol, {})

        # Filter params
        fn = strategy_mod.generate_signals
        sig = inspect.signature(fn)
        valid_keys = set(sig.parameters.keys()) - {"df"}
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}

        # Generate signals
        result = fn(df=df, **filtered_params)

        if len(result) == 5:
            long_entries, long_exits, short_entries, short_exits, _ = result
        else:
            long_entries, long_exits, short_entries, short_exits = result

        # Check last bar for active signal
        if len(long_entries) == 0:
            return None

        last_long_entry = bool(long_entries.iloc[-1]) if len(long_entries) > 0 else False
        last_short_entry = bool(short_entries.iloc[-1]) if len(short_entries) > 0 else False
        last_long_exit = bool(long_exits.iloc[-1]) if len(long_exits) > 0 else False
        last_short_exit = bool(short_exits.iloc[-1]) if len(short_exits) > 0 else False

        side = None
        if last_long_entry and not last_long_exit:
            side = "long"
        elif last_short_entry and not last_short_exit:
            side = "short"

        if side is None:
            return None

        # Compute ATR from recent bars
        from backtest.data.resampler import compute_atr
        atr_series = compute_atr(df, period=14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else None

        return {
            "symbol": symbol,
            "side": side,
            "price": float(df["close"].iloc[-1]),
            "atr": current_atr,
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strategy_name,
            "timeframe": timeframe,
        }
