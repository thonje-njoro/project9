"""Live signal generation engine."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INSTRUMENTS, STRATEGY_PARAMS, RISK_CONFIG, AGENT_VALIDATOR_CONFIG
from risk.position_sizer import compute_atr, atr_position_sizes
from risk.regime_filter import RegimeFilter, create_regime_mask
from risk.vol_regime_stops import compute_vol_calibrated_stop


class LiveSignalEngine:
    """Generates trading signals from live bar data."""

    def __init__(self) -> None:
        self.regime_filters: dict[str, RegimeFilter] = {}
        self._current_df: dict[str, pd.DataFrame] = {}
        self._validator = None
        if AGENT_VALIDATOR_CONFIG.get("enabled", False):
            try:
                from paper_trading.agent_validator import SignalValidator
                self._validator = SignalValidator(
                    provider=AGENT_VALIDATOR_CONFIG.get("provider", "mimo"),
                    model=AGENT_VALIDATOR_CONFIG.get("model", "mimo-v2.5"),
                    max_tokens=AGENT_VALIDATOR_CONFIG.get("max_tokens", 200),
                    temperature=AGENT_VALIDATOR_CONFIG.get("temperature", 0.3),
                    min_confidence=AGENT_VALIDATOR_CONFIG.get("min_confidence", 0.6),
                )
            except Exception:
                self._validator = None

    def initialize_regime_filters(self, data: dict[str, pd.DataFrame]) -> None:
        for symbol, df in data.items():
            rf = RegimeFilter(n_states=2, lookback=60)
            try:
                rf.fit(df)
            except Exception:
                pass
            self.regime_filters[symbol] = rf
            self._current_df[symbol] = df

    def _import_signals(self, strategy: str):
        if strategy == "mean_reversion":
            from strategies.mean_reversion import generate_signals
        elif strategy == "momentum_breakout":
            from strategies.momentum_breakout import generate_signals
        elif strategy == "trend_following":
            from strategies.trend_following import generate_signals
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return generate_signals

    def evaluate(self, symbol: str, df: pd.DataFrame) -> dict:
        info = INSTRUMENTS[symbol]
        strategy = info["strategy"]
        params = STRATEGY_PARAMS[strategy][symbol]

        signal_params = {
            k: v for k, v in params.items()
            if k not in ("trail_atr_mult", "use_vol_calibrated_stop",
                         "vol_stop_base_mult", "vol_stop_min_mult", "vol_stop_max_mult")
        }

        generate_signals = self._import_signals(strategy)
        result = generate_signals(df, **signal_params)

        if len(result) == 5:
            long_entries, long_exits, short_entries, short_exits, strategy_trailing = result
        else:
            long_entries, long_exits, short_entries, short_exits = result
            strategy_trailing = None

        long_entries, long_exits, short_entries, short_exits = self._apply_regime_filter(
            symbol, df, long_entries, long_exits, short_entries, short_exits
        )

        last_idx = len(df) - 1
        action = "hold"
        if long_entries.iloc[last_idx]:
            action = "long_entry"
        elif long_exits.iloc[last_idx]:
            action = "long_exit"
        elif short_entries.iloc[last_idx]:
            action = "short_entry"
        elif short_exits.iloc[last_idx]:
            action = "short_exit"

        price = float(df["close"].iloc[last_idx])

        atr = compute_atr(df, RISK_CONFIG["atr_period"])
        position_sizes = atr_position_sizes(
            10000, atr, df["close"],
            RISK_CONFIG["max_risk_per_trade_pct"],
            RISK_CONFIG.get("max_exposure_pct", 0.25),
        )
        size_notional = float(position_sizes.iloc[last_idx] * price)

        trailing_stop_pct = 0.0
        if params.get("use_vol_calibrated_stop", False) and symbol in self.regime_filters:
            regime_probs = self.regime_filters[symbol].get_regime_probabilities(df)
            trend_probs = regime_probs.get("trending", pd.Series(0.5, index=df.index))
            trailing_stop = compute_vol_calibrated_stop(
                df, atr, trend_probs,
                base_mult=params.get("vol_stop_base_mult", 3.0),
                vol_scale_range=(
                    params.get("vol_stop_min_mult", 1.8) / params.get("vol_stop_base_mult", 3.0),
                    params.get("vol_stop_max_mult", 5.4) / params.get("vol_stop_base_mult", 3.0),
                ),
            )
            trailing_stop_pct = float(trailing_stop.iloc[last_idx] / price)
        elif "trail_atr_mult" in params:
            trailing_stop_pct = float(atr.iloc[last_idx] * params["trail_atr_mult"] / price)

        regime = "unknown"
        if symbol in self.regime_filters:
            try:
                regimes = self.regime_filters[symbol].predict(df)
                regime = str(regimes.iloc[last_idx])
            except Exception:
                pass

        atr_val = float(atr.iloc[last_idx]) if len(atr) > 0 else 0.0
        validation = self._validate_with_agent(symbol, action, price, regime, df, atr_val)

        if not validation.get("approved", True):
            action = "hold"

        return {
            "action": action,
            "price": price,
            "regime": regime,
            "size_notional": size_notional,
            "trailing_stop_pct": trailing_stop_pct,
            "agent_validation": validation,
        }

    def _validate_with_agent(
        self, symbol: str, action: str, price: float, regime: str, df: pd.DataFrame, atr_val: float
    ) -> dict:
        """Validate signal using LLM agent if enabled."""
        if self._validator is None or action == "hold":
            return {"approved": True, "confidence": 1.0, "reasoning": "No validator or hold action"}

        try:
            from paper_trading.agent_context import build_context
            context = build_context(symbol, df, regime, action, price, atr_val)
            result = self._validator.validate(symbol, action, price, regime, context)
            return result
        except Exception as e:
            return {"approved": True, "confidence": 0.5, "reasoning": f"Validation error: {e}"}

    def _apply_regime_filter(
        self, symbol: str, df: pd.DataFrame,
        long_entries, long_exits, short_entries, short_exits
    ):
        if symbol not in self.regime_filters:
            return long_entries, long_exits, short_entries, short_exits

        rf = self.regime_filters[symbol]

        try:
            regimes = rf.predict(df)
            probs = rf.get_regime_probabilities(df)
        except Exception:
            return long_entries, long_exits, short_entries, short_exits

        strategy_type = INSTRUMENTS[symbol]["strategy"]
        if strategy_type == "mean_reversion":
            regime_label = "mean_reverting"
        else:
            regime_label = "trending"

        prob_series = probs.get(regime_label, pd.Series(0.5, index=regimes.index))

        trending_pct = (regimes == "trending").mean()
        if trending_pct < 0.1:
            regime_mask = pd.Series(True, index=regimes.index)
        else:
            regime_mask = create_regime_mask(
                regimes, regime_label, min_bars=3,
                probability=prob_series, prob_threshold=0.35,
            )

        regime_mask = regime_mask.reindex(long_entries.index, fill_value=False)

        filtered_long = long_entries & regime_mask
        filtered_long_exits = long_exits | (~regime_mask & long_entries.shift(1).fillna(False))
        filtered_short = short_entries & regime_mask
        filtered_short_exits = short_exits | (~regime_mask & short_entries.shift(1).fillna(False))

        return filtered_long, filtered_long_exits, filtered_short, filtered_short_exits
