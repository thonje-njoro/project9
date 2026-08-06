"""Core backtesting engine using vectorbt.

Includes all 7 improvements:
- Item 1: Fractional Kelly position sizing
- Item 2: Risk parity allocation
- Item 3: 3-state HMM regime filter
- Item 4: VWAP trailing exits
- Item 5: Bootstrap diagnostics (in reporting)
- Item 6: GARCH volatility forecast for stops
- Item 7: Cointegration pairs (constructed in main.py)
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
from typing import Optional

from config import INSTRUMENTS, STRATEGY_PARAMS, BACKTEST_CONFIG, RISK_CONFIG, SYSTEM_HEALTH_CONFIG, ENTRY_FILTERS, REFLECTION_CONFIG, ENGINE_CONFIG
from risk.position_sizer import compute_atr, atr_position_sizes, compute_kelly_from_trades, compute_kelly_fraction
from risk.regime_filter import RegimeFilter, create_regime_mask, compute_regime_stop_multipliers
from risk.vol_regime_stops import compute_vol_calibrated_stop
from risk.system_health_monitor import SystemHealthMonitor
from risk.portfolio_optimizer import compute_risk_parity_allocation
from optimization.entry_filters import apply_filters


class BacktestEngine:
    def __init__(self, data: dict[str, pd.DataFrame], config: dict, use_regime_filter: bool = True) -> None:
        self.data = data
        self.config = config
        self.portfolios: dict[str, vbt.Portfolio] = {}
        self.use_regime_filter = use_regime_filter
        self.regime_filters: dict[str, RegimeFilter] = {}
        self.kelly_factors: dict[str, float] = {}  # Item 1: per-instrument Kelly mult
        self.risk_parity_weights: Optional[pd.Series] = None  # Item 2

    def _fit_regime_filters(self) -> None:
        """Fit HMM regime filters for each instrument. 3-state when configured."""
        if not self.use_regime_filter:
            return
        n_states = ENGINE_CONFIG.get("hmm_states", 2)
        use_garch = ENGINE_CONFIG.get("use_garch", True)
        print(f"Fitting {n_states}-state HMM regime filters (GARCH={'on' if use_garch else 'off'})...")
        for symbol, df in self.data.items():
            info = INSTRUMENTS.get(symbol, {})
            if info.get("asset_class") == "synthetic" and info.get("strategy") == "cointegration_pair":
                # Cointegration pairs need a RegimeFilter too
                rf = RegimeFilter(n_states=n_states, lookback=60, use_garch=use_garch)
            else:
                rf = RegimeFilter(n_states=n_states, lookback=60, use_garch=use_garch)
            rf.fit(df)
            self.regime_filters[symbol] = rf
            regimes = rf.predict(df)
            regime_counts = regimes.value_counts()
            print(f"  {symbol}: {dict(regime_counts)}")

    def _import_signals(self, strategy: str):
        if strategy == "mean_reversion":
            from strategies.mean_reversion import generate_signals
        elif strategy == "momentum_breakout":
            from strategies.momentum_breakout import generate_signals
        elif strategy == "trend_following":
            from strategies.trend_following import generate_signals
        elif strategy == "crypto_momentum":
            from strategies.crypto_momentum import generate_signals
        elif strategy == "gold_oil_ratio":
            from strategies.gold_oil_ratio import generate_signals
        elif strategy == "short_volatility":
            from strategies.short_volatility import generate_signals
        elif strategy == "kalman_trend":
            from strategies.kalman_trend import generate_signals
        elif strategy == "cointegration_pair":
            from strategies.cointegration_pair import generate_signals
        elif strategy == "tsmom":
            from strategies.tsmom import generate_signals
        elif strategy == "cper_gld_ratio":
            from strategies.cper_gld_ratio import generate_signals
        elif strategy == "prop_firm_sprint":
            from strategies.prop_firm_sprint import generate_signals
        elif strategy == "vwap_mean_reversion":
            from strategies.vwap_mean_reversion import generate_signals
        elif strategy == "orb":
            from strategies.orb_strategy import generate_signals
        elif strategy == "xauusd_session_mr":
            from strategies.xauusd_session_mr import generate_signals
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return generate_signals

    def _verify_signals(self, entries: pd.Series, name: str) -> None:
        if len(entries) > 0 and entries.iloc[0]:
            raise ValueError(
                f"Signal '{name}' fires on bar 0 — lookahead bias detected. "
                f"All signals must be shifted by 1 bar."
            )

    def _apply_regime_filter(
        self,
        symbol: str,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Apply regime-based signal filtering with 2-state or 3-state HMM.

        Returns (long_entries, long_exits, short_entries, short_exits, size_scaling).
        size_scaling is a Series of multipliers (0.0-1.0) applied to position sizes.

        Key improvements vs hard binary mask:
          - Never forces exits on regime change (preserves winning trends)
          - Scales position size smoothly by regime confidence
          - Crisis regime reduces positions progressively, doesn't block entirely
        """
        # Default: no scaling
        size_scaling = pd.Series(1.0, index=long_entries.index)

        if not self.use_regime_filter or symbol not in self.regime_filters:
            return long_entries, long_exits, short_entries, short_exits, size_scaling

        rf = self.regime_filters[symbol]

        if SYSTEM_HEALTH_CONFIG.get("use_kl_enhanced_regime", False):
            probs = rf.get_kl_enhanced_probabilities(self.data[symbol])
        else:
            probs = rf.get_regime_probabilities(self.data[symbol])
        regimes = rf.predict(self.data[symbol])

        strategy_type = INSTRUMENTS[symbol]["strategy"]

        # Determine which regime label this strategy prefers
        if strategy_type == "mean_reversion":
            regime_label = "mean_reverting"
        elif strategy_type in ("momentum_breakout", "crypto_momentum", "trend_following", "tsmom"):
            regime_label = "trending"
        elif strategy_type == "kalman_trend":
            params = STRATEGY_PARAMS.get(strategy_type, {}).get(symbol, {})
            regime_label = "mean_reverting" if params.get("mean_revert", False) else "trending"
        elif strategy_type in ("gold_oil_ratio", "cointegration_pair", "vwap_mean_reversion"):
            regime_label = "mean_reverting"
        else:
            regime_label = "mean_reverting"

        # === SOFT SCALING: position size multiplier based on regime confidence ===
        # Blend: probability of preferred regime gives full size, crisis gives reduced

        # Start with preferred regime probability
        if regime_label in probs.columns:
            preferred_prob = probs[regime_label].reindex(size_scaling.index, fill_value=0.5)
        else:
            preferred_prob = pd.Series(0.5, index=size_scaling.index)

        # Crisis override: reduce size during crisis
        if "crisis" in probs.columns:
            crisis_prob = probs["crisis"].reindex(size_scaling.index, fill_value=0.0)
            # Scale: at 0% crisis → 1.0x, at 100% crisis → 0.3x
            crisis_factor = 1.0 - crisis_prob * 0.7
        else:
            crisis_factor = pd.Series(1.0, index=size_scaling.index)

        # Final scaling: blend preferred regime prob with crisis factor
        # If P(preferred) is high → full size. If crisis → reduce smoothly.
        # Map P(preferred) from [0, 1] to [0.25, 1.0] — never fully block
        regime_scale = 0.25 + 0.75 * preferred_prob.clip(0, 1)
        size_scaling = (regime_scale * crisis_factor).clip(0.1, 1.0)

        return long_entries, long_exits, short_entries, short_exits, size_scaling

    def _generate_portfolio_signals(
        self, symbol: str, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate and filter signals for a single instrument."""
        info = INSTRUMENTS[symbol]
        strategy = info["strategy"]
        params = STRATEGY_PARAMS[strategy][symbol]
        signal_params = {
            k: v for k, v in params.items()
            if k not in ("use_vol_calibrated_stop",
                         "vol_stop_base_mult", "vol_stop_min_mult", "vol_stop_max_mult",
                         "use_trailing_stop")
        }

        generate_signals = self._import_signals(strategy)

        if strategy == "cointegration_pair" and "pair_src" in info:
            # Cointegration pair needs the spread df (already in self.data[symbol])
            # and the hedge instrument data for reference
            hedge_sym = info["pair_hedge"]
            hedge_df = self.data.get(hedge_sym, df)
            result = generate_signals(df, hedge_df, pair_name=symbol, **signal_params)
        else:
            result = generate_signals(df, **signal_params)

        if len(result) == 5:
            long_entries, long_exits, short_entries, short_exits, strategy_trailing = result
        else:
            long_entries, long_exits, short_entries, short_exits = result
            strategy_trailing = None

        # Apply regime filter — soft scaling (affects position sizes, not entries/exits)
        long_entries, long_exits, short_entries, short_exits, regime_scaling = self._apply_regime_filter(
            symbol, long_entries, long_exits, short_entries, short_exits
        )

        # Apply entry filters
        strategy_filters = ENTRY_FILTERS.get(strategy, [])
        if strategy_filters:
            long_entries = apply_filters(long_entries, df, strategy_filters, symbol=symbol)
            short_entries = apply_filters(short_entries, df, strategy_filters, symbol=symbol)

        # Position sizing (ATR-based + Kelly scaling + regime scaling)
        atr = compute_atr(df, RISK_CONFIG["atr_period"])
        kelly_mult = self.kelly_factors.get(symbol, 1.0)

        position_sizes = atr_position_sizes(
            self.config["initial_capital"], atr, df["close"],
            RISK_CONFIG["max_risk_per_trade_pct"],
            RISK_CONFIG.get("max_exposure_pct", 0.25),
            kelly_mult=kelly_mult,
        )

        # Apply regime-based position scaling (soft overlay, never full block)
        regime_scaling = regime_scaling.reindex(position_sizes.index, fill_value=1.0)
        position_sizes = position_sizes * regime_scaling

        # Trailing stop (Items 4 & 6 — VWAP exit handled in strategy, GARCH-adjusted stops)
        trailing_stop = None
        if params.get("use_vol_calibrated_stop", False) and symbol in self.regime_filters:
            # GARCH-enhanced vol-calibrated stop (Item 6)
            regime_probs = self.regime_filters[symbol].get_regime_probabilities(df)
            trend_probs = regime_probs.get("trending", pd.Series(0.5, index=df.index))

            # Check for crisis regime — tighten stops
            if "crisis" in regime_probs.columns:
                crisis_probs = regime_probs["crisis"]
                adjusted_base = params.get("vol_stop_base_mult", 3.0) * (1 - crisis_probs * 0.5)
            else:
                adjusted_base = params.get("vol_stop_base_mult", 3.0)

            trailing_stop = compute_vol_calibrated_stop(
                df, atr, trend_probs,
                base_mult=adjusted_base,
                vol_scale_range=(
                    params.get("vol_stop_min_mult", 1.8) / adjusted_base,
                    params.get("vol_stop_max_mult", 5.4) / adjusted_base,
                ),
            )
        elif strategy_trailing is not None and strategy_trailing.sum() > 0:
            trailing_stop = strategy_trailing
        elif "trail_atr_mult" in params and params.get("trail_atr_mult", 0) > 0:
            trailing_stop = atr * params["trail_atr_mult"]

        # NOTE: For SPY 15-min mean reversion, set trail_atr_mult=0 (or not in params)
        # to disable trailing stop. The 0.28% trail on 15-min bars killed 80% of
        # winners by exiting before reversion completed. Pure SMA-cross exit works
        # better: 63% win rate, 1.177 PF at 0.01% fee.

        return long_entries, long_exits, short_entries, short_exits, position_sizes, trailing_stop

    def _compute_kelly_factors(self) -> dict[str, float]:
        """Item 1: Compute Kelly fractions from individual backtest trade stats."""
        factors = {}
        print("\nComputing Kelly position sizing factors...")
        for symbol, pf in self.portfolios.items():
            trades = pf.trades.count()
            if trades < RISK_CONFIG.get("kelly_min_trades", 5):
                factors[symbol] = 1.0  # not enough data, use ATR-only
                print(f"  {symbol:14s}  only {trades} trades — using ATR-only (kelly=1.0)")
                continue

            try:
                # vectorbt v1: access returns via records_readable
                records = pf.trades.records_readable
                if len(records) < 3:
                    factors[symbol] = 1.0
                    print(f"  {symbol:14s}  only {len(records)} trades — using ATR-only (kelly=1.0)")
                    continue
                # Extract returns column (field name varies by vectorbt version)
                ret_col = [c for c in records.columns if 'return' in c.lower()]
                if not ret_col:
                    factors[symbol] = 1.0
                    continue
                returns = records[ret_col[0]].values.astype(float)
                returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
                if len(returns) < 3:
                    factors[symbol] = 1.0
                    continue
                win_rate, avg_win, avg_loss, kelly = compute_kelly_from_trades(
                    pd.Series(returns), kelly_fraction=RISK_CONFIG.get("kelly_fraction", 0.25)
                )
                factors[symbol] = kelly
                kelly_label = f"kelly={kelly:.2f}x" if kelly > 0 else "SKIP"
                print(f"  {symbol:14s}  WR={win_rate:.0%} avgW={avg_win:.4f} avgL={avg_loss:.4f}  {kelly_label}")
            except Exception as e:
                print(f"  {symbol:14s}  Kelly calc error: {e}")
                factors[symbol] = 1.0

        return factors

    def _log_reflections(self, portfolios: dict[str, vbt.Portfolio]) -> None:
        """Log trade outcomes for reflection system."""
        if not REFLECTION_CONFIG.get("enabled", False):
            return

        try:
            from risk.reflection import TradeReflector
            reflector = TradeReflector()

            for symbol, pf in portfolios.items():
                if pf.trades.count() == 0:
                    continue

                records = pf.trades.records_readable
                for _, rec in records.iterrows():
                    entry_date = str(rec.get("Entry Timestamp", ""))
                    exit_date = str(rec.get("Exit Timestamp", ""))
                    return_pct = float(rec.get("Return", 0.0)) * 100

                    regime = "unknown"
                    if symbol in self.regime_filters:
                        try:
                            regimes = self.regime_filters[symbol].predict(self.data[symbol])
                            regime = str(regimes.iloc[-1])
                        except Exception:
                            pass

                    strategy = INSTRUMENTS[symbol]["strategy"]
                    reflector.log_trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        return_pct=return_pct,
                        regime=regime,
                        strategy=strategy,
                    )

                reflector.save_lessons(symbol)

            print("  Reflection lessons saved.")
        except Exception as e:
            print(f"  Warning: Reflection logging failed: {e}")

    def run(self) -> dict[str, vbt.Portfolio]:
        """Run individual instrument backtests.

        Phase 1: Run all instruments with kelly_mult=1.0 to collect trade stats.
        Phase 2: Compute Kelly factors from those stats (out-of-sample relative
                 to the combined portfolio — Kelly is NOT fed back into individuals).

        Returns:
            dict of symbol -> vbt.Portfolio
        """
        self._fit_regime_filters()

        results: dict[str, vbt.Portfolio] = {}
        for symbol, df in self.data.items():
            (long_entries, long_exits, short_entries, short_exits,
             position_sizes, trailing_stop) = self._generate_portfolio_signals(symbol, df)

            self._verify_signals(long_entries, f"{symbol}_long_entries")

            freq_map = {"15Min": "15min", "1H": "1h", "4H": "4h", "1h": "1h", "1D": "1d"}
            info = INSTRUMENTS[symbol]
            freq = freq_map.get(info["target_tf"], "1min")

            # Asset-class-specific commission (slippage baked in)
            ac = info["asset_class"]
            if ac == "forex":
                fee_rate = self.config.get("commission_forex", 0.00002)
            elif ac == "crypto":
                fee_rate = self.config.get("commission_crypto", 0.0010)
            else:
                fee_rate = self.config.get("commission_stock", 0.0005)
            slippage_bps = self.config.get("slippage_bps", 0.001)
            total_fee = fee_rate + slippage_bps

            kwargs = {
                "close": df["close"],
                "entries": long_entries,
                "exits": long_exits,
                "short_entries": short_entries,
                "short_exits": short_exits,
                "size": position_sizes,
                "size_type": "amount",
                "init_cash": self.config["initial_capital"],
                "fees": total_fee,
                "freq": freq,
                "accumulate": False,
            }
            if trailing_stop is not None:
                # Compute a stable trail percentage from the median ATR/close ratio
                trail_pct_series = trailing_stop / df["close"]
                stable_trail_pct = float(trail_pct_series.median())
                stable_trail_pct = max(min(stable_trail_pct, 0.15), 0.002)
                kwargs["sl_stop"] = stable_trail_pct
                kwargs["sl_trail"] = True

            results[symbol] = vbt.Portfolio.from_signals(**kwargs)
            print(f"  {symbol}: {results[symbol].trades.count()} trades, "
                  f"return={results[symbol].total_return()*100:.1f}%")

        self.portfolios = results

        # Compute Kelly factors from individual results (for use in combined only)
        if RISK_CONFIG.get("use_kelly_sizing", False):
            self.kelly_factors = self._compute_kelly_factors()
        else:
            self.kelly_factors = {sym: 1.0 for sym in results}

        self._log_reflections(results)
        return results

    def _run_single(self, symbol: str, df: pd.DataFrame, kelly_mult: float = 1.0) -> vbt.Portfolio:
        """Run a single-instrument backtest with a specific Kelly multiplier.

        Used internally by parameter sweep and walk-forward validation.
        Does not populate self.portfolios or self.kelly_factors.
        """
        (long_entries, long_exits, short_entries, short_exits,
         position_sizes, trailing_stop) = self._generate_portfolio_signals(symbol, df)

        self._verify_signals(long_entries, f"{symbol}_long_entries")

        freq_map = {"15Min": "15min", "1H": "1h", "4H": "4h", "1h": "1h", "1D": "1d"}
        info = INSTRUMENTS[symbol]
        freq = freq_map.get(info["target_tf"], "1min")

        ac = info["asset_class"]
        if ac == "forex":
            fee_rate = self.config.get("commission_forex", 0.00002)
        elif ac == "crypto":
            fee_rate = self.config.get("commission_crypto", 0.0010)
        else:
            fee_rate = self.config.get("commission_stock", 0.0005)
        slippage_bps = self.config.get("slippage_bps", 0.001)
        total_fee = fee_rate + slippage_bps

        kwargs = {
            "close": df["close"],
            "entries": long_entries,
            "exits": long_exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
            "size": position_sizes * kelly_mult,
            "size_type": "amount",
            "init_cash": self.config["initial_capital"],
            "fees": total_fee,
            "freq": freq,
            "accumulate": False,
        }
        if trailing_stop is not None:
            trail_pct_series = trailing_stop / df["close"]
            stable_trail_pct = float(trail_pct_series.median())
            stable_trail_pct = max(min(stable_trail_pct, 0.15), 0.002)
            kwargs["sl_stop"] = stable_trail_pct
            kwargs["sl_trail"] = True

        return vbt.Portfolio.from_signals(**kwargs)

    def run_combined(self) -> vbt.Portfolio:
        """Run combined portfolio backtest with risk parity allocation (Item 2).

        Uses risk parity weights when ENGINE_CONFIG['use_risk_parity'] is True.
        Otherwise falls back to equal-weight with correlation reduction.
        """
        all_close, all_entries, all_exits = [], [], []
        all_short_entries, all_short_exits, all_sizes = [], [], []
        symbols = list(self.data.keys())

        # Build daily returns for risk parity calculation
        daily_returns = {}

        for symbol in symbols:
            df = self.data[symbol]
            (long_entries, long_exits, short_entries, short_exits,
             position_sizes, _) = self._generate_portfolio_signals(symbol, df)

            all_close.append(df["close"])
            all_entries.append(long_entries)
            all_exits.append(long_exits)
            all_short_entries.append(short_entries)
            all_short_exits.append(short_exits)

            # Position sizes already have Kelly scaling from _generate_portfolio_signals
            all_sizes.append(position_sizes)

            # Collect daily returns for risk parity
            try:
                dr = df["close"].resample("1D").last().pct_change().dropna()
                daily_returns[symbol] = dr
            except Exception:
                pass

        combined_close = pd.concat(all_close, axis=1, keys=symbols)
        combined_entries = pd.concat(all_entries, axis=1, keys=symbols)
        combined_exits = pd.concat(all_exits, axis=1, keys=symbols)
        combined_short_entries = pd.concat(all_short_entries, axis=1, keys=symbols)
        combined_short_exits = pd.concat(all_short_exits, axis=1, keys=symbols)
        combined_sizes = pd.concat(all_sizes, axis=1, keys=symbols)

        combined_close = combined_close.ffill()
        combined_exits = combined_exits.fillna(False).infer_objects(copy=False)
        combined_short_exits = combined_short_exits.fillna(False).infer_objects(copy=False)
        combined_entries = combined_entries.fillna(False).infer_objects(copy=False)
        combined_short_entries = combined_short_entries.fillna(False).infer_objects(copy=False)
        combined_sizes = combined_sizes.fillna(0)

        # Item 2: Risk parity weighting (overrides the old SPY/QQQ correlation reduction)
        if ENGINE_CONFIG.get("use_risk_parity", False) and len(daily_returns) > 1:
            rp_lookback = ENGINE_CONFIG.get("risk_parity_lookback", 60)
            rp_max_weight = ENGINE_CONFIG.get("risk_parity_max_weight", 0.35)
            rp_min_weight = ENGINE_CONFIG.get("risk_parity_min_weight", 0.03)

            # Build daily returns DataFrame, filtered to available symbols
            dr_df = pd.DataFrame(daily_returns)
            available_syms = [s for s in symbols if s in dr_df.columns]

            if len(available_syms) >= 2:
                rp_weights = compute_risk_parity_allocation(
                    dr_df, available_syms,
                    lookback=rp_lookback,
                    max_weight=rp_max_weight,
                    min_weight=rp_min_weight,
                )
                self.risk_parity_weights = rp_weights
                print(f"\n  Risk parity weights:")
                for sym, w in rp_weights.items():
                    print(f"    {sym:14s}  {w:.1%}")
                    if sym in combined_sizes.columns:
                        combined_sizes[sym] = combined_sizes[sym] * w
        else:
            # Fallback: correlation reduction for overlapping positions (original logic)
            spy_active = combined_entries.get("SPY", pd.Series(False, index=combined_close.index)) | \
                         combined_short_entries.get("SPY", pd.Series(False, index=combined_close.index))
            qqq_active = combined_entries.get("QQQ", pd.Series(False, index=combined_close.index)) | \
                         combined_short_entries.get("QQQ", pd.Series(False, index=combined_close.index))
            spy_qqq_overlap = spy_active & qqq_active

            corr_reduction = RISK_CONFIG.get("correlation_reduction", 0.5)
            if "SPY" in combined_sizes.columns:
                combined_sizes.loc[spy_qqq_overlap, "SPY"] *= corr_reduction
            if "QQQ" in combined_sizes.columns:
                combined_sizes.loc[spy_qqq_overlap, "QQQ"] *= corr_reduction

        # Limit concurrent positions
        max_pos = RISK_CONFIG.get("max_concurrent_positions", 5)
        active = (combined_entries | combined_short_entries).sum(axis=1)
        over_limit = active > max_pos
        if over_limit.any():
            scale = pd.Series(1.0, index=combined_close.index)
            scale[over_limit] = max_pos / active[over_limit]
            for col in combined_sizes.columns:
                combined_sizes[col] *= scale

        # Compute blended fee rate for combined portfolio (weighted by allocation)
        ac_counts = {}
        for sym in symbols:
            ac = INSTRUMENTS[sym]["asset_class"]
            ac_counts[ac] = ac_counts.get(ac, 0) + 1
        total = len(symbols)
        blended_fee = 0.0
        for ac, count in ac_counts.items():
            if ac == "forex":
                fr = self.config.get("commission_forex", 0.00002)
            elif ac == "crypto":
                fr = self.config.get("commission_crypto", 0.0010)
            else:
                fr = self.config.get("commission_stock", 0.0005)
            blended_fee += fr * (count / total)
        blended_fee += self.config.get("slippage_bps", 0.001)

        combined = vbt.Portfolio.from_signals(
            close=combined_close,
            entries=combined_entries,
            exits=combined_exits,
            short_entries=combined_short_entries,
            short_exits=combined_short_exits,
            size=combined_sizes,
            size_type="amount",
            cash_sharing=True,
            call_seq="auto",
            init_cash=self.config["initial_capital"],
            fees=blended_fee,
            freq="1min",
            accumulate=False,
        )
        print(f"  Combined: {combined.trades.count()} trades, "
              f"return={combined.total_return()*100:.1f}%")

        # --- Drawdown kill switch check ---
        eq = combined.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd_pct = float(dd.min() * 100)

        # Determine if kill switch would have triggered
        HALVE_THRESHOLD = -0.05   # -5% → halve sizes
        STOP_THRESHOLD = -0.07    # -7% → stop trading
        would_halve = (dd < HALVE_THRESHOLD).any()
        would_stop = (dd < STOP_THRESHOLD).any()
        kill_switch_msg = ""
        if would_stop:
            kill_switch_msg = f" WARNING: -7% DD breached — kill switch would have stopped trading"
        elif would_halve:
            kill_switch_msg = f" NOTE: -5% DD breached — kill switch would have halved positions"

        # Compute time spent in drawdown > -5%
        dd_days = (dd < HALVE_THRESHOLD).sum()
        total_days = len(dd)
        dd_pct_time = dd_days / total_days * 100 if total_days > 0 else 0

        print(f"  Max drawdown: {max_dd_pct:.2f}% | "
              f"Time DD < -5%: {dd_pct_time:.1f}% of days"
              f"{kill_switch_msg}")

        return combined
