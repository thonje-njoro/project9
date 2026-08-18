"""
config.py — Single source of truth for ALL parameters.
The AI loop patches this file. Never hardcode params in strategy files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Fall back to .env.example for development
    load_dotenv(Path(__file__).parent / ".env.example")

# ─── API Keys (from environment) ─────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
LSE_API_KEY = os.getenv("LSE_API_KEY", "")

# ─── Backtest Configuration ──────────────────────────────────────────────────
BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 50_000,
    "commission_stock": 0.0005,
    "commission_forex": 0.00002,
    "slippage_bps": 0.001,
    "min_ticket_fee": 1.00,
}

# ─── Risk Configuration ─────────────────────────────────────────────────────
RISK_CONFIG = {
    "max_exposure_pct": 0.25,       # Max position size as % of equity
    "risk_per_trade": 0.01,         # Risk 1% equity per trade
    "max_correlation": 0.70,        # Block new trade if corr with existing > this
    "regime_lookback_days": 63,     # UpFraction lookback
    "regime_entry_thresh": 0.55,    # Min UpFraction to allow entries
    "regime_exit_thresh": 0.45,     # UpFraction below which to exit
}

# ─── Prop Firm Rules ─────────────────────────────────────────────────────────
PROP_FIRM_RULES = {
    "ftmo_2step": {
        "profit_target_pct": 0.10,
        "max_drawdown_pct": 0.10,
        "daily_loss_pct": 0.05,
        "min_trading_days": 10,
        "daily_profit_cap_pct": 0.30,
    },
    "the5ers_high_stakes": {
        "profit_target_pct": 0.08,
        "max_drawdown_pct": 0.06,
        "daily_loss_pct": 0.03,
        "min_trading_days": 10,
        "daily_profit_cap_pct": 0.30,
    },
    "fundingpips": {
        "profit_target_pct": 0.08,
        "max_drawdown_pct": 0.10,
        "daily_loss_pct": 0.05,
        "min_trading_days": 5,
        "daily_profit_cap_pct": 0.30,
    },
}

# ─── Instruments ─────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "GLD":        {"asset_class": "equity_etf",  "strategy": "kalman_trend",       "timeframe": "4h",    "data_source": "alpaca"},
    "TLT":        {"asset_class": "bond_etf",    "strategy": "kalman_trend",       "timeframe": "4h",    "data_source": "alpaca"},
    "IWM":        {"asset_class": "equity_etf",  "strategy": "kalman_trend",       "timeframe": "4h",    "data_source": "alpaca"},
    "CPER":       {"asset_class": "equity_etf",  "strategy": "kalman_trend",       "timeframe": "4h",    "data_source": "alpaca"},
    "CPER_GLD":   {"asset_class": "synthetic",   "strategy": "cper_gld_ratio",     "timeframe": "4h",    "data_source": "alpaca"},
    "SPY":        {"asset_class": "equity_etf",  "strategy": "vwap_mean_reversion","timeframe": "15min", "data_source": "alpaca"},
    "IWM_VWAP":   {"asset_class": "equity_etf",  "strategy": "vwap_mean_reversion","timeframe": "15min", "data_source": "alpaca"},
    "SPY_ORB":    {"asset_class": "equity_etf",  "strategy": "orb_strategy",       "timeframe": "15min", "data_source": "alpaca"},
    "QQQ_ORB":    {"asset_class": "equity_etf",  "strategy": "orb_strategy",       "timeframe": "15min", "data_source": "alpaca"},
    "NVDA_MORB":  {"asset_class": "equity",      "strategy": "momentum_orb",       "timeframe": "5min",  "data_source": "alpaca"},
    "AMD_MORB":   {"asset_class": "equity",      "strategy": "momentum_orb",       "timeframe": "5min",  "data_source": "alpaca"},
    "PLTR_MORB":  {"asset_class": "equity",      "strategy": "momentum_orb",       "timeframe": "5min",  "data_source": "alpaca"},
    "MRVL_MORB":  {"asset_class": "equity",      "strategy": "momentum_orb",       "timeframe": "5min",  "data_source": "alpaca"},
    "XAUUSD_MR":  {"asset_class": "forex",       "strategy": "xauusd_session_mr",  "timeframe": "1h",    "data_source": "lse"},
}

# ─── Strategy Parameters ─────────────────────────────────────────────────────
STRATEGY_PARAMS = {
    "kalman_trend": {
        "GLD": {
            "Q": 0.01, "R": 1.0, "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True, "trail_atr_mult": 2.5,
            "long_only": True, "use_vwap_exit": True, "use_pullback_entry": False,
            "atr_period": 14, "vwap_period": 20,
        },
        "TLT": {
            "Q": 0.01, "R": 1.0, "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True, "trail_atr_mult": 2.5,
            "long_only": False, "use_vwap_exit": True, "use_pullback_entry": True,
            "pullback_max_candles": 2, "entry_window_periods": 3,
            "atr_period": 14, "vwap_period": 20,
        },
        "IWM": {
            "Q": 0.01, "R": 1.0, "velocity_threshold_pct": 0.10,
            "mean_revert": False, "use_trailing_stop": True, "trail_atr_mult": 2.5,
            "long_only": True, "use_vwap_exit": True,
            "atr_period": 14, "vwap_period": 20,
        },
        "CPER": {
            "Q": 0.02, "R": 1.0, "velocity_threshold_pct": 0.15,
            "mean_revert": False, "use_trailing_stop": True, "trail_atr_mult": 3.0,
            "long_only": True, "use_vwap_exit": True, "use_pullback_entry": True,
            "pullback_max_candles": 2, "entry_window_periods": 4,
            "atr_period": 14, "vwap_period": 20,
        },
    },
    "cper_gld_ratio": {
        "CPER_GLD": {
            "z_entry": 1.8, "z_exit": 0.0, "z_take_profit": 0.0,
            "window": 20, "use_trailing_stop": True, "trail_atr_mult": 2.0,
            "atr_period": 14,
        },
    },
    "orb_strategy": {
        "SPY_ORB": {
            "orb_period": 1, "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "rel_vol_lookback": 14, "min_rel_volume": 1.0, "atr_period": 14,
            "atr_stop_pct": 0.10, "risk_per_trade": 0.01,
            "min_price": 5.0, "min_avg_volume": 1_000_000,
        },
        "QQQ_ORB": {
            "orb_period": 1, "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "rel_vol_lookback": 14, "min_rel_volume": 0.8, "atr_period": 14,
            "atr_stop_pct": 0.10, "risk_per_trade": 0.01,
            "min_price": 5.0, "min_avg_volume": 100_000,
        },
    },
    "momentum_orb": {
        "NVDA_MORB": {
            "or_minutes": 60, "atr_mult_stop": 1.5, "trail_mult": 1.5,
            "min_price": 5.0, "min_volume": 100_000,
            "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "atr_period": 14,
        },
        "AMD_MORB": {
            "or_minutes": 60, "atr_mult_stop": 1.5, "trail_mult": 1.5,
            "min_price": 5.0, "min_volume": 100_000,
            "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "atr_period": 14,
        },
        "PLTR_MORB": {
            "or_minutes": 60, "atr_mult_stop": 2.0, "trail_mult": 2.0,
            "min_price": 5.0, "min_volume": 100_000,
            "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "atr_period": 14,
        },
        "MRVL_MORB": {
            "or_minutes": 5, "atr_mult_stop": 1.0, "trail_mult": 1.0,
            "min_price": 5.0, "min_volume": 100_000,
            "session_open_hour": 9, "session_open_minute": 30,
            "session_close_hour": 16, "tz": "US/Eastern",
            "atr_period": 14,
        },
    },
    "vwap_mean_reversion": {
        "SPY": {
            "z_score_lookback": 20, "z_entry": 2.0, "z_exit": 0.0,
            "require_reversal": True, "reversal_lookback": 3,
            "skip_trend_days": True, "adx_threshold": 25.0,
            "use_relative_volume": True, "max_relative_volume": 1.5,
            "use_atr_change_filter": True, "max_atr_change": 0.05,
            "use_time_filter": True, "entry_start_hour_et": 10, "entry_end_hour_et": 15,
            "long_only": True, "max_hold_bars": 48,
            "atr_period": 14, "trail_atr_mult": 2.0,
        },
        "IWM_VWAP": {
            "z_score_lookback": 20, "z_entry": 2.0, "z_exit": 0.0,
            "require_reversal": True, "reversal_lookback": 3,
            "skip_trend_days": True, "adx_threshold": 25.0,
            "use_relative_volume": True, "max_relative_volume": 1.5,
            "use_atr_change_filter": True, "max_atr_change": 0.05,
            "use_time_filter": True, "entry_start_hour_et": 10, "entry_end_hour_et": 15,
            "long_only": True, "max_hold_bars": 48,
            "atr_period": 14, "trail_atr_mult": 2.0,
        },
    },
    "xauusd_session_mr": {
        "XAUUSD_MR": {
            "z_entry": 1.5, "z_exit": 0.0,
            "london_start_utc": 8, "london_end_utc": 12,
            "ny_start_utc": 13, "ny_start_minute_utc": 30,
            "ny_end_utc": 17,
            "spread_pips": 1.5, "pip_value": 0.01,
            "asian_range_multiplier": 2.0,
            "atr_period": 14, "trail_atr_mult": 2.0,
            "use_trailing_stop": True,
        },
    },
}

# ─── Engine Configuration ────────────────────────────────────────────────────
ENGINE_CONFIG = {
    "use_regime_filter": True,
    "regime_bypass_if_coverage_lt": 0.10,  # Skip regime if <10% bars qualify
}

# ─── Data Configuration ──────────────────────────────────────────────────────
DATA_CONFIG = {
    "cache_dir": str(Path(__file__).parent / "data" / "cache"),
    "cache_freshness_hours": 24,
    "lse_rate_limit_seconds": 7,
    "alpaca_rate_limit_per_minute": 200,
    "alpaca_base_url": "https://paper-api.alpaca.markets",
    "lse_base_url": "https://api.londonstrategicedge.com/vault",
}

# ─── Validation Thresholds ───────────────────────────────────────────────────
VALIDATION_THRESHOLDS = {
    "min_oos_sharpe": 0.5,
    "min_deflated_sharpe": 0.95,
    "min_wfv_consistency": 0.60,
    "min_mc_survival_rate": 0.85,
    "max_is_drawdown": 0.15,
    "min_trades_is": 30,
    "mc_simulations": 2000,
    "mc_max_dd_kill": 0.12,
}
