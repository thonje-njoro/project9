"""Paper trading configuration."""

PAPER_TRADING_CONFIG = {
    "dry_run": False,
    "state_dir": "paper_trading/state/",
    "lookback_bars": {
        "SPY": 5000,
        "QQQ": 5000,
        "BTC/USD": 5000,
        "GLD": 1500,
        "USO": 1500,
    },
}
