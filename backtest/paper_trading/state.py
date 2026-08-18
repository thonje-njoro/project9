"""
paper_trading/state.py — Persist equity curve + positions to JSON.
Survives VM restarts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).parent / "state"


class StateManager:
    """
    Persists paper trading state to disk.
    Files:
        state/equity_curve.json — daily equity history
        state/positions.json — current open positions
        state/trade_log.json — completed trades
        state/config_snapshot.json — last applied config
    """

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(self, filename: str) -> dict | list:
        """Load JSON file, return empty if missing."""
        path = self.state_dir / filename
        if not path.exists():
            return {} if filename.endswith("positions.json") or filename.endswith("config_snapshot.json") else []
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {} if "positions" in filename or "config" in filename else []

    def _save_json(self, filename: str, data):
        """Save data to JSON file."""
        path = self.state_dir / filename
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")

    # ─── Equity Curve ────────────────────────────────────────────────────────

    def append_equity(self, equity: float, timestamp: str | None = None):
        """Append a point to the equity curve."""
        curve = self._load_json("equity_curve.json")
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        curve.append({"timestamp": timestamp, "equity": equity})
        self._save_json("equity_curve.json", curve)

    def get_equity_curve(self) -> list:
        """Get full equity curve history."""
        return self._load_json("equity_curve.json")

    # ─── Positions ───────────────────────────────────────────────────────────

    def save_positions(self, positions: dict):
        """Save current open positions."""
        self._save_json("positions.json", positions)

    def get_positions(self) -> dict:
        """Get saved positions."""
        return self._load_json("positions.json")

    # ─── Trade Log ───────────────────────────────────────────────────────────

    def log_trade(self, symbol: str, side: str, qty: float,
                  entry_price: float, exit_price: float,
                  pnl: float, timestamp: str | None = None):
        """Log a completed trade."""
        trades = self._load_json("trade_log.json")
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        trades.append({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "timestamp": timestamp,
        })
        self._save_json("trade_log.json", trades)

    def get_trade_log(self) -> list:
        """Get full trade log."""
        return self._load_json("trade_log.json")

    # ─── Config Snapshot ─────────────────────────────────────────────────────

    def save_config_snapshot(self, config: dict):
        """Save a snapshot of the current config."""
        self._save_json("config_snapshot.json", {
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
        })

    def get_config_snapshot(self) -> dict:
        """Get last saved config snapshot."""
        return self._load_json("config_snapshot.json")

    # ─── Heartbeat ───────────────────────────────────────────────────────────

    def save_heartbeat(self, status: str = "ok"):
        """Save heartbeat timestamp."""
        self._save_json("heartbeat.json", {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
        })

    def get_last_heartbeat(self) -> dict:
        """Get last heartbeat."""
        return self._load_json("heartbeat.json")
