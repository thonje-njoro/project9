"""State persistence for paper trading."""

import json
import pickle
import pandas as pd
from pathlib import Path


class StateManager:
    """Handles persistence of trade log, equity curve, regime state, positions."""

    def __init__(self, state_dir: str = "paper_trading/state/") -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = self.state_dir / "trade_log.csv"
        self.equity_path = self.state_dir / "equity_curve.csv"
        self.regime_path = self.state_dir / "regime_state.pkl"
        self.positions_path = self.state_dir / "open_positions.json"
        self.risk_path = self.state_dir / "risk_state.json"

    def append_trade(self, trade: dict) -> None:
        df = pd.DataFrame([trade])
        if self.trade_log_path.exists():
            df.to_csv(self.trade_log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.trade_log_path, index=False)

    def get_trade_log(self) -> pd.DataFrame:
        if self.trade_log_path.exists():
            return pd.read_csv(self.trade_log_path)
        return pd.DataFrame()

    def save_equity_snapshot(self, timestamp: str, equity: float, positions_value: float) -> None:
        row = pd.DataFrame([{
            "timestamp": timestamp,
            "equity": equity,
            "positions_value": positions_value,
        }])
        if self.equity_path.exists():
            row.to_csv(self.equity_path, mode="a", header=False, index=False)
        else:
            row.to_csv(self.equity_path, index=False)

    def load_equity_curve(self) -> pd.DataFrame:
        if self.equity_path.exists():
            return pd.read_csv(self.equity_path)
        return pd.DataFrame(columns=["timestamp", "equity", "positions_value"])

    def save_regime_state(self, regime_filters: dict) -> None:
        with open(self.regime_path, "wb") as f:
            pickle.dump(regime_filters, f)

    def load_regime_state(self) -> dict | None:
        if self.regime_path.exists():
            with open(self.regime_path, "rb") as f:
                return pickle.load(f)
        return None

    def save_positions(self, positions: dict) -> None:
        with open(self.positions_path, "w") as f:
            json.dump(positions, f, indent=2, default=str)

    def load_positions(self) -> dict:
        if self.positions_path.exists():
            with open(self.positions_path, "r") as f:
                return json.load(f)
        return {}

    def save_risk_state(self, risk_data: dict) -> None:
        with open(self.risk_path, "w") as f:
            json.dump(risk_data, f, indent=2, default=str)

    def load_risk_state(self) -> dict:
        if self.risk_path.exists():
            with open(self.risk_path, "r") as f:
                return json.load(f)
        return {}
