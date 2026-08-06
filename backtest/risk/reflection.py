"""Post-trade reflection system.

Tracks trade outcomes and generates lessons that improve future signal filtering.
Inspired by TradingAgents' memory/reflection loop.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


REFLECTION_DIR = Path(__file__).parent.parent / "data" / "reflections"


class TradeReflector:
    """Tracks trades and generates lessons for future signal filtering."""

    def __init__(self, reflection_dir: Optional[Path] = None):
        self.reflection_dir = reflection_dir or REFLECTION_DIR
        self.reflection_dir.mkdir(parents=True, exist_ok=True)
        self.trades: list[dict] = []
        self._load()

    def _trades_path(self) -> Path:
        return self.reflection_dir / "trades.json"

    def _lessons_path(self, symbol: str) -> Path:
        return self.reflection_dir / f"lessons_{symbol.replace('/', '_')}.json"

    def _load(self):
        path = self._trades_path()
        if path.exists():
            try:
                self.trades = json.loads(path.read_text())
            except Exception:
                self.trades = []

    def save(self):
        path = self._trades_path()
        path.write_text(json.dumps(self.trades, indent=2, default=str))

    def log_trade(
        self,
        symbol: str,
        entry_date: str,
        exit_date: str,
        return_pct: float,
        regime: str,
        strategy: str,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        """Log a completed trade for reflection."""
        trade = {
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "return_pct": return_pct,
            "regime": regime,
            "strategy": strategy,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "logged_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.trades.append(trade)

    def get_symbol_trades(self, symbol: str) -> list[dict]:
        return [t for t in self.trades if t["symbol"] == symbol]

    def compute_reflection(self, symbol: str, lookback: int = 20) -> dict:
        """Analyze recent trades and generate lessons.

        Returns:
            dict with keys: win_rate, avg_return, regime_performance, lessons
        """
        trades = self.get_symbol_trades(symbol)[-lookback:]
        if not trades:
            return {"win_rate": 0.5, "avg_return": 0.0, "regime_performance": {}, "lessons": []}

        returns = [t["return_pct"] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / max(len(returns), 1)
        avg_return = np.mean(returns) if returns else 0.0

        regime_perf = {}
        for t in trades:
            r = t.get("regime", "unknown")
            if r not in regime_perf:
                regime_perf[r] = []
            regime_perf[r].append(t["return_pct"])

        regime_avg = {k: np.mean(v) for k, v in regime_perf.items()}

        lessons = []

        if win_rate < 0.3 and len(returns) >= 5:
            lessons.append({
                "type": "low_win_rate",
                "message": f"Win rate is {win_rate:.0%} over last {len(returns)} trades. Consider tighter entry filters.",
                "severity": "high",
            })

        for regime, avg in regime_avg.items():
            if avg < -1.0 and len(regime_perf[regime]) >= 3:
                lessons.append({
                    "type": "regime_loss",
                    "message": f"Avg return in '{regime}' regime is {avg:.1f}%. Avoid trading in this regime.",
                    "regime": regime,
                    "severity": "medium",
                })

        if len(losses) >= 3:
            recent_losses = losses[-3:]
            if all(r < -2.0 for r in recent_losses):
                lessons.append({
                    "type": "consecutive_losses",
                    "message": "3 consecutive losses > 2%. Skip next signal or reduce size.",
                    "severity": "high",
                })

        return {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "regime_performance": regime_avg,
            "lessons": lessons,
        }

    def save_lessons(self, symbol: str):
        """Compute and save lessons for a symbol."""
        reflection = self.compute_reflection(symbol)
        path = self._lessons_path(symbol)
        path.write_text(json.dumps(reflection, indent=2, default=str))

    def get_lessons(self, symbol: str) -> list[dict]:
        """Load previously saved lessons for a symbol."""
        path = self._lessons_path(symbol)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return data.get("lessons", [])
        except Exception:
            return []
