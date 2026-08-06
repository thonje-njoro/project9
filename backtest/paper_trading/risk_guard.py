"""Real-time prop firm risk enforcement."""

from datetime import datetime, date


class RiskGuard:
    """Enforces daily DD, max DD, and consistency rules in real-time."""

    def __init__(self, rules: dict, state_dir: str = "paper_trading/state/") -> None:
        self.rules = rules
        self.initial_equity = 0.0
        self.peak_equity = 0.0
        self.equity_history: list[tuple[str, float]] = []
        self.day_start_equity: float | None = None
        self._current_date: date | None = None

    def initialize(self, initial_equity: float, equity_history: list | None = None) -> None:
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.equity_history = equity_history or []
        if equity_history:
            self.peak_equity = max(eq for _, eq in equity_history)

        today = date.today()
        self._current_date = today
        for ts, eq in reversed(self.equity_history):
            if not ts.startswith(str(today)):
                self.day_start_equity = eq
                break
        if self.day_start_equity is None:
            self.day_start_equity = initial_equity

    def update_equity(self, timestamp: str, equity: float) -> None:
        self.equity_history.append((timestamp, equity))
        if equity > self.peak_equity:
            self.peak_equity = equity

        today = date.today()
        if self._current_date != today:
            self._current_date = today
            self.day_start_equity = equity
        elif self.day_start_equity is None:
            self.day_start_equity = equity

        if len(self.equity_history) > 200:
            self.equity_history = self.equity_history[-200:]

    def check_daily_drawdown(self, equity: float) -> bool:
        if self.day_start_equity is None or self.day_start_equity == 0:
            return True
        dd = (equity - self.day_start_equity) / self.day_start_equity
        allowed = dd >= -self.rules["daily_drawdown_pct"]
        if not allowed:
            print(f"RISK BLOCK: daily drawdown {dd*100:.2f}% exceeds "
                  f"{self.rules['daily_drawdown_pct']*100:.0f}% limit")
        return allowed

    def check_max_drawdown(self, equity: float) -> bool:
        if self.peak_equity == 0:
            return True
        dd = (equity - self.peak_equity) / self.peak_equity
        allowed = dd >= -self.rules["max_drawdown_pct"]
        if not allowed:
            print(f"RISK BLOCK: max drawdown {dd*100:.2f}% exceeds "
                  f"{self.rules['max_drawdown_pct']*100:.0f}% limit")
        return allowed

    def check_consistency(self, equity: float, proposed_profit: float = 0.0) -> bool:
        total_profit = equity - self.initial_equity
        if total_profit <= 0:
            return True

        if self.day_start_equity is None:
            day_pnl = 0.0
        else:
            day_pnl = equity - self.day_start_equity

        projected_day_pnl = day_pnl + proposed_profit
        ratio = abs(projected_day_pnl) / abs(total_profit) if total_profit != 0 else 0
        allowed = ratio <= self.rules["consistency_block_pct"]
        if not allowed:
            print(f"RISK BLOCK: consistency ratio {ratio*100:.1f}% exceeds "
                  f"{self.rules['consistency_block_pct']*100:.0f}% limit")
        return allowed

    def can_trade(self, equity: float, proposed_profit: float = 0.0) -> bool:
        return (
            self.check_daily_drawdown(equity)
            and self.check_max_drawdown(equity)
            and self.check_consistency(equity, proposed_profit)
        )
