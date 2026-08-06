"""Challenge lifecycle manager — phase-aware risk scaling, circuit breakers, and progress tracking.

Designed for prop firm challenge sprint (FTMO/MFF/FundedNext style):
- $50k account, 10% profit target in ~22 trading days
- 4% daily DD limit, 10% max DD limit
- Three phases: probing → acceleration → preservation

Usage:
    manager = ChallengeLifecycleManager(initial_equity=50_000)
    risk_mult = manager.get_risk_multiplier(day=5, current_equity=52_500)
    can_trade, reason = manager.can_trade(equity=51_200, trade_count=3)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseResult:
    """Result of a challenge lifecycle evaluation."""
    phase: str
    risk_multiplier: float
    max_trades_today: int
    can_trade: bool
    reason: str
    days_elapsed: int
    profit_pct: float
    drawdown_from_peak_pct: float


class DailyCircuitBreaker:
    """Enforces daily loss limit and trade count cap.

    Prevents a single bad day from breaching the 4% daily DD limit
    by stopping trading when losses approach the threshold.
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 0.035,
        max_trades_per_day: int = 4,
        initial_equity: float = 50_000,
    ):
        self.daily_loss_limit = daily_loss_limit_pct
        self.max_trades_per_day = max_trades_per_day
        self.initial_equity = initial_equity

        self.day_start_equity = initial_equity
        self.peak_equity = initial_equity
        self.trade_count = 0
        self.day_pnl = 0.0
        self.consecutive_losses = 0
        self.current_day = None

    def on_new_day(self, current_equity: float, day_number: int):
        """Reset at start of each trading day."""
        self.day_start_equity = current_equity
        self.trade_count = 0
        self.day_pnl = 0.0
        self.current_day = day_number
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def can_trade(self, current_equity: float) -> tuple:
        """Check if trading is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        if self.trade_count >= self.max_trades_per_day:
            return False, f"Max trades per day reached ({self.trade_count}/{self.max_trades_per_day})"

        day_pnl = current_equity - self.day_start_equity
        day_loss_pct = -day_pnl / self.day_start_equity if self.day_start_equity > 0 else 0

        if day_loss_pct >= self.daily_loss_limit:
            return False, f"Daily loss limit reached (loss={day_loss_pct:.1%} >= limit={self.daily_loss_limit:.0%})"

        return True, "OK"

    def on_trade_result(self, pnl: float, won: bool):
        """Record trade outcome and update consecutive loss counter."""
        self.trade_count += 1
        self.day_pnl += pnl
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def get_consecutive_losses(self) -> int:
        return self.consecutive_losses

    def get_day_pnl_pct(self) -> float:
        """Return day P&L as fraction of start equity."""
        if self.day_start_equity == 0:
            return 0.0
        return self.day_pnl / self.day_start_equity


class ChallengeLifecycleManager:
    """Manages the 30-day sprint lifecycle with phase-aware risk scaling.

    Phases:
      Probing (days 1-3):    50% risk, 3 max trades/day — calibrate
      Acceleration (days 4-20): 100% risk, 4 max trades/day — build profit
      Preservation (days 21-30): 35-70% risk based on profit level — protect gains
    """

    def __init__(
        self,
        initial_equity: float = 50_000,
        profit_target_pct: float = 0.10,
        daily_dd_limit_pct: float = 0.035,
        max_dd_limit_pct: float = 0.09,
        max_trading_days: int = 22,
    ):
        self.initial_equity = initial_equity
        self.profit_target = profit_target_pct
        self.daily_dd_limit = daily_dd_limit_pct
        self.max_dd_limit = max_dd_limit_pct
        self.max_trading_days = max_trading_days

        self.peak_equity = initial_equity
        self.current_phase = "probing"
        self.days_elapsed = 0
        self.total_profit_pct = 0.0
        self.trades_taken = 0

        self.circuit_breaker = DailyCircuitBreaker(
            daily_loss_limit_pct=daily_dd_limit_pct,
            max_trades_per_day=4,
            initial_equity=initial_equity,
        )

        # Track profit by day for consistency rule monitoring
        self.daily_profits: list[float] = []

    def _determine_phase(self, day: int) -> str:
        """Determine which phase we're in based on elapsed trading days."""
        if day <= 3:
            return "probing"
        elif day <= 20:
            return "acceleration"
        else:
            return "preservation"

    def _compute_preservation_multiplier(self, profit_pct: float) -> float:
        """In preservation phase, scale risk down as profit increases.

        The closer we are to 10%, the more conservative we get.
        """
        if profit_pct < 0.05:
            return 0.70  # Behind schedule — moderate risk
        elif profit_pct < 0.07:
            return 0.50  # On track — half risk
        elif profit_pct < 0.09:
            return 0.35  # Close — 35% risk
        else:
            return 0.15  # Very close — minimal risk, protect gains

    def update(self, current_equity: float, day: int) -> PhaseResult:
        """Update lifecycle state and return current phase parameters.

        Args:
            current_equity: Current account equity
            day: Current trading day number (1-indexed)

        Returns:
            PhaseResult with risk multiplier, trade limits, and status
        """
        self.days_elapsed = day

        # Track peak equity for drawdown calculation
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate metrics
        self.total_profit_pct = (current_equity - self.initial_equity) / self.initial_equity
        dd_from_peak = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Determine phase
        self.current_phase = self._determine_phase(day)

        # Compute risk multiplier
        if self.current_phase == "probing":
            risk_mult = 0.5
            max_trades = 3
        elif self.current_phase == "acceleration":
            risk_mult = 1.0
            max_trades = 4

            # Profit lock: if >6% profit, reduce risk
            if self.total_profit_pct > 0.06:
                risk_mult = 0.70

            # Aggressive acceleration: if >5% by day 10, increase risk
            if day <= 10 and self.total_profit_pct > 0.05:
                risk_mult = 1.2
        else:  # preservation
            risk_mult = self._compute_preservation_multiplier(self.total_profit_pct)
            max_trades = 2

        # Apply consecutive loss reduction
        consec_losses = self.circuit_breaker.get_consecutive_losses()
        if consec_losses >= 2:
            loss_reduction = {2: 0.75, 3: 0.50, 4: 0.25}.get(consec_losses, 0.25)
            risk_mult *= loss_reduction

        # Apply max drawdown kill switch
        if dd_from_peak > 0.07:
            risk_mult = min(risk_mult, 0.3)  # Max 30% of normal risk
        if dd_from_peak > 0.09:
            # Abort zone — minimal survival trading only
            risk_mult = 0.10
            max_trades = 1

        # Check if target is hit
        if self.total_profit_pct >= self.profit_target:
            return PhaseResult(
                phase=self.current_phase,
                risk_multiplier=0.0,  # No more trading
                max_trades_today=0,
                can_trade=False,
                reason=f"TARGET HIT: {self.total_profit_pct:.1%} achieved",
                days_elapsed=day,
                profit_pct=self.total_profit_pct,
                drawdown_from_peak_pct=dd_from_peak,
            )

        # Check circuit breaker
        can_trade, reason = self.circuit_breaker.can_trade(current_equity)
        if not can_trade and "Daily loss limit" in reason:
            risk_mult = 0.0  # Stop trading for the day

        # Clamp risk multiplier
        risk_mult = max(0.0, min(risk_mult, 2.0))

        return PhaseResult(
            phase=self.current_phase,
            risk_multiplier=risk_mult,
            max_trades_today=max_trades,
            can_trade=can_trade or ("Max trades" in reason),
            reason=reason if not can_trade else "OK",
            days_elapsed=day,
            profit_pct=self.total_profit_pct,
            drawdown_from_peak_pct=dd_from_peak,
        )

    def record_trade(self, pnl_dollars: float):
        """Record a completed trade and update circuit breaker."""
        won = pnl_dollars > 0
        self.circuit_breaker.on_trade_result(pnl_dollars, won)
        self.trades_taken += 1

    def on_new_day(self, current_equity: float, day: int):
        """Call at the start of each trading day."""
        # Record yesterday's daily profit
        if self.circuit_breaker.day_pnl != 0:
            self.daily_profits.append(self.circuit_breaker.day_pnl)

        self.circuit_breaker.on_new_day(current_equity, day)

    def consistency_check(self) -> tuple:
        """Check FTMO consistency rule (no single day >30% of total profit).

        Returns:
            (passes: bool, max_day_ratio: float, message: str)
        """
        if len(self.daily_profits) < 3:
            return True, 0.0, "Not enough data"

        total_profit = sum(self.daily_profits)
        if total_profit <= 0:
            return True, 0.0, "No positive total profit yet"

        max_day = max(self.daily_profits)
        max_ratio = max_day / total_profit if total_profit != 0 else 0

        if max_ratio > 0.30:
            return (
                False,
                max_ratio,
                f"CONSISTENCY WARNING: Best day {max_day:.0f} is {max_ratio:.0%} of total {total_profit:.0f} (limit: 30%)",
            )

        return True, max_ratio, f"OK (best day {max_ratio:.0%} of total)"

    def get_status_summary(self) -> dict:
        """Return a summary of the current challenge state."""
        return {
            "phase": self.current_phase,
            "days_elapsed": self.days_elapsed,
            "profit_pct": self.total_profit_pct * 100,
            "peak_equity": self.peak_equity,
            "circuit_breaker_trades": self.circuit_breaker.trade_count,
            "consecutive_losses": self.circuit_breaker.get_consecutive_losses(),
            "trades_taken": self.trades_taken,
            "daily_profits": self.daily_profits,
        }
