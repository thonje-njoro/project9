"""
paper_trading/risk_guard.py — Real-time drawdown check against prop firm rules.
"""

import logging
from datetime import datetime

from backtest.config import PROP_FIRM_RULES, RISK_CONFIG

logger = logging.getLogger(__name__)


class RiskGuard:
    """
    Real-time risk management for paper trading.
    Checks daily loss limits and total drawdown against prop firm rules.
    """

    def __init__(self, firm_rules: str = "ftmo_2step"):
        """
        Args:
            firm_rules: Key into PROP_FIRM_RULES dict.
        """
        self.rules = PROP_FIRM_RULES.get(firm_rules, PROP_FIRM_RULES["ftmo_2step"])
        self.firm_name = firm_rules

        # State
        self.start_of_day_equity = 0.0
        self.peak_equity = 0.0
        self.is_halted = False
        self.halt_reason = None
        self.blocked_today = False

    def update_equity(self, current_equity: float):
        """
        Update equity tracking. Call on each bar.

        Args:
            current_equity: Current account equity.
        """
        if self.start_of_day_equity == 0:
            self.start_of_day_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def new_day(self, equity: float):
        """Reset daily tracking at market open."""
        self.start_of_day_equity = equity
        self.blocked_today = False
        logger.info(f"New trading day — equity: ${equity:,.2f}")

    def check(self, current_equity: float) -> dict:
        """
        Check all risk limits.

        Args:
            current_equity: Current account equity.

        Returns:
            dict with:
                allow_trading: bool
                daily_loss_pct: float
                total_dd_pct: float
                reason: str or None
        """
        if self.is_halted:
            return {
                "allow_trading": False,
                "reason": f"HALTED: {self.halt_reason}",
            }

        # Daily loss check
        if self.start_of_day_equity > 0:
            daily_loss = (self.start_of_day_equity - current_equity) / self.start_of_day_equity
        else:
            daily_loss = 0

        if daily_loss > self.rules["daily_loss_pct"]:
            self.blocked_today = True
            logger.warning(
                f"Daily loss {daily_loss:.3%} exceeds limit "
                f"{self.rules['daily_loss_pct']:.3%} — blocking new entries"
            )
            return {
                "allow_trading": False,
                "daily_loss_pct": daily_loss,
                "reason": f"daily_loss_exceeded_{daily_loss:.3%}",
            }

        # Total drawdown check
        if self.peak_equity > 0:
            total_dd = (self.peak_equity - current_equity) / self.peak_equity
        else:
            total_dd = 0

        if total_dd > self.rules["max_drawdown_pct"]:
            self.is_halted = True
            self.halt_reason = f"total_dd_{total_dd:.3%}_exceeds_{self.rules['max_drawdown_pct']:.3%}"
            logger.error(f"RISK HALT: {self.halt_reason}")
            return {
                "allow_trading": False,
                "total_dd_pct": total_dd,
                "reason": self.halt_reason,
            }

        return {
            "allow_trading": True,
            "daily_loss_pct": daily_loss,
            "total_dd_pct": total_dd,
            "reason": None,
        }

    def get_status(self) -> dict:
        """Get current risk guard status."""
        return {
            "firm": self.firm_name,
            "start_of_day_equity": self.start_of_day_equity,
            "peak_equity": self.peak_equity,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "blocked_today": self.blocked_today,
            "rules": self.rules,
        }
