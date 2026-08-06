"""Trade memory log — append-only markdown decision log with outcome tracking.

Ported pattern from TradingAgents (TauricResearch) v0.3.0 TradingMemoryLog + Reflector.

Each completed trade creates an entry:

    [2024-05-10 | GLD | Buy | +3.2% | +2.1% | 5d]

    DECISION:
    Entry at $185.30, exit at $191.20 on KF velocity cross. Regime: trending.
    Position: 2.0% risk, ATR-sized.

    REFLECTION:
    Directional call correct (+3.2% raw, +2.1% alpha vs SPY). The trend
    regime held throughout. Lesson: widen trail_atr_mult to 3.0 in trending
    regimes to capture full moves.
    <!-- ENTRY_END -->

On each run, the last N same-instrument and M cross-instrument resolved entries
are available as context for parameter tuning or reflection filter injection.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import REFLECTION_CONFIG, BACKTEST_CONFIG

# ── paths ──────────────────────────────────────────────────────────────────

_HOME = Path.home() / ".project9"
_MEMORY_DIR = _HOME / "trade_memory"
_MEMORY_LOG = _MEMORY_DIR / "trade_memory.md"
_LESSONS_DIR = _MEMORY_DIR / "lessons"


def _ensure_dirs() -> None:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _LESSONS_DIR.mkdir(parents=True, exist_ok=True)


# ── entry format ───────────────────────────────────────────────────────────

# HTML comment separator: cannot appear in LLM prose output, safe delimiter.
_ENTRY_SEP = "\n\n<!-- ENTRY_END -->\n\n"

# Tag line: [date | ticker | direction | raw_return | alpha_return | holding_days]
_TAG_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2})\s*\|\s*(\S+)\s*\|\s*(\S+)\s*(?:\|\s*([^\]]*?)\s*)?\]"
)


def _format_tag(
    trade_date: str,
    ticker: str,
    direction: str,
    raw_return: float | None = None,
    alpha_return: float | None = None,
    holding_days: int | None = None,
    is_pending: bool = True,
) -> str:
    """Build a structured tag line."""
    if is_pending:
        return f"[{trade_date} | {ticker} | {direction} | pending]"
    raw = f"{raw_return:+.1%}" if raw_return is not None else "?"
    alpha = f"{alpha_return:+.1%}" if alpha_return is not None else "?"
    days = f"{holding_days}d" if holding_days is not None else "?"
    return f"[{trade_date} | {ticker} | {direction} | {raw} | {alpha} | {days}]"


# ── log writer ─────────────────────────────────────────────────────────────


def log_trade(
    ticker: str,
    trade_date: str,
    direction: str,
    entry_price: float,
    regime: str = "",
    strategy: str = "",
    notes: str = "",
) -> None:
    """Append a pending trade entry to the memory log.

    The entry is stored as 'pending' until ``resolve_trade()`` is called with
    the actual outcome.
    """
    if not REFLECTION_CONFIG.get("enabled", False):
        return

    _ensure_dirs()
    tag = _format_tag(trade_date, ticker, direction, is_pending=True)

    decision = (
        f"Entry at ${entry_price:.2f}. Regime: {regime}. "
        f"Strategy: {strategy}."
    )
    if notes:
        decision += f" {notes}"

    entry = f"{tag}\n\nDECISION:\n{decision}{_ENTRY_SEP}"

    _append_entry(entry)


def resolve_trade(
    ticker: str,
    trade_date: str,
    exit_price: float,
    holding_days: int,
    raw_return: float,
    alpha_return: float | None = None,
    benchmark: str = "SPY",
    reflection: str = "",
) -> bool:
    """Resolve a pending trade entry with actual outcome.

    Finds the first pending entry matching (trade_date, ticker) and replaces
    its tag + appends a REFLECTION section.

    Returns True if a pending entry was updated.
    """
    if not REFLECTION_CONFIG.get("enabled", False):
        return False

    if not _MEMORY_LOG.exists():
        return False

    text = _MEMORY_LOG.read_text(encoding="utf-8")
    blocks = text.split(_ENTRY_SEP)

    pending_prefix = f"[{trade_date} | {ticker} |"
    updated = False
    new_blocks = []

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            new_blocks.append(block)
            continue

        lines = stripped.splitlines()
        tag_line = lines[0].strip()

        if (
            not updated
            and tag_line.startswith(pending_prefix)
            and tag_line.endswith("| pending]")
        ):
            # Parse direction from existing tag
            fields = [f.strip() for f in tag_line[1:-1].split("|")]
            direction = fields[2] if len(fields) > 2 else "?"

            new_tag = _format_tag(
                trade_date, ticker, direction,
                raw_return=raw_return, alpha_return=alpha_return,
                holding_days=holding_days, is_pending=False,
            )

            # Add a simple auto-generated reflection if none provided
            if not reflection:
                reflection = _auto_reflection(
                    direction, raw_return, alpha_return, benchmark,
                )

            rest = "\n".join(lines[1:])
            new_blocks.append(f"{new_tag}\n\n{rest}\n\nREFLECTION:\n{reflection}")
            updated = True
        else:
            new_blocks.append(block)

    if not updated:
        return False

    new_text = _ENTRY_SEP.join(new_blocks)
    _atomic_write(new_text)
    return True


# ── reader ─────────────────────────────────────────────────────────────────


def load_entries(include_pending: bool = False) -> list[dict]:
    """Parse all entries from the log.

    Returns list of dicts with keys:
        date, ticker, direction, raw, alpha, holding, decision, reflection, pending
    """
    if not _MEMORY_LOG.exists():
        return []

    text = _MEMORY_LOG.read_text(encoding="utf-8")
    raw_entries = [e.strip() for e in text.split(_ENTRY_SEP) if e.strip()]
    entries = []
    for raw in raw_entries:
        parsed = _parse_entry(raw)
        if parsed:
            entries.append(parsed)
    if not include_pending:
        entries = [e for e in entries if not e.get("pending")]
    return entries


def get_pending_entries() -> list[dict]:
    """Return entries still awaiting outcome resolution."""
    return [e for e in load_entries(include_pending=True) if e.get("pending")]


def get_context(
    ticker: str,
    n_same: int = 5,
    n_cross: int = 3,
) -> str:
    """Return formatted context string of past resolved entries.

    Injects last N same-ticker and M cross-ticker resolved entries for use
    in parameter tuning or reflection filter injection.
    """
    entries = [e for e in load_entries() if not e.get("pending")]
    if not entries:
        return ""

    same, cross = [], []
    for e in reversed(entries):
        if len(same) >= n_same and len(cross) >= n_cross:
            break
        if e["ticker"] == ticker and len(same) < n_same:
            same.append(e)
        elif e["ticker"] != ticker and len(cross) < n_cross:
            cross.append(e)

    if not same and not cross:
        return ""

    parts = []
    if same:
        parts.append(f"Past trades in {ticker} (most recent first):")
        for e in same:
            parts.append(_format_entry_short(e))
    if cross:
        parts.append("Cross-instrument lessons:")
        for e in cross:
            parts.append(_format_entry_short(e))
    return "\n\n".join(parts)


def get_lessons(ticker: str) -> list[dict]:
    """Load previously saved lessons for a ticker (from the lessons store)."""
    path = _LESSONS_DIR / f"lessons_{ticker.replace('/', '_')}.json"
    if not path.exists():
        return _compute_lessons(ticker)
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return _compute_lessons(ticker)


def compute_and_save_lessons(ticker: str) -> list[dict]:
    """Compute lessons from trade history and save to JSON."""
    lessons = _compute_lessons(ticker)
    path = _LESSONS_DIR / f"lessons_{ticker.replace('/', '_')}.json"
    import json
    path.write_text(json.dumps(lessons, indent=2, default=str))
    return lessons


# ── internal ───────────────────────────────────────────────────────────────


def _append_entry(entry: str) -> None:
    """Atomically append an entry to the log."""
    _ensure_dirs()
    if _MEMORY_LOG.exists():
        existing = _MEMORY_LOG.read_text(encoding="utf-8")
        # Idempotency guard: skip if exact tag already exists
        tag_line = entry.splitlines()[0].strip()
        if tag_line in existing:
            return
        _atomic_write(existing.rstrip() + "\n\n" + entry.lstrip())
    else:
        _atomic_write(entry.lstrip())


def _atomic_write(content: str) -> None:
    """Write with temp-file + rename for crash safety."""
    tmp = _MEMORY_LOG.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(_MEMORY_LOG)


def _parse_entry(raw: str) -> dict | None:
    """Parse a single entry block into a dict."""
    lines = raw.strip().splitlines()
    if not lines:
        return None
    tag_line = lines[0].strip()
    m = _TAG_RE.match(tag_line)
    if not m:
        return None

    entry = {
        "date": m.group(1),
        "ticker": m.group(2),
        "direction": m.group(3),
        "pending": "pending" in tag_line,
        "raw": None,
        "alpha": None,
        "holding": None,
        "decision": "",
        "reflection": "",
    }

    if not entry["pending"]:
        # Parse outcome fields from tag
        # tag: [date | ticker | direction | raw | alpha | holding]
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) >= 5:
            entry["raw"] = fields[3]
            entry["alpha"] = fields[4] if len(fields) > 4 else None
            entry["holding"] = fields[5] if len(fields) > 5 else None

    body = "\n".join(lines[1:]).strip()
    decision_match = re.search(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", body, re.DOTALL)
    reflection_match = re.search(r"REFLECTION:\n(.*?)$", body, re.DOTALL)
    if decision_match:
        entry["decision"] = decision_match.group(1).strip()
    if reflection_match:
        entry["reflection"] = reflection_match.group(1).strip()

    return entry


def _format_entry_short(e: dict) -> str:
    """Format an entry as a single-line summary."""
    raw = e["raw"] or "?"
    alpha = e["alpha"] or "?"
    tag = f"[{e['date']} | {e['ticker']} | {e['direction']} | {raw} | {alpha}]"
    if e.get("reflection"):
        return f"{tag} {e['reflection'][:200]}"
    return f"{tag} {e['decision'][:100]}"


def _auto_reflection(
    direction: str,
    raw_return: float,
    alpha_return: float | None,
    benchmark: str = "SPY",
) -> str:
    """Generate a quantitative auto-reflection when no LLM is available.

    This is the non-LLM fallback.  When LLM integration is enabled, a prompt
    would replace this with a narrative reflection.
    """
    is_win = raw_return > 0
    parts = [
        f"{'Correct' if is_win else 'Incorrect'} directional call "
        f"({raw_return:+.1%} raw)",
    ]
    if alpha_return is not None:
        parts.append(f"alpha vs {benchmark}: {alpha_return:+.1%}")
    parts.append(f"Direction was {direction.lower()}.")
    return ". ".join(parts) + "."


def _compute_lessons(ticker: str) -> list[dict]:
    """Compute quantitative lessons from trade history."""
    entries = load_entries()
    ticker_entries = [e for e in entries if e["ticker"] == ticker]
    if not ticker_entries:
        return []

    wins = [e for e in ticker_entries if e.get("raw") and _parse_pct(e["raw"]) > 0]
    losses = [e for e in ticker_entries if e.get("raw") and _parse_pct(e["raw"]) <= 0]

    win_rate = len(wins) / max(len(ticker_entries), 1)

    lessons = []
    if win_rate < 0.3 and len(ticker_entries) >= 5:
        lessons.append({
            "type": "low_win_rate",
            "message": f"Win rate {win_rate:.0%} over {len(ticker_entries)} trades. Tighten entry filters.",
            "severity": "high",
        })

    if len(losses) >= 3:
        recent = [e for e in losses[-3:] if e.get("raw")]
        if len(recent) >= 3 and all(_parse_pct(e["raw"]) < -2.0 for e in recent):
            lessons.append({
                "type": "consecutive_losses",
                "message": "3 consecutive losses > 2%. Reduce position sizing.",
                "severity": "high",
            })

    return lessons


def _parse_pct(s: str) -> float:
    """Parse '+3.2%' -> 3.2."""
    try:
        return float(s.strip("%+"))
    except (ValueError, AttributeError):
        return 0.0
