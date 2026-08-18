"""
ai_loop/ai_loop.py — Nightly AI parameter optimization loop.
Runs via systemd timer (18:00 ET daily). Does NOT require human intervention.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.config import MIMO_API_KEY, STRATEGY_PARAMS

logger = logging.getLogger(__name__)


def query_mimo(metrics: dict) -> dict:
    """
    Query MiMo v2.5 Pro for parameter change proposals.

    CRITICAL MiMo v2.5 gotchas:
    - max_tokens=8192 is mandatory (model burns tokens in internal reasoning)
    - temperature=0.0 for deterministic JSON output
    - System message for JSON-only is mandatory (otherwise <think> tags)
    - Always strip ```json fences before parsing
    """
    if not MIMO_API_KEY or MIMO_API_KEY == "your_mimo_api_key_here":
        logger.warning("MIMO_API_KEY not configured")
        return {}

    try:
        from openai import OpenAI
        from ai_loop.prompts import get_system_prompt, build_optimization_prompt

        client = OpenAI(
            base_url="https://token-plan-sgp.xiaomimimo.com/v1",
            api_key=MIMO_API_KEY,
        )

        response = client.chat.completions.create(
            model="mimo-v2.5-pro",
            max_tokens=8192,  # CRITICAL: lower values produce empty responses
            temperature=0.0,  # CRITICAL: deterministic JSON
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": build_optimization_prompt(metrics)},
            ],
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences (model occasionally adds them)
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"MiMo returned invalid JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"MiMo API call failed: {e}")
        return {}


def validate_proposal(proposal: dict) -> dict:
    """
    Validate a parameter proposal via walk-forward validation.

    Args:
        proposal: Dict with strategy, symbol, parameter, proposed_value.

    Returns:
        Validation result dict.
    """
    from backtest.engine import BacktestEngine, _import_strategy, _filter_params
    from backtest.optimization.purged_walk_forward import PurgedWalkForward
    from backtest.data.fetcher import fetch
    from backtest.config import INSTRUMENTS

    strategy = proposal.get("strategy")
    symbol = proposal.get("symbol")
    parameter = proposal.get("parameter")
    proposed_value = proposal.get("proposed_value")

    if not all([strategy, symbol, parameter, proposed_value is not None]):
        return {"passes": False, "reason": "incomplete_proposal"}

    # Get current params
    params = STRATEGY_PARAMS.get(strategy, {}).get(symbol, {}).copy()
    params[parameter] = proposed_value

    # Fetch data
    instrument = INSTRUMENTS.get(symbol, {})
    timeframe = instrument.get("timeframe", "15min")
    df = fetch(symbol, timeframe)

    if df.empty:
        return {"passes": False, "reason": "no_data"}

    # Run walk-forward validation
    strategy_mod = _import_strategy(strategy)
    fn = strategy_mod.generate_signals
    filtered_params = _filter_params(fn, params)

    wfv = PurgedWalkForward(n_splits=6, embargo=20)
    result = wfv.validate(df, fn, filtered_params)

    return result


def patch_config(proposal: dict) -> bool:
    """
    Apply a validated proposal to config.py.
    """
    from ai_loop.config_patcher import ConfigPatcher

    patcher = ConfigPatcher()
    return patcher.patch(
        strategy=proposal["strategy"],
        symbol=proposal["symbol"],
        parameter=proposal["parameter"],
        value=proposal["proposed_value"],
    )


def log_change(metrics: dict, proposal: dict, validation: dict):
    """Log a config change to the change log."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"changes_{datetime.now().strftime('%Y%m')}.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_snapshot": metrics,
        "proposal": proposal,
        "validation": validation,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_rejection(metrics: dict, proposal: dict, reason: str):
    """Log a rejected proposal."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"rejections_{datetime.now().strftime('%Y%m')}.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "proposal": proposal,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_nightly_loop():
    """
    Main nightly optimization loop.

    1. Run backtest with CURRENT config
    2. Query MiMo for parameter changes
    3. Validate proposal survives walk-forward
    4. Apply to config.py if validated
    5. Log change
    """
    logger.info("=== AI Loop Starting ===")

    # Step 1: Run backtest
    logger.info("Running backtest with current config...")
    try:
        result = subprocess.run(
            [sys.executable, "backtest/main.py", "--json"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"Backtest failed: {result.stderr}")
            return

        metrics = json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        return

    # Step 2: Query MiMo
    logger.info("Querying MiMo for optimization proposal...")
    proposal = query_mimo(metrics)

    if not proposal:
        logger.info("No proposal received from MiMo")
        return

    logger.info(f"Proposal: {proposal}")

    # Step 3: Validate
    logger.info("Validating proposal via walk-forward...")
    validation = validate_proposal(proposal)

    if not validation.get("passes"):
        log_rejection(metrics, proposal, reason="failed_validation")
        logger.info(f"Proposal rejected: {validation}")
        return

    # Step 4: Apply
    logger.info("Proposal validated — applying to config.py")
    success = patch_config(proposal)

    if success:
        log_change(metrics, proposal, validation)
        logger.info("Config updated successfully")
    else:
        logger.error("Failed to apply config patch")

    logger.info("=== AI Loop Complete ===")


def main():
    """Entry point for AI loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_nightly_loop()


if __name__ == "__main__":
    main()
