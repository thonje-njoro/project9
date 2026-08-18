"""
ai_loop/prompts.py — MiMo system + user prompt templates for parameter optimization.
"""


def get_system_prompt() -> str:
    """System prompt for MiMo v2.5 Pro."""
    return (
        "You are a quantitative trading parameter optimizer. "
        "Respond ONLY with a valid JSON object. No thinking tags. "
        "No markdown. No preamble. Pure JSON only."
    )


def build_optimization_prompt(metrics: dict) -> str:
    """
    Build the user prompt for parameter optimization.

    Args:
        metrics: Current backtest metrics dict.

    Returns:
        Formatted prompt string.
    """
    import json
    return f"""Current backtest metrics (2022-01-01 to 2024-12-31):

{json.dumps(metrics, indent=2)}

Analyze these results and propose ONE parameter change that is most likely to
improve out-of-sample Sharpe Ratio.

Rules:
- Only change ONE parameter at a time (to maintain experimental control)
- Changes must be within ±30% of current value (avoid overfitting via large jumps)
- Do not change parameters for strategies with Sharpe > 1.5 (they are working)
- Prioritize fixing the lowest-Sharpe strategy first
- If MORB strategies show 0 trades, check timezone configuration

Return JSON with exactly this structure:
{{
  "strategy": "<strategy_name>",
  "symbol": "<instrument_name>",
  "parameter": "<param_name>",
  "current_value": <current_value>,
  "proposed_value": <new_value>,
  "reasoning": "<one sentence explanation>"
}}"""


def build_validation_prompt(metrics: dict, proposal: dict,
                            validation_result: dict) -> str:
    """
    Build prompt for analyzing validation results of a proposal.

    Args:
        metrics: Current metrics.
        proposal: The proposal that was tested.
        validation_result: WFV + MC results.

    Returns:
        Formatted prompt string.
    """
    import json
    return f"""A parameter change was proposed and validated:

Proposal:
{json.dumps(proposal, indent=2)}

Validation Result:
{json.dumps(validation_result, indent=2)}

Current Metrics:
{json.dumps(metrics, indent=2)}

Was this change beneficial? If not, suggest what to try next.
Return JSON:
{{
  "action": "apply" | "reject" | "try_next",
  "reasoning": "<explanation>",
  "next_proposal": {{ ... }} // only if action is "try_next"
}}"""
