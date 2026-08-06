SYSTEM_CONTEXT = """
You are iterating on a trading strategy written in Python using the backtesting.py library.

Rules for every strategy file you write:
- Must define exactly ONE class subclassing backtesting.Strategy.
- Must implement init() and next() following backtesting.py's API.
- Position sizing MUST be risk-based: never risk more than 1-2% of equity per trade,
  sized from a real stop-loss distance (e.g. ATR-based), not a flat fraction of equity.
- Every entry must include an explicit stop-loss (sl=...) — no unprotected positions.
- Do not hardcode a profit target into the strategy logic itself (e.g. do not stop
  trading once some return % is hit) — the strategy should trade its logic consistently
  throughout the whole dataset, since prop firms evaluate consistency, not a single
  lucky stretch.
- Prefer strategies with a plausible economic rationale (trend-following, mean-reversion,
  volatility breakout, etc.) over complex rule stacks with many tunable parameters —
  more parameters increases overfitting risk and will be penalized in scoring.

You will be told, after each attempt, whether it was kept or discarded and why
(unprofitable on train, unprofitable on validation, large train/validation gap
meaning likely overfit, or failing prop firm rules). Use that feedback to make a
DIFFERENT structural change, not just a parameter tweak, if the same failure
reason repeats more than twice in a row.
"""

def build_iteration_prompt(previous_result, iteration_num):
    if previous_result is None:
        return SYSTEM_CONTEXT + """
Write a first strategy in strategies/candidate_current.py.
Try a well-known, simple approach as a starting point (e.g. trend-following with
a volatility filter, or mean-reversion with a regime filter). Keep it simple —
this is iteration 1.
"""

    feedback = f"""
Iteration {iteration_num - 1} result:
  Kept: {previous_result['keep']}
  Train return: {previous_result['train_return_pct']:.2f}%
  Validation return: {previous_result['val_return_pct']:.2f}%
  Reasons: {previous_result['reasons']}

Revise strategies/candidate_current.py based on this feedback.
"""
    return SYSTEM_CONTEXT + feedback
def build_structural_pivot_prompt(previous_result, iteration_num, direction):
    feedback = f"""
Iteration {iteration_num - 1} result:
  Train return: {previous_result['train_return_pct']:.2f}%
  Validation return: {previous_result['val_return_pct']:.2f}%
  Reasons for discard: {previous_result['reasons']}

The previous approach (breakout-based) overtraded on 15m gold data with a low win rate.
For this iteration, implement a STRUCTURALLY DIFFERENT approach: {direction}

Write this as strategies/candidate_current.py, replacing the previous content entirely.
Follow all the same rules as before: exactly one Strategy subclass, explicit stop-loss
on every entry, risk-based position sizing (1-2% risk per trade via ATR or similar),
no hardcoded profit-target-based trading halt.
"""
    return SYSTEM_CONTEXT + feedback
