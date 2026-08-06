
"""
Combines train-set stats, validation-set stats, and prop-firm verdicts
into one decision: keep this candidate, or discard it.
"""

def score_candidate(train_stats, train_eval_results, val_stats, val_eval_results):
    """
    train_stats / val_stats: backtesting.py stats objects for train/validation runs
    train_eval_results / val_eval_results: lists of evaluate_against_rules() outputs,
                                             one per firm, for train/validation respectively

    Returns dict with: keep (bool), reason (str), robustness_gap (float), score (float)
    """
    train_return = train_stats["Return [%]"]
    val_return = val_stats["Return [%]"]

    # Robustness check: validation shouldn't collapse relative to train.
    # A strategy that does great on train but terribly on validation is overfit,
    # regardless of how good either individual number looks alone.
    if train_return <= 0 and val_return <= 0:
        robustness_gap = 0.0  # both unprofitable — no meaningful "overfit" claim to make either way
    elif train_return <= 0:
        robustness_gap = float("inf")  # train unprofitable but val profitable — can't trust the ratio
    else:
        robustness_gap = (train_return - val_return) / abs(train_return)

    reasons = []

    if train_return <= 0:
        reasons.append("unprofitable on train data")
    if val_return <= 0:
        reasons.append("unprofitable on validation data")
    if robustness_gap > 0.5:
        reasons.append(f"large train/validation gap ({robustness_gap:.0%}) — likely overfit")

    val_passes_any_firm = any(r["passed"] for r in val_eval_results)
    if not val_passes_any_firm:
        reasons.append("fails all prop firm rule sets on validation data")

    keep = len(reasons) == 0

    # Simple composite score for ranking candidates that all "keep" — not used for pass/fail,
    # only for sorting multiple survivors later.
    score = val_return - (robustness_gap * 100 if robustness_gap != float("inf") else 1000)

    return {
        "keep": keep,
        "reasons": reasons,
        "robustness_gap": robustness_gap,
        "train_return_pct": train_return,
        "val_return_pct": val_return,
        "score": score,
    }
