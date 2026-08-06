MAX_ITERATIONS = 5   # hard cap — do not remove this
STRATEGY_FILE = "strategies/candidate_current.py"
STRATEGY_MODULE = "strategies.candidate_current"


import json
import os

def call_hermes(prompt, iteration_num):
    usage_path = f"runs/iteration_{iteration_num:04d}/hermes_usage.json"
    os.makedirs(os.path.dirname(usage_path), exist_ok=True)

    result = subprocess.run(
        [
            "hermes", "-z", prompt,
            "--yolo",                      # required for unattended runs — see note below
            "--accept-hooks",
            "--usage-file", usage_path,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Hermes call failed: {result.stderr}")

    # Track spend so the loop can stop if cost runs away
    cost = 0.0
    if os.path.exists(usage_path):
        with open(usage_path) as f:
            usage = json.load(f)
            cost = usage.get("estimated_cost", 0.0)

    return result.stdout, cost


def main():
    previous_result = None
    kept_candidates = []
    total_cost = 0.0
    MAX_BUDGET_USD = 5.0   # set this to whatever you're actually comfortable spending — adjust freely

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*50}\nITERATION {i}/{MAX_ITERATIONS}  (spend so far: ${total_cost:.4f})\n{'='*50}")

        if total_cost >= MAX_BUDGET_USD:
            print(f"Budget cap (${MAX_BUDGET_USD}) reached — stopping.")
            break

        prompt = build_iteration_prompt(previous_result, i)
        _, cost = call_hermes(prompt, i)
        total_cost += cost

        result = run_iteration(i, STRATEGY_MODULE, symbol_prefix="XAUUSD_15m")
        previous_result = result

        if result["keep"]:
            kept_candidates.append(result)
            print(f"CANDIDATE FOUND at iteration {i}")

        if len(kept_candidates) >= 3:
            print("3 candidates found — stopping loop for manual review.")
            break

    print(f"\nLoop finished. {len(kept_candidates)} candidate(s) kept. Total spend: ${total_cost:.4f}")