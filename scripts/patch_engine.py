import re

# Read engine.py
with open('/home/admin1/project9/backtest/engine.py', 'r') as f:
    content = f.read()

# Add new strategy imports
old = '        elif strategy == "vwap_mean_reversion":\n            from strategies.vwap_mean_reversion import generate_signals\n        else:'
new = '''        elif strategy == "vwap_mean_reversion":
            from strategies.vwap_mean_reversion import generate_signals
        elif strategy == "orb":
            from strategies.orb_strategy import generate_signals
        elif strategy == "xauusd_session_mr":
            from strategies.xauusd_session_mr import generate_signals
        else:'''

content = content.replace(old, new)

with open('/home/admin1/project9/backtest/engine.py', 'w') as f:
    f.write(content)

print("engine.py updated")
