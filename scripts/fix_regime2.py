path = "/home/admin1/project9/backtest/strategies/xauusd_session_mr.py"
with open(path) as f:
    lines = f.readlines()

# Find the line "    entry_idx = 0" and add regime init after it
new_lines = []
for line in lines:
    new_lines.append(line)
    if line.strip() == "entry_idx = 0":
        # Add regime initialization after this line
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + 'regime = "uncertain"  # Will be set by regime detection\n')

with open(path, "w") as f:
    f.writelines(new_lines)

print("Fixed: added regime initialization")
