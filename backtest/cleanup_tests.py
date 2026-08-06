#!/usr/bin/env python3
"""
Clean up temporary test files.
"""
import os
base = os.path.dirname(os.path.abspath(__file__))
for f in ["research_area_test.py", "research_improvements.py", "research_tlt_short.py"]:
    p = os.path.join(base, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed {f}")
print("Done.")
