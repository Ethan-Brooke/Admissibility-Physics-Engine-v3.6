#!/usr/bin/env python3
"""
Convenience wrapper — run the Admissibility Physics Engine v3.6.

Usage:
    python3 run.py              # Full verification display
    python3 run.py --json       # JSON output
    python3 run.py --audit-gaps # Gap analysis
    python3 run.py --export-dashboard  # Export dashboard_data.json
"""
import runpy, sys

sys.argv = [
    "Admissibility_Physics_Engine_V3_6.py"
] + sys.argv[1:]

runpy.run_module(
    "Admissibility_Physics_Engine_V3_6",
    run_name="__main__",
    alter_sys=True,
)
