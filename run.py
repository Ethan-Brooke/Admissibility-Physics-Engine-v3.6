#!/usr/bin/env python3
"""
FCF Master Verification -- single entry point.

Usage:
    python3 run.py                  # Full display
    python3 run.py --json           # JSON export
    python3 run.py --audit-gaps     # Gap analysis
    python3 run.py --deps T24       # Dependency tree for T24
    python3 run.py --reverse-deps A1  # What depends on A1

Delegates to Admissibility_Physics_Engine_V4_2.py.
"""
import subprocess
import sys
import os

engine = os.path.join(os.path.dirname(__file__), "Admissibility_Physics_Engine_V4_2.py")
sys.exit(subprocess.call([sys.executable, engine] + sys.argv[1:]))
