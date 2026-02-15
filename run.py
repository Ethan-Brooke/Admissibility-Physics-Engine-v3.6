#!/usr/bin/env python3
"""
Entry point for the Admissibility Physics Engine.

Usage:
    python3 run.py                     # Full display
    python3 run.py --json              # JSON export
    python3 run.py --audit-gaps        # Gap analysis for every [P_structural]
    python3 run.py --deps T_CKM       # Dependency tree for a theorem
    python3 run.py --reverse-deps A1   # What depends on a theorem
    python3 run.py --export-dashboard  # Write dashboard_data.json

Source files:
    FCF_Theorem_Bank_v4_3.py           # All 89 entries (Tiers 0-5 + 3F)
    Admissibility_Physics_Engine_V4_3.py  # Master verification engine
"""

import sys
import json


def main():
    # --export-dashboard: run engine, write JSON to dashboard_data.json
    if '--export-dashboard' in sys.argv:
        from Admissibility_Physics_Engine_V4_3 import run_master, export_json
        master = run_master()
        data = json.loads(export_json(master))
        # Inject theorem_checker for dashboard badge
        data['theorem_checker'] = {
            'available': True,
            'passed': data['all_pass'],
            'n_pass': data['passed'],
            'total': data['total_theorems'],
        }
        with open('dashboard_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Wrote dashboard_data.json ({data['total_theorems']} theorems, "
              f"{'ALL PASS' if data['all_pass'] else 'FAILURES'})")
        sys.exit(0 if data['all_pass'] else 1)

    # All other flags are handled by the engine's own __main__ block.
    # Re-dispatch to the engine module.
    import Admissibility_Physics_Engine_V4_3  # noqa: F401
    # The engine's __main__ block runs on import when invoked via runpy,
    # but since we're importing it as a module, we need to call it explicitly.
    from Admissibility_Physics_Engine_V4_3 import (
        run_master, display, display_audit_gaps, export_json,
        trace_deps, reverse_deps,
    )

    master = run_master()

    if '--json' in sys.argv:
        print(export_json(master))
    elif '--audit-gaps' in sys.argv:
        display(master)
        display_audit_gaps(master)
    elif '--deps' in sys.argv:
        idx = sys.argv.index('--deps')
        if idx + 1 < len(sys.argv):
            tid = sys.argv[idx + 1]
            if tid in master['all_results']:
                print(f"Dependency tree for {tid}:\n")
                for line in trace_deps(master['all_results'], tid):
                    print(line)
            else:
                print(f"Unknown theorem: {tid}")
                print(f"Available: {', '.join(sorted(master['all_results'].keys()))}")
        else:
            print("Usage: --deps <theorem_id>")
    elif '--reverse-deps' in sys.argv:
        idx = sys.argv.index('--reverse-deps')
        if idx + 1 < len(sys.argv):
            tid = sys.argv[idx + 1]
            deps = reverse_deps(master['all_results'], tid)
            print(f"Theorems that depend on {tid} ({len(deps)}):\n")
            for d in deps:
                r = master['all_results'][d]
                print(f"  {d:20s} [{r['epistemic']}]")
        else:
            print("Usage: --reverse-deps <theorem_id>")
    else:
        display(master)

    sys.exit(0 if master['all_pass'] else 1)


if __name__ == '__main__':
    main()
