#!/usr/bin/env python3
"""
sync_dashboard.py — Bridge between theorems.py and the dashboard HTML.

Runs the theorem bank, extracts results, and patches the dashboard's
hardcoded `var D={...};` data blob with current values.

Usage:
  python3 sync_dashboard.py                    # default files
  python3 sync_dashboard.py --dashboard path   # custom dashboard path
  python3 sync_dashboard.py --dry-run          # show changes without writing

What it updates:
  - total_theorems, passed counts
  - epistemic_counts (P, P_structural, etc.)
  - tier_stats
  - per-theorem epistemic status, key_result, dependencies, gap_type
  - p_structural_reasons (adds/removes as needed)
  - version string + date
  - theorem_checker stats

What it does NOT touch:
  - predictions[] array (manually curated)
  - audit_checks[] array (manually curated)
  - math_imports{} (manually curated)
  - Crystal/DAG metrics (those come from enforcement_crystal_v2.py)
  - Any HTML/CSS/JS outside the data blob
"""

import sys
import os
import re
import json
import datetime
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Run theorem bank
# ---------------------------------------------------------------------------

def run_theorems():
    """Import and execute theorems.py, return results dict."""
    # Add current dir to path so theorems.py can be imported
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from theorems import run_all, THEOREM_REGISTRY
    results = run_all()
    return results, len(THEOREM_REGISTRY)


# ---------------------------------------------------------------------------
# 2. Build updated data blob fields
# ---------------------------------------------------------------------------

def build_updates(results):
    """From theorem results, compute all dashboard-relevant fields."""
    total = len(results)
    passed = sum(1 for r in results.values() if r['passed'])
    all_pass = (passed == total)

    # Epistemic counts
    epi_counter = Counter(r['epistemic'] for r in results.values())
    epistemic_counts = dict(epi_counter)

    # Tier stats
    tier_names = {
        0: 'Axiom Foundations + Quantum',
        1: 'Gauge Group Selection',
        2: 'Particle Content',
        3: 'Continuous Constants / RG',
        4: 'Gravity + Dark Sector',
        5: 'Delta_geo Closure',
    }
    tier_stats = {}
    for tier in range(6):
        tier_results = [r for r in results.values() if r['tier'] == tier]
        if tier_results:
            tier_stats[str(tier)] = {
                'name': tier_names.get(tier, f'Tier {tier}'),
                'passed': sum(1 for r in tier_results if r['passed']),
                'total': len(tier_results),
            }

    # Theorem checker
    failures = [tid for tid, r in results.items() if not r['passed']]
    theorem_checker = {
        'available': True,
        'passed': all_pass,
        'total': total,
        'n_pass': passed,
        'n_fail': len(failures),
        'failures': failures,
    }

    # P_structural reasons (preserve existing, add new, remove upgraded)
    ps_reasons = {}
    for tid, r in results.items():
        if r['epistemic'] == 'P_structural':
            # Try to infer reason from artifacts or existing patterns
            artifacts = r.get('artifacts', {})
            reason = artifacts.get('ps_reason', 'open_physics')
            ps_reasons[tid] = reason

    # Per-theorem data for the theorems{} blob
    theorems = {}
    for tid, r in results.items():
        t = {
            'name': r['name'],
            'tier': r['tier'],
            'passed': r['passed'],
            'epistemic': r['epistemic'],
            'key_result': r['key_result'],
            'gap_type': r.get('gap_type', 'closed' if r['epistemic'] == 'P' else 'open_physics'),
            'dependencies': r.get('dependencies', []),
        }
        if 'imported_theorems' in r:
            t['imported_theorems'] = r['imported_theorems']
        if r['epistemic'] == 'P_structural':
            t['ps_reason'] = ps_reasons.get(tid, 'open_physics')
        theorems[tid] = t

    return {
        'total_theorems': total,
        'passed': passed,
        'all_pass': all_pass,
        'epistemic_counts': epistemic_counts,
        'tier_stats': tier_stats,
        'theorem_checker': theorem_checker,
        'p_structural_reasons': ps_reasons,
        'theorems': theorems,
    }


# ---------------------------------------------------------------------------
# 3. Patch dashboard HTML
# ---------------------------------------------------------------------------

def extract_blob(html):
    """Extract the var D={...}; JSON blob from dashboard HTML."""
    match = re.search(r'var D=(\{.*?\});\s*\n', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'var D={...};' in dashboard HTML")
    return json.loads(match.group(1)), match.start(1), match.end(1)


def patch_blob(old_blob, updates):
    """Merge updates into the existing blob, preserving manually curated fields."""
    new_blob = dict(old_blob)  # shallow copy

    # Update computed fields
    new_blob['total_theorems'] = updates['total_theorems']
    new_blob['passed'] = updates['passed']
    new_blob['all_pass'] = updates['all_pass']
    new_blob['epistemic_counts'] = updates['epistemic_counts']
    new_blob['tier_stats'] = updates['tier_stats']
    new_blob['theorem_checker'] = updates['theorem_checker']
    new_blob['p_structural_reasons'] = updates['p_structural_reasons']

    # Update per-theorem data (merge, don't replace — preserve extra fields)
    old_theorems = new_blob.get('theorems', {})
    new_theorems = updates['theorems']

    merged_theorems = {}
    for tid in set(list(old_theorems.keys()) + list(new_theorems.keys())):
        if tid in new_theorems:
            if tid in old_theorems:
                # Merge: new values win, but preserve old extra fields
                merged = dict(old_theorems[tid])
                merged.update(new_theorems[tid])
                merged_theorems[tid] = merged
            else:
                # New theorem not in dashboard yet
                merged_theorems[tid] = new_theorems[tid]
        else:
            # Theorem in dashboard but not in theorems.py — keep it
            # (might be manually added)
            merged_theorems[tid] = old_theorems[tid]

    new_blob['theorems'] = merged_theorems

    # Update date
    new_blob['date'] = datetime.date.today().isoformat()

    return new_blob


def write_dashboard(html, new_blob, start, end, output_path):
    """Replace the blob in the HTML and write to output."""
    blob_json = json.dumps(new_blob, ensure_ascii=False, separators=(',', ':'))
    new_html = html[:start] + blob_json + html[end:]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_html)


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------

def report_changes(old_blob, new_blob):
    """Print a summary of what changed."""
    changes = []

    # Theorem count
    old_t = old_blob.get('total_theorems', 0)
    new_t = new_blob['total_theorems']
    if old_t != new_t:
        changes.append(f"  total_theorems: {old_t} → {new_t}")

    # Epistemic counts
    old_epi = old_blob.get('epistemic_counts', {})
    new_epi = new_blob['epistemic_counts']
    for key in sorted(set(list(old_epi.keys()) + list(new_epi.keys()))):
        old_v = old_epi.get(key, 0)
        new_v = new_epi.get(key, 0)
        if old_v != new_v:
            changes.append(f"  {key}: {old_v} → {new_v}")

    # Per-theorem epistemic changes
    old_thms = old_blob.get('theorems', {})
    new_thms = new_blob.get('theorems', {})
    epi_changes = []
    for tid in sorted(new_thms.keys()):
        old_e = old_thms.get(tid, {}).get('epistemic', '?')
        new_e = new_thms[tid].get('epistemic', '?')
        if old_e != new_e:
            epi_changes.append(f"  {tid}: {old_e} → {new_e}")

    new_thm_ids = set(new_thms.keys()) - set(old_thms.keys())
    removed_ids = set(old_thms.keys()) - set(new_thms.keys())

    print("\n" + "=" * 60)
    print("SYNC REPORT")
    print("=" * 60)

    if not changes and not epi_changes and not new_thm_ids and not removed_ids:
        print("  No changes detected. Dashboard is up to date.")
        return False

    if changes:
        print("\nGLOBAL CHANGES:")
        for c in changes:
            print(c)

    if epi_changes:
        print(f"\nEPISTEMIC UPGRADES ({len(epi_changes)}):")
        for c in epi_changes:
            print(c)

    if new_thm_ids:
        print(f"\nNEW THEOREMS ({len(new_thm_ids)}):")
        for tid in sorted(new_thm_ids):
            print(f"  + {tid}: {new_thms[tid].get('name', '?')}")

    if removed_ids:
        print(f"\nREMOVED FROM CODE (kept in dashboard) ({len(removed_ids)}):")
        for tid in sorted(removed_ids):
            print(f"  - {tid}")

    print(f"\nFINAL STATE: {new_blob['total_theorems']} theorems, "
          f"{new_blob['epistemic_counts'].get('P', 0)} [P], "
          f"{new_blob['epistemic_counts'].get('P_structural', 0)} [P_structural]")
    print("=" * 60)
    return True


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sync theorems.py → dashboard HTML')
    parser.add_argument('--dashboard', default=None,
                        help='Path to dashboard HTML file')
    parser.add_argument('--output', default=None,
                        help='Output path (default: overwrite dashboard)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show changes without writing')
    args = parser.parse_args()

    # Find dashboard file
    dashboard_path = args.dashboard
    if not dashboard_path:
        # Auto-detect: look for common names
        candidates = [
            'index.html',
            'thedashboard_v41.html',
            'thedashboard_v38__3_.html',
            'dashboard.html',
        ]
        for c in candidates:
            if os.path.exists(c):
                dashboard_path = c
                break
        if not dashboard_path:
            # Try glob
            import glob
            htmls = glob.glob('thedashboard_v*.html') + glob.glob('*dashboard*.html')
            if htmls:
                dashboard_path = sorted(htmls)[-1]  # latest version
            else:
                print("ERROR: No dashboard HTML found. Use --dashboard <path>")
                sys.exit(1)

    print(f"Dashboard: {dashboard_path}")

    # Read dashboard
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Extract existing blob
    old_blob, blob_start, blob_end = extract_blob(html)
    print(f"Old blob: {old_blob.get('total_theorems', '?')} theorems, "
          f"P={old_blob.get('epistemic_counts', {}).get('P', '?')}")

    # Run theorems
    print("Running theorem bank...")
    results, n_registered = run_theorems()
    print(f"Theorem bank: {len(results)} results from {n_registered} registered checks")

    # Build updates
    updates = build_updates(results)

    # Patch blob
    new_blob = patch_blob(old_blob, updates)

    # Report
    has_changes = report_changes(old_blob, new_blob)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    if not has_changes:
        print("No changes to write.")
        return

    # Write
    output_path = args.output or dashboard_path
    write_dashboard(html, new_blob, blob_start, blob_end, output_path)
    print(f"\n✓ Dashboard written to: {output_path}")


if __name__ == '__main__':
    main()
