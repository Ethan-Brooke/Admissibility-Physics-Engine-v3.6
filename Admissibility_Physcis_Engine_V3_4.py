#!/usr/bin/env python3
"""
================================================================================
MASTER VERIFICATION ENGINE -- FCF v3.4.2 (Engine v9.2)
================================================================================

The single entry point that runs EVERYTHING.

Source:
    Admissbility_Physics_Theorms_V3_4.py  -- All 58 theorems (Tiers 0-5)

Produces:
    Unified epistemic scorecard across all 58 theorems
    Dependency DAG validation
    Tier-by-tier pass/fail
    Gap audit for every [P_structural] theorem
    JSON export for CI integration

Run:  python3 Admissibility_Physcis_Engine_V3_4.py
      python3 Admissibility_Physcis_Engine_V3_4.py --json
      python3 Admissibility_Physcis_Engine_V3_4.py --audit-gaps

No numpy. No external dependencies. Stdlib only.
================================================================================
"""

import sys
import json
import time
import inspect
import re
from typing import Dict, Any, List


# ======================================================================
#  IMPORTS -- unified theorem bank
# ======================================================================

from Admissbility_Physics_Theorms_V3_4 import (
    run_all as run_theorem_bank,
    THEOREM_REGISTRY,
)


# ======================================================================
#  AXIOMS (always satisfied by definition)
# ======================================================================

AXIOMS = {'A1', 'A2', 'A3', 'A4', 'A5'}


# ======================================================================
#  DEPENDENCY DAG VALIDATION
# ======================================================================

def validate_dependencies(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Check that every theorem's dependencies are satisfied."""
    known_ids = set(all_results.keys()) | AXIOMS

    issues = []
    for tid, r in all_results.items():
        for dep in r.get('dependencies', []):
            dep_clean = dep.split('(')[0].strip()
            if dep_clean not in known_ids and dep not in known_ids:
                if not any(dep.startswith(a) for a in AXIOMS):
                    issues.append(f"{tid} depends on '{dep}' -- not in registry")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_checked': len(all_results),
    }


# ======================================================================
#  CYCLE DETECTION
# ======================================================================

def find_cycles(all_results: Dict[str, Any]) -> List[str]:
    """Find theorems involved in dependency cycles."""
    deps = {}
    for tid, r in all_results.items():
        deps[tid] = [d for d in r.get('dependencies', []) if d in all_results]

    def has_cycle(start, current=None, visited=None):
        if current is None:
            current = start
        if visited is None:
            visited = set()
        if current in visited:
            return True
        visited.add(current)
        for d in deps.get(current, []):
            if d in deps and has_cycle(start, d, visited.copy()):
                return True
        return False

    return [tid for tid in deps if has_cycle(tid)]


# ======================================================================
#  MASTER RUN
# ======================================================================

TIER_NAMES = {
    0: 'TIER 0: Foundations + Quantum',
    1: 'TIER 1: Gauge Group Selection',
    2: 'TIER 2: Particle Content / Generations',
    3: 'TIER 3: Continuous Constants / RG',
    4: 'TIER 4: Gravity / Cosmology',
    5: 'TIER 5: Delta_geo Structural Corollaries',
}


def run_master() -> Dict[str, Any]:
    """Execute the complete verification chain."""
    t0 = time.time()

    # 1. Run unified theorem bank
    all_results = run_theorem_bank()

    elapsed = time.time() - t0

    # 2. Validate dependencies
    dep_check = validate_dependencies(all_results)

    # 3. Cycle detection
    cycles = find_cycles(all_results)

    # 4. Compute statistics
    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r['passed'])

    epistemic_counts = {}
    for r in all_results.values():
        e = r['epistemic']
        epistemic_counts[e] = epistemic_counts.get(e, 0) + 1

    # 5. Tier breakdown
    tier_stats = {}
    for tier in range(6):
        tier_results = {k: v for k, v in all_results.items() if v.get('tier') == tier}
        if tier_results:
            p_count = sum(1 for r in tier_results.values() if r['epistemic'] == 'P')
            ps_count = sum(1 for r in tier_results.values() if r['epistemic'] == 'P_structural')
            tier_stats[tier] = {
                'name': TIER_NAMES.get(tier, f'Tier {tier}'),
                'total': len(tier_results),
                'passed': sum(1 for r in tier_results.values() if r['passed']),
                'P_count': p_count,
                'P_structural_count': ps_count,
                'theorems': sorted(tier_results.keys()),
            }

    # 6. Sector verdicts
    def sector_ok(theorem_ids):
        return all(
            all_results[t]['passed']
            for t in theorem_ids
            if t in all_results
        )

    gauge_ok = sector_ok(['T_channels', 'T7', 'T_gauge', 'T5', 'T4'])
    gravity_ok = sector_ok(['T7B', 'T8', 'T9_grav', 'T10', 'T11'])
    rg_ok = sector_ok(['T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25a', 'T25b'])
    cosmo_ok = sector_ok(['T11', 'T12', 'T12E'])
    foundations_ok = sector_ok(['T0', 'T1', 'T2', 'T3', 'L_T2', 'L_nc', 'L_epsilon*'])
    geometry_ok = sector_ok([
        'Delta_ordering', 'Delta_fbc', 'Delta_continuum',
        'Delta_signature', 'Delta_closure', 'Delta_particle',
    ])

    # 7. Assertion count
    n_asserts = sum(
        inspect.getsource(f).count('assert ')
        for f in THEOREM_REGISTRY.values()
    )

    return {
        'version': '3.4.2',
        'engine_version': 'v9.2',
        'total_theorems': total,
        'passed': passed,
        'all_pass': passed == total,
        'all_results': all_results,
        'epistemic_counts': epistemic_counts,
        'tier_stats': tier_stats,
        'dependency_check': dep_check,
        'cycles': cycles,
        'n_assertions': n_asserts,
        'elapsed_s': round(elapsed, 3),
        'sector_verdicts': {
            'foundations': foundations_ok,
            'gauge': gauge_ok,
            'rg_mechanism': rg_ok,
            'gravity': gravity_ok,
            'cosmology': cosmo_ok,
            'geometry': geometry_ok,
        },
    }


# ======================================================================
#  DISPLAY
# ======================================================================

def display(master: Dict[str, Any]):
    W = 74

    def header(text):
        print(f"\n{'=' * W}")
        print(f"  {text}")
        print(f"{'=' * W}")

    def subheader(text):
        print(f"\n{'-' * W}")
        print(f"  {text}")
        print(f"{'-' * W}")

    header(f"MASTER VERIFICATION ENGINE -- FCF v{master['version']} (Engine {master['engine_version']})")
    print(f"\n  Total theorems:  {master['total_theorems']}")
    print(f"  Passed:          {master['passed']}/{master['total_theorems']}")
    print(f"  All pass:        {'YES' if master['all_pass'] else 'NO'}")
    print(f"  Assertions:      {master['n_assertions']}")
    print(f"  Runtime:         {master['elapsed_s']:.2f}s")

    # -- Sector verdicts --
    subheader("SECTOR VERDICTS")
    for sector, ok in master['sector_verdicts'].items():
        mark = 'ok' if ok else '!!'
        print(f"  [{mark}] {sector}")

    # -- Tier breakdown --
    for tier in sorted(master['tier_stats'].keys()):
        ts = master['tier_stats'][tier]
        subheader(
            f"{ts['name']} -- {ts['passed']}/{ts['total']} pass "
            f"({ts['P_count']}[P] {ts['P_structural_count']}[Ps])"
        )
        for tid in ts['theorems']:
            r = master['all_results'][tid]
            mark = 'ok' if r['passed'] else '!!'
            epi = f"[{r['epistemic']}]"
            kr = r.get('key_result', '')
            if len(kr) > 48:
                kr = kr[:45] + '...'
            print(f"  [{mark}] {tid:20s} {epi:18s} {kr}")

    # -- Epistemic summary --
    header("EPISTEMIC DISTRIBUTION")
    for e in sorted(master['epistemic_counts'].keys()):
        ct = master['epistemic_counts'][e]
        bar = '#' * ct
        print(f"  [{e:14s}] {ct:3d}  {bar}")

    # -- Dependency check --
    subheader("DEPENDENCY VALIDATION")
    dc = master['dependency_check']
    print(f"  Checked: {dc['total_checked']} theorems")
    print(f"  Valid:   {'YES' if dc['valid'] else 'NO'}")
    if dc['issues']:
        for issue in dc['issues'][:10]:
            print(f"    !! {issue}")

    # -- Cycles --
    subheader("CYCLE DETECTION")
    if master['cycles']:
        print(f"  {len(master['cycles'])} theorems in cycles (co-determined quantities):")
        print(f"    {', '.join(sorted(master['cycles']))}")
    else:
        print(f"  No dependency cycles detected.")

    # -- The honest scorecard --
    header("THE HONEST SCORECARD")

    print("""
  +--------------------------------------------------------------+
  |  WHAT IS PROVED [P] (39 theorems)                            |
  |                                                              |
  |  Tier 0 (15):                                                |
  |    L_T2: finite GNS (zero imports, constructive)             |
  |    L_nc: non-commutativity witness                           |
  |    L_epsilon*: meaningful -> epsilon > 0 (compactness)       |
  |    T0-T3: axiom encoding -> quantum -> operator -> gauge     |
  |    T_Born: Born rule from Gleason (dim >= 3)                 |
  |    T_CPTP, T_Hermitian, T_M, T_entropy, T_epsilon, T_eta    |
  |    T_tensor: tensor product from independent subsystems      |
  |                                                              |
  |  Tier 1 (2):                                                 |
  |    T5: unique hypercharge (z^2 - 2z - 8 = 0)                |
  |    T_gauge: SU(3)*SU(2)*U(1) = capacity optimum             |
  |                                                              |
  |  Tier 2 (5):                                                 |
  |    T4E: N_gen >= 3 from CP -> baryogenesis                   |
  |    T4F: N_gen <= 3 from capacity saturation (C_int = 8)      |
  |    T7: N_gen = 3 (combined)                                  |
  |    T9: 3! = 6 record sectors                                 |
  |    T_channels: channels_EW = 4 (anomaly scan)                |
  |                                                              |
  |  Tier 3 (10):                                                |
  |    T19-T20: routing + RG flow from capacity competition      |
  |    T22-T23: competition matrix + fixed point                 |
  |    T25a-T25b: overlap bounds from coexistence + monogamy     |
  |    T26: gamma ratio bounds (inequality chain)                |
  |    T27c-T27d: x* and gamma* at saturation                   |
  |    T_S0: admissibility gate                                  |
  |                                                              |
  |  Tier 4 (1): T_Bek (Bekenstein bound)                        |
  |  Tier 5 (6): Delta_ordering/fbc/continuum/signature/         |
  |              closure/particle                                |
  +--------------------------------------------------------------+
  |  WHAT IS STRUCTURALLY DERIVED [P_structural] (19 theorems)   |
  |                                                              |
  |  Tier 0 (1): T_kappa (kappa=2, exhaustion vs minimality)     |
  |  Tier 1 (2): T4 (anomaly-free), T_particle (mass gap)       |
  |  Tier 2 (4): T4G, T4G_Q31, T_Higgs, T_field                |
  |  Tier 3 (5): T6, T6B, T21, T24, T_sin2theta                |
  |  Tier 4 (6): T7B, T8, T10, T11, T12, T12E                  |
  |  Tier 5 (1): T9_grav                                        |
  +--------------------------------------------------------------+
  |  KEY NUMERICAL PREDICTIONS                                   |
  |                                                              |
  |  sin^2(theta_W) = 3/13 = 0.23077 (obs: 0.23122, 0.19%)     |
  |  N_gen = 3 (exact)                                           |
  |  Gauge group = SU(3)*SU(2)*U(1) (exact)                     |
  |  f_b = 3/19 = 0.1579 (obs: 0.1571, 0.49%)                  |
  |  Omega_Lambda = 42/61 = 0.6885 (obs: 0.6889, 0.05%)        |
  |  Omega_m = 19/61 = 0.3115 (obs: 0.3111, 0.12%)             |
  +--------------------------------------------------------------+
  |  IMPORTED EXTERNAL THEOREMS (not framework gaps)             |
  |                                                              |
  |  Kochen-Specker (1967) -- contextuality                     |
  |  GNS construction (1943) -- C*-algebra -> Hilbert space      |
  |  Kadison / Hahn-Banach (1951) -- state existence             |
  |  Skolem-Noether -- Aut(M_n) structure                        |
  |  Doplicher-Roberts (1989) -- gauge group recovery            |
  |  Anomaly cancellation -- standard gauge anomaly machinery    |
  |  1-loop beta-coefficients -- standard QFT                    |
  |  Lovelock theorem -- unique tensor in d=4                    |
  |  Hawking-King-McCarthy (1976) -- causal -> conformal         |
  |  Kolmogorov extension (1933) -- consistent -> measure        |
  +--------------------------------------------------------------+
  |  REGIME ASSUMPTIONS (explicit, not axioms)                   |
  |                                                              |
  |  R12.0: No superselection onto gauge-charged subspace        |
  |  R12.1: Linear enforcement cost scaling (proxy)              |
  |  R12.2: Capacity-efficient realization (selection)           |
  +--------------------------------------------------------------+
""")

    # -- Derivation chain --
    header("THE COMPLETE DERIVATION CHAIN")
    print("""
  AXIOMS A1-A5  (finite enforceability -> orientational structure)
      |
      +== TIER 0: Foundations =====================================
      |   L_T2: Finite GNS (sigma_x, sigma_z -> M_2(C))    [P]
      |   L_nc: Non-commutativity witness                    [P]
      |   L_epsilon*: Meaningful -> epsilon > 0              [P]
      |   T0: Axiom encoding                                 [P]
      |   T1: Non-closure -> incompatible observables        [P]
      |   T2: -> Operator algebra (Layer 1: L_T2 [P])       [P]
      |   T3: Locality -> gauge bundles                      [P]
      |   T_Born: Born rule from Gleason                     [P]
      |   T_epsilon,eta,kappa,M: Granularity + monogamy      [P/Ps]
      |
      +== TIER 1: Gauge Group ====================================
      |   T4: Anomaly-free chiral net                  [P_structural]
      |   T5: Unique hypercharge (z^2-2z-8 = 0)              [P]
      |   T_gauge: SU(3)*SU(2)*U(1) = capacity optimum       [P]
      |
      +== TIER 2: Particles ======================================
      |   T_channels: channels = 4 (anomaly scan)             [P]
      |   T7: N_gen = 3 (T4E + T4F combined)                  [P]
      |   T4E-G: Mass hierarchy, Yukawa, nu bound       [P/P_struct]
      |   T_field: 5 multiplet types                   [P_structural]
      |   T_Higgs: Massive scalar from EW pivot        [P_structural]
      |
      +== TIER 3: RG / Constants =================================
      |   T6/T6B: Beta-function + fixed point          [P_structural]
      |   T19-T23: Routing -> competition -> fixed pt    [P/P_struct]
      |   T24: sin^2(theta_W) = 3/13 witness           [P_structural]
      |   T25-27: Overlap + gamma bounds                 [P/P_struct]
      |
      +== TIER 4: Gravity / Cosmology ============================
      |   T7B: Shared interface -> metric              [P_structural]
      |   T8: d = 4                                    [P_structural]
      |   T9_grav: Einstein eqns (Lovelock)            [P_structural]
      |   T10: kappa ~ 1/C_*                           [P_structural]
      |   T11: Lambda from global capacity residual    [P_structural]
      |   T12: DM from capacity stratification         [P_structural]
      |        (MECE audit: partition clean, budget closes)
      |   T12E: f_b = 3/19, Omega_Lambda = 42/61      [P_structural]
      |   T_Bek: Bekenstein bound                             [P]
      |
      +== TIER 5: Delta_geo Closure ==============================
          Delta_ordering: R1-R4 from A4                        [P]
          Delta_fbc: Lipschitz fluctuation bound               [P]
          Delta_continuum: lattice -> C^2 metric               [P]
          Delta_signature: (-,+,+,+) from A4                   [P]
          Delta_closure: all A9.1-A9.5 derived                 [P]
          Delta_particle: particle emergence                    [P]
""")

    print(f"{'=' * W}")
    all_ok = master['all_pass']
    status = 'ALL THEOREMS PASS' if all_ok else 'SOME FAILURES'
    print(f"  FRAMEWORK STATUS: {status}")
    print(f"  {master['passed']}/{master['total_theorems']} theorems verified")
    print(f"  {master['n_assertions']} assertions | {master['elapsed_s']:.2f}s")
    print(f"{'=' * W}")


# ======================================================================
#  GAP REGISTRY -- What makes each [P_structural] less than [P]?
# ======================================================================

GAP_REGISTRY = {
    # ================================================================
    # TIER 0: FOUNDATIONS
    # ================================================================
    'T_kappa': {
        'anchor': 'A4 (backward) + A5 (forward) uniqueness proof',
        'gap': 'OPEN QUESTION. kappa >= 2 is unambiguous [P] (independent '
               'forward + backward commitments). kappa <= 2 depends on '
               'whether axiom exhaustion (only A4,A5 generate per-direction '
               'obligations) constitutes a proof or a minimality assumption. '
               'If A1-A5 are definitionally complete, exhaustion gives [P].',
        'to_close': 'Clarify whether A1-A5 completeness is definitional or '
                    'assumed. If definitional: upgrade to [P].',
    },
    # ================================================================
    # TIER 1: GAUGE GROUP
    # ================================================================
    'T4': {
        'anchor': 'Anomaly cancellation (standard QFT)',
        'gap': 'IMPORT. A2 (non-closure) forces chiral fermions in anomaly-free '
               'representations. The anomaly cancellation conditions are standard '
               'QFT results verified experimentally.',
        'to_close': 'N/A -- importing experimentally verified QFT results.',
    },
    'T_particle': {
        'anchor': 'V(Phi) from L_epsilon*, T_M, A1',
        'gap': 'STRUCTURAL. SSB forced by A4. Mass gap from d2V > 0 at binding '
               'well. Particles require T1+T2 quantum structure (no classical '
               'soliton localizes). The structural argument is complete but '
               'the connection to specific particle masses requires T10.',
        'to_close': 'Particle spectrum needs T10 (Planck mass). Mechanism is closed.',
    },
    # ================================================================
    # TIER 2: PARTICLES
    # ================================================================
    'T4G': {
        'anchor': 'Yukawa structure from capacity hierarchy',
        'gap': 'OPEN PHYSICS. Structural derivation of Yukawa pattern is complete. '
               'Quantitative neutrino mass prediction requires Majorana/Dirac '
               'identification -- a genuine open physics question.',
        'to_close': 'Requires resolving Majorana vs Dirac nature (experimental).',
    },
    'T4G_Q31': {
        'anchor': 'Q31 neutrino mass bound',
        'gap': 'OPEN PHYSICS. Upper bound correct order of magnitude. '
               'Sharpening requires Majorana/Dirac identification.',
        'to_close': 'Same as T4G -- requires experimental neutrino physics.',
    },
    'T_Higgs': {
        'anchor': 'Massive scalar from EW pivot',
        'gap': 'CLOSED for structural claim. SSB forced by A4. Positive curvature '
               'at pivot -> massive Higgs-like scalar. Screening exponent derived. '
               'Absolute mass scale requires T10 (Planck mass).',
        'to_close': 'Structural claim closed. Mass value needs T10.',
    },
    'T_field': {
        'anchor': 'AF bounds are QFT imports',
        'gap': 'IMPORT. The scan + exclusion logic is pure combinatorics [P]. '
               'The asymptotic freedom bounds that constrain representation '
               'content are imported from perturbative QFT. If T_gauge absorbs '
               'the AF import, T_field\'s own contribution is [P].',
        'to_close': 'Clarify whether AF bounds propagate through T_gauge or '
                    'are independently imported by T_field.',
    },
    # ================================================================
    # TIER 3: RG / CONSTANTS
    # ================================================================
    'T6': {
        'anchor': 'Capacity-weighted beta-function',
        'gap': 'IMPORT. Beta-function FORM (capacity competition drives running) '
               'is derived from A1 + A5. The 1-loop COEFFICIENTS (b_i) are '
               'imported from standard QFT perturbation theory.',
        'to_close': 'N/A -- 1-loop coefficients are imported, experimentally confirmed.',
    },
    'T6B': {
        'anchor': 'Capacity running fixed point',
        'gap': 'IMPORT. Fixed-point existence is structural. Location uses '
               'imported beta-coefficients. Same import as T6.',
        'to_close': 'N/A -- same import as T6.',
    },
    'T21': {
        'anchor': 'Beta-saturation bound',
        'gap': 'REDUCED. Saturation mechanism fully derived. Same import '
               'situation as T6/T6B for numerical coefficients.',
        'to_close': 'Mechanism closed. Same import status as T6.',
    },
    'T24': {
        'anchor': 'sin^2(theta_W) = 3/13 witness',
        'gap': 'STRUCTURAL CHAIN. x* from T27c, gamma* from T27d, gate S0. '
               'The chain is fully mechanized but depends on T6 imports. '
               'The prediction (0.19% accuracy) is a strong structural witness.',
        'to_close': 'Remove T6 import dependency (would require deriving '
                    'beta-coefficients from first principles).',
    },
    'T_sin2theta': {
        'anchor': 'Fixed-point mechanism for Weinberg angle',
        'gap': 'Same chain as T24. The mechanism (capacity competition -> '
               'fixed point -> mixing angle) is fully derived. Numerical '
               'value depends on imported beta-coefficients.',
        'to_close': 'Same as T24.',
    },
    # ================================================================
    # TIER 4: GRAVITY / COSMOLOGY
    # ================================================================
    'T7B': {
        'anchor': 'Shared interface -> metric tensor',
        'gap': 'STRUCTURAL. Polarization identity gives metric from quadratic '
               'feasibility functional. The functional form (quadratic in '
               'displacement) is structural but not fully proven from A1-A5 '
               'alone without continuity assumptions.',
        'to_close': 'Derive quadratic form from Delta_fbc + A1 bounds.',
    },
    'T8': {
        'anchor': 'Spacetime dimension d = 4',
        'gap': 'STRUCTURAL. d <= 3 hard-excluded (no chiral fermions). d = 4 '
               'minimally admissible. d >= 5 requires showing capacity excess. '
               'Structural argument complete but optimality proof could be tightened.',
        'to_close': 'Show d >= 5 violates capacity bound or prove d = 4 '
                    'is the unique minimum.',
    },
    'T10': {
        'anchor': 'Gravitational coupling kappa ~ 1/C_*',
        'gap': 'OPEN PHYSICS. Structural derivation: coupling inversely proportional '
               'to geometric capacity. Quantitative value requires UV completion.',
        'to_close': 'Requires UV completion (same as quantitative Lambda).',
    },
    'T11': {
        'anchor': 'Lambda from global capacity residual',
        'gap': 'STRUCTURAL. Mechanism: globally locked correlations -> uniform '
               'curvature pressure -> cosmological constant. Lambda > 0 from '
               'positive enforcement cost. Quantitative value via T12E capacity '
               'counting: Omega_Lambda = 42/61 = 0.6885 (obs: 0.6889, 0.05%). '
               'Regime assumptions: capacity saturation.',
        'to_close': 'Derive saturation fraction from first principles.',
    },
    'T12': {
        'anchor': 'DM from capacity stratification',
        'gap': 'STRUCTURAL. Core argument [P]: gauge and gravity couple to '
               'different scope interfaces, so C_local = C_gauge + C_singlet. '
               'Existence of C_singlet > 0 requires R12.0 (no superselection). '
               'Dominance (Omega_DM > Omega_b) requires R12.1 + R12.2 (cost '
               'scaling + efficiency). MECE audit clean: partition exhaustive, '
               'exclusive, budget closes. Alpha overhead = 4 consistent with '
               'observed DM/baryon ratio.',
        'to_close': 'Derive R12.0 from axioms (show superselection violates A5). '
                    'Exact ratio requires UV completion.',
    },
    'T12E': {
        'anchor': 'f_b = 3/19 from capacity counting',
        'gap': 'STRUCTURAL. Capacity budget: 3 gen labels + 16 multiplet refs = '
               '19 matter + 42 vacuum = 61 total. f_b = 3/19 = 0.1579 (obs: '
               '0.1571, 0.49%). The counting depends on T_field content and '
               'T_Higgs identification. Alpha consistency check: Omega_DM/Omega_b '
               '= 5.33 > alpha = 4 (gauge overhead floor).',
        'to_close': 'Derive the 61-unit total from first principles rather than '
                    'from the gauge group + field content.',
    },
    # ================================================================
    # TIER 5: GEOMETRY
    # ================================================================
    'T9_grav': {
        'anchor': 'Einstein field equations via Lovelock',
        'gap': 'IMPORT. Lovelock theorem: in d = 4, the unique divergence-free, '
               'second-order, symmetric tensor built from the metric is the '
               'Einstein tensor + Lambda term. This is a mathematical theorem, '
               'not a conjecture. Delta_closure derives all Lovelock premises.',
        'to_close': 'N/A -- Lovelock is a proven mathematical theorem.',
    },
}


def _classify_gap(tid: str) -> str:
    """Classify gap type for a P_structural theorem."""
    if tid not in GAP_REGISTRY:
        return 'unclassified'
    gap_text = GAP_REGISTRY[tid].get('gap', '').strip().upper()
    # Check the FIRST WORD to avoid false matches from descriptions
    # that mention "import" or "structural" in passing
    first_word = gap_text.split('.')[0].split()[0] if gap_text else ''
    if first_word == 'CLOSED':
        return 'closed'
    if first_word == 'IMPORT':
        return 'import'
    if gap_text.startswith('OPEN PHYSICS'):
        return 'open_physics'
    if first_word == 'REDUCED':
        return 'reduced'
    if gap_text.startswith('OPEN QUESTION'):
        return 'open_question'
    if first_word in ('STRUCTURAL', 'SAME'):
        return 'structural'
    return 'other'


# ======================================================================
#  AUDIT-GAPS REPORTER
# ======================================================================

def display_audit_gaps(master: Dict[str, Any]):
    """Display detailed gap analysis for every [P_structural] theorem."""
    W = 74
    all_r = master['all_results']

    print(f"\n{'=' * W}")
    print(f"  GAP AUDIT -- Every [P_structural] theorem")
    print(f"  What specifically prevents upgrade to [P]?")
    print(f"{'=' * W}")

    # Group P_structural by tier
    p_struct = {
        tid: r for tid, r in all_r.items()
        if r['epistemic'] == 'P_structural'
    }
    by_type = {}
    for tid in p_struct:
        gtype = _classify_gap(tid)
        by_type.setdefault(gtype, []).append(tid)

    # Summary first
    print(f"\n  {len(p_struct)} theorems at [P_structural]:")
    for gtype in sorted(by_type.keys()):
        tids = by_type[gtype]
        print(f"    [{gtype:14s}] {len(tids):2d}  {', '.join(sorted(tids))}")

    # Detail by tier
    tiers = {}
    for tid, r in p_struct.items():
        tiers.setdefault(r['tier'], {})[tid] = r

    for tier in sorted(tiers.keys()):
        tier_ps = tiers[tier]
        print(f"\n{'-' * W}")
        print(f"  {TIER_NAMES.get(tier, f'TIER {tier}')}")
        print(f"{'-' * W}")

        for tid in sorted(tier_ps.keys()):
            gap_info = GAP_REGISTRY.get(tid, {})
            anchor = gap_info.get('anchor', '(no anchor registered)')
            gap = gap_info.get('gap', '(no gap description)')
            to_close = gap_info.get('to_close', '(not specified)')
            gtype = _classify_gap(tid)

            print(f"\n  {tid}")
            print(f"    Anchor:   {anchor}")
            print(f"    Gap type: [{gtype}]")
            print(f"    Gap:      {gap}")
            print(f"    To close: {to_close}")

    # Imported theorems
    print(f"\n{'=' * W}")
    print(f"  IMPORTED EXTERNAL THEOREMS")
    print(f"{'=' * W}")
    imports = [
        ('T1',             'Kochen-Specker (1967)',        'Contextuality of quantum observables'),
        ('T2',             'GNS construction (1943)',      'State -> Hilbert space representation'),
        ('T2',             'Kadison / Hahn-Banach (1951)', 'State existence on C*-algebras'),
        ('T3',             'Skolem-Noether',               'Automorphism structure of matrix algebras'),
        ('T3',             'Doplicher-Roberts (1989)',     'Gauge group recovery from superselection'),
        ('T4',             'Anomaly cancellation',         'Standard gauge anomaly machinery'),
        ('T6',             '1-loop beta-coefficients',     'Standard QFT perturbative results'),
        ('T9_grav',        'Lovelock theorem',             'Unique 2nd-order tensor in d=4'),
        ('Delta_signature','HKM (1976) / Malament (1977)', 'Causal structure -> conformal Lorentzian'),
        ('Delta_continuum','Kolmogorov extension (1933)',  'Consistent families -> measure'),
    ]
    for tid, name, desc in imports:
        print(f"  {tid:18s} <- {name}")
        print(f"  {'':18s}    {desc}")

    # Summary stats
    print(f"\n{'=' * W}")
    print(f"  CLOSURE SUMMARY")
    print(f"{'=' * W}")

    n_closed = len(by_type.get('closed', []))
    n_import = len(by_type.get('import', []))
    n_structural = len(by_type.get('structural', []))
    n_open = len(by_type.get('open_physics', []))
    n_reduced = len(by_type.get('reduced', []))
    n_question = len(by_type.get('open_question', []))

    print(f"\n  {len(p_struct)} [P_structural] theorems assessed:")
    print(f"    {n_closed} closed (structural claim complete)")
    print(f"    {n_import} imports (proven external theorems, not gaps)")
    print(f"    {n_structural} structural (mechanism complete, regime assumptions)")
    print(f"    {n_reduced} reduced (mechanism complete, details remain)")
    print(f"    {n_question} open questions (definitional clarity needed)")
    print(f"    {n_open} open physics (requires experimental input)")
    print(f"\n  Effective gap count: {n_open} experimental + "
          f"{n_question} definitional = {n_open + n_question} actionable")
    print(f"{'=' * W}")


# ======================================================================
#  JSON EXPORT
# ======================================================================

def export_json(master: Dict[str, Any]) -> str:
    """Export machine-readable report."""
    report = {
        'version': master['version'],
        'engine_version': master['engine_version'],
        'total_theorems': master['total_theorems'],
        'passed': master['passed'],
        'all_pass': master['all_pass'],
        'n_assertions': master['n_assertions'],
        'elapsed_s': master['elapsed_s'],
        'epistemic_counts': master['epistemic_counts'],
        'sector_verdicts': master['sector_verdicts'],
        'tier_stats': {
            str(k): {
                'name': v['name'],
                'passed': v['passed'],
                'total': v['total'],
                'P_count': v['P_count'],
                'P_structural_count': v['P_structural_count'],
            }
            for k, v in master['tier_stats'].items()
        },
        'dependency_check': {
            'valid': master['dependency_check']['valid'],
            'issues': master['dependency_check']['issues'],
        },
        'cycles': master['cycles'],
        'theorems': {},
    }
    for tid, r in master['all_results'].items():
        entry = {
            'name': r['name'],
            'tier': r.get('tier', -1),
            'passed': r['passed'],
            'epistemic': r['epistemic'],
            'key_result': r.get('key_result', ''),
        }
        if r['epistemic'] == 'P_structural' and tid in GAP_REGISTRY:
            gap = GAP_REGISTRY[tid]
            entry['gap'] = {
                'anchor': gap.get('anchor', ''),
                'type': _classify_gap(tid),
                'description': gap.get('gap', ''),
                'to_close': gap.get('to_close', ''),
            }
        report['theorems'][tid] = entry

    return json.dumps(report, indent=2)


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == '__main__':
    master = run_master()

    if '--json' in sys.argv:
        print(export_json(master))
    elif '--audit-gaps' in sys.argv:
        display_audit_gaps(master)
    else:
        display(master)

    sys.exit(0 if master['all_pass'] else 1)
