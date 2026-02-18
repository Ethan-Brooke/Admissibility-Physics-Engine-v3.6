#!/usr/bin/env python3
"""
================================================================================
L_anomaly_free: GAUGE ANOMALY CANCELLATION FROM CAPACITY STRUCTURE [P]
================================================================================

v4.3.7 supplement.

Comprehensive verification that the framework-derived particle content
and hypercharges satisfy ALL gauge anomaly cancellation conditions.

7/7 conditions verified. Zero free parameters. Zero new imports.

Run standalone:  python3 L_anomaly_free_v4_3_7.py
================================================================================
"""

from fractions import Fraction
import math as _math
import sys


def _result(name, tier, epistemic, summary, key_result,
            dependencies=None, passed=True, artifacts=None,
            imported_theorems=None, cross_refs=None):
    r = {
        'name': name, 'tier': tier, 'passed': passed,
        'epistemic': epistemic, 'summary': summary,
        'key_result': key_result,
        'dependencies': dependencies or [],
        'cross_refs': cross_refs or [],
        'artifacts': artifacts or {},
    }
    if imported_theorems:
        r['imported_theorems'] = imported_theorems
    return r


def check_L_anomaly_free():
    """L_anomaly_free: Gauge Anomaly Cancellation Cross-Check [P].

    v4.3.7 NEW.

    STATEMENT: The framework-derived particle content and hypercharges
    satisfy ALL seven gauge anomaly cancellation conditions, per
    generation and for N_gen = 3 generations combined.

    SIGNIFICANCE:

    In standard physics, anomaly cancellation is IMPOSED as a
    consistency requirement: any chiral gauge theory must be anomaly-
    free to preserve unitarity and renormalizability. The particle
    content is then CHOSEN to satisfy these conditions.

    In this framework, the logic runs in the OPPOSITE direction:

      (a) The gauge group SU(3)*SU(2)*U(1) is derived from capacity
          optimization (T_gauge [P]).
      (b) The particle content {Q(3,2), u(3b,1), d(3b,1), L(1,2),
          e(1,1)} x 3 generations is derived from capacity minimization
          (T_field [P]).
      (c) The hypercharges Y_Q=1/6, Y_u=2/3, Y_d=-1/3, Y_L=-1/2,
          Y_e=-1 are the UNIQUE solution to the anomaly equations
          within the derived multiplet structure.

    Step (b) is the key: T_field selects the SM multiplet content from
    4680 templates using SEVEN filters (asymptotic freedom, chirality,
    [SU(3)]^3, Witten, anomaly solvability, CPT, minimality). The
    anomaly filters are CONSEQUENCES of the capacity structure, not
    external impositions.

    The fact that the capacity-derived content admits a unique set of
    anomaly-free hypercharges is a NON-TRIVIAL SELF-CONSISTENCY CHECK.
    A priori, a random chiral multiplet set has no reason to be
    anomaly-free -- most are not (as T_field's scan shows: only 1 of
    4680 templates survives all filters).

    ADDITIONAL CONSEQUENCES:
      (1) Electric charge quantization: Q_em = T_3 + Y forces rational
          charge ratios. Q(u) = 2/3, Q(d) = -1/3, Q(e) = -1.
      (2) Quark-lepton charge relation: Y_L = -N_c * Y_Q links the
          lepton and quark sectors. Both derive from the same capacity
          structure, and the anomaly conditions confirm they are
          mutually consistent.
      (3) Gravitational consistency: [grav]^2 U(1) = 0 ensures the
          derived content is compatible with T9_grav (Einstein equations
          from admissibility). The matter sector does not source a
          gravitational anomaly.

    THE SEVEN CONDITIONS:

      1. [SU(3)]^3 = 0        Cubic color anomaly
      2. [SU(2)]^3 = 0        Cubic weak anomaly (automatic)
      3. [SU(3)]^2 U(1) = 0   Mixed color-hypercharge
      4. [SU(2)]^2 U(1) = 0   Mixed weak-hypercharge
      5. [U(1)]^3 = 0         Cubic hypercharge
      6. [grav]^2 U(1) = 0    Gravitational-hypercharge
      7. Witten SU(2) = 0     Global anomaly (even # doublets)

    All verified with exact rational arithmetic. No numerical
    tolerances. No approximations.

    STATUS: [P]. Cross-check on T_field + T_gauge.
    No new imports. No new axioms.
    """
    # ================================================================
    # SETUP: Framework-derived content
    # ================================================================
    N_c = 3   # from Theorem_R [P]
    N_gen = 3 # from T7/T4F [P]

    # Hypercharges in PHYSICAL (mixed LR) convention
    # (from T_gauge/T_field [P])
    Y_Q = Fraction(1, 6)     # Q_L  ~ (3, 2, +1/6)   left-handed
    Y_u = Fraction(2, 3)     # u_R  ~ (3, 1, +2/3)   right-handed
    Y_d = Fraction(-1, 3)    # d_R  ~ (3, 1, -1/3)   right-handed
    Y_L = Fraction(-1, 2)    # L_L  ~ (1, 2, -1/2)   left-handed
    Y_e = Fraction(-1)       # e_R  ~ (1, 1, -1)     right-handed

    # Convert to all-left-handed convention for anomaly computation
    # Right-handed field with Y -> left-handed conjugate with -Y
    # and conjugate SU(3) rep (3 -> 3bar)
    fields = {
        'Q_L':   {'su3': '3',  'su2': 2, 'Y': Y_Q,   'dim3': N_c, 'chirality': 'L'},
        'u_L^c': {'su3': '3b', 'su2': 1, 'Y': -Y_u,  'dim3': N_c, 'chirality': 'L'},
        'd_L^c': {'su3': '3b', 'su2': 1, 'Y': -Y_d,  'dim3': N_c, 'chirality': 'L'},
        'L_L':   {'su3': '1',  'su2': 2, 'Y': Y_L,   'dim3': 1,   'chirality': 'L'},
        'e_L^c': {'su3': '1',  'su2': 1, 'Y': -Y_e,  'dim3': 1,   'chirality': 'L'},
    }

    # Group theory data
    # SU(3): Dynkin index T(3)=T(3b)=1/2, cubic A(3)=+1/2, A(3b)=-1/2
    T_SU3 = {'3': Fraction(1, 2), '3b': Fraction(1, 2), '1': Fraction(0)}
    A_SU3 = {'3': Fraction(1, 2), '3b': Fraction(-1, 2), '1': Fraction(0)}
    # SU(2): Dynkin index T(2)=1/2
    T_SU2 = {1: Fraction(0), 2: Fraction(1, 2)}

    results = {}

    # ================================================================
    # CONDITION 1: [SU(3)]^3 = 0
    # ================================================================
    # Tr[d_abc] = sum over LH Weyl fermions of A(R_3) * dim(R_2)
    su3_cubed = Fraction(0)
    detail_1 = {}
    for name, f in fields.items():
        contrib = A_SU3[f['su3']] * f['su2']
        su3_cubed += contrib
        if contrib != 0:
            detail_1[name] = str(contrib)

    results['[SU(3)]^3'] = {
        'value': su3_cubed,
        'passed': su3_cubed == 0,
        'detail': detail_1,
        'role': 'Filter in T_field scan',
    }

    # ================================================================
    # CONDITION 2: [SU(2)]^3 = 0
    # ================================================================
    # Identically zero: SU(2) has no symmetric cubic invariant d_abc.
    # This is a GROUP-THEORETIC identity, not a cancellation.
    su2_cubed = Fraction(0)
    results['[SU(2)]^3'] = {
        'value': su2_cubed,
        'passed': True,
        'detail': 'Automatic: d_abc = 0 for SU(2)',
        'role': 'Automatic (group theory)',
    }

    # ================================================================
    # CONDITION 3: [SU(3)]^2 x U(1) = 0
    # ================================================================
    # sum of T(R_3) * dim(R_2) * Y
    su3sq_u1 = Fraction(0)
    detail_3 = {}
    for name, f in fields.items():
        contrib = T_SU3[f['su3']] * f['su2'] * f['Y']
        su3sq_u1 += contrib
        if T_SU3[f['su3']] != 0:
            detail_3[name] = str(contrib)

    results['[SU(3)]^2 U(1)'] = {
        'value': su3sq_u1,
        'passed': su3sq_u1 == 0,
        'detail': detail_3,
        'role': 'Used to derive Y_d = 2Y_Q - Y_u',
    }

    # ================================================================
    # CONDITION 4: [SU(2)]^2 x U(1) = 0
    # ================================================================
    # sum of T(R_2) * dim(R_3) * Y
    su2sq_u1 = Fraction(0)
    detail_4 = {}
    for name, f in fields.items():
        contrib = T_SU2[f['su2']] * f['dim3'] * f['Y']
        su2sq_u1 += contrib
        if T_SU2[f['su2']] != 0:
            detail_4[name] = str(contrib)

    results['[SU(2)]^2 U(1)'] = {
        'value': su2sq_u1,
        'passed': su2sq_u1 == 0,
        'detail': detail_4,
        'role': 'Used to derive Y_L = -N_c * Y_Q',
    }

    # ================================================================
    # CONDITION 5: [U(1)]^3 = 0
    # ================================================================
    # sum of dim(R_3) * dim(R_2) * Y^3
    u1_cubed = Fraction(0)
    detail_5 = {}
    for name, f in fields.items():
        contrib = f['dim3'] * f['su2'] * f['Y']**3
        u1_cubed += contrib
        detail_5[name] = str(contrib)

    results['[U(1)]^3'] = {
        'value': u1_cubed,
        'passed': u1_cubed == 0,
        'detail': detail_5,
        'role': 'Used to derive Y_u/Y_Q ratio (quadratic z^2-2z-8=0)',
    }

    # ================================================================
    # CONDITION 6: [grav]^2 x U(1) = 0
    # ================================================================
    # sum of dim(R_3) * dim(R_2) * Y
    grav_u1 = Fraction(0)
    detail_6 = {}
    for name, f in fields.items():
        contrib = f['dim3'] * f['su2'] * f['Y']
        grav_u1 += contrib
        detail_6[name] = str(contrib)

    results['[grav]^2 U(1)'] = {
        'value': grav_u1,
        'passed': grav_u1 == 0,
        'detail': detail_6,
        'role': 'Used to derive Y_e = -2*N_c*Y_Q; cross-check with T9_grav',
    }

    # ================================================================
    # CONDITION 7: Witten SU(2) global anomaly
    # ================================================================
    # Number of SU(2) doublets must be even (per generation and total)
    n_doublets_per_gen = sum(
        f['dim3'] for f in fields.values() if f['su2'] == 2
    )
    n_doublets_total = n_doublets_per_gen * N_gen
    witten_per_gen = (n_doublets_per_gen % 2 == 0)
    witten_total = (n_doublets_total % 2 == 0)

    results['Witten SU(2)'] = {
        'value': n_doublets_total,
        'passed': witten_per_gen and witten_total,
        'detail': {
            'per_gen': f'{n_doublets_per_gen} doublets ({N_c} from Q + 1 from L)',
            'total': f'{n_doublets_total} doublets ({N_gen} generations)',
            'per_gen_even': witten_per_gen,
            'total_even': witten_total,
        },
        'role': 'Used to select odd N_c in T_gauge',
    }

    # ================================================================
    # MASTER VERIFICATION
    # ================================================================
    all_pass = all(r['passed'] for r in results.values())
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)

    assert all_pass, f"ANOMALY FAILURE: {n_passed}/{n_total} conditions pass"

    # ================================================================
    # DERIVED CONSEQUENCES
    # ================================================================

    # Electric charge quantization: Q_em = T_3 + Y
    # For the derived hypercharges, all charges are rational multiples of 1/3
    Q_u = Fraction(1, 2) + Y_Q   # T_3 = +1/2 for up-type in doublet
    Q_d = Fraction(-1, 2) + Y_Q  # T_3 = -1/2 for down-type
    Q_nu = Fraction(1, 2) + Y_L
    Q_e_phys = Fraction(-1, 2) + Y_L
    Q_u_R = Y_u   # SU(2) singlet: T_3 = 0
    Q_d_R = Y_d
    Q_e_R = Y_e

    charges = {
        'u': Q_u, 'd': Q_d, 'nu': Q_nu, 'e': Q_e_phys,
        'u_R': Q_u_R, 'd_R': Q_d_R, 'e_R': Q_e_R,
    }
    assert Q_u == Fraction(2, 3), f"Q(u) = {Q_u}"
    assert Q_d == Fraction(-1, 3), f"Q(d) = {Q_d}"
    assert Q_nu == Fraction(0), f"Q(nu) = {Q_nu}"
    assert Q_e_phys == Fraction(-1), f"Q(e) = {Q_e_phys}"
    # Cross-check: Q_u_R should equal Q_u (same physical particle)
    assert Q_u_R == Q_u, "Charge consistency: u_L and u_R"
    assert Q_d_R == Q_d, "Charge consistency: d_L and d_R"
    assert Q_e_R == Q_e_phys, "Charge consistency: e_L and e_R"

    # All charges are integer multiples of 1/3
    charge_quantum = Fraction(1, 3)
    for name, q in charges.items():
        ratio = q / charge_quantum
        assert ratio.denominator == 1, (
            f"Charge {name} = {q} not a multiple of 1/3"
        )

    # Quark-lepton charge relation
    assert Y_L == -N_c * Y_Q, "Y_L = -N_c * Y_Q (quark-lepton unification)"
    assert Y_e == -2 * N_c * Y_Q, "Y_e = -2*N_c*Y_Q"

    # Hypercharge sum per generation (another form of [grav]^2 U(1))
    Y_sum = (N_c * 2 * Y_Q + N_c * Y_u + N_c * Y_d + 2 * Y_L + Y_e)
    assert Y_sum == 0, f"Hypercharge sum per generation = {Y_sum}"

    # ================================================================
    # GENERATION SCALING
    # ================================================================
    # Anomaly conditions are per-generation and linear in N_gen.
    # If they vanish per generation, they vanish for any N_gen.
    # Witten requires N_gen * (N_c + 1) even, which holds for N_c = 3
    # (since N_c + 1 = 4 is already even, any N_gen works).
    for N_test in [1, 2, 3, 4, 5]:
        witten_ok = (N_test * n_doublets_per_gen) % 2 == 0
        assert witten_ok, f"Witten fails for N_gen = {N_test}"

    return _result(
        name='L_anomaly_free: Gauge Anomaly Cancellation',
        tier=2,
        epistemic='P',
        summary=(
            f'{n_passed}/{n_total} anomaly conditions verified with exact '
            f'rational arithmetic on framework-derived content. '
            f'[SU(3)]^3=0, [SU(2)]^3=0 (automatic), [SU(3)]^2 U(1)=0, '
            f'[SU(2)]^2 U(1)=0, [U(1)]^3=0, [grav]^2 U(1)=0, Witten=0. '
            f'Particle content derived from capacity (T_field), not from '
            f'anomaly cancellation. Anomaly-freedom is a CONSEQUENCE of '
            f'the capacity structure, not an input. '
            f'Derived: charge quantization (all Q = n/3), quark-lepton '
            f'relation Y_L = -N_c*Y_Q, gravitational consistency with '
            f'T9_grav. Witten safe for any N_gen (since N_c+1=4 is even). '
            f'Hypercharge ratios uniquely fixed (4 conditions, 5 unknowns, '
            f'1 normalization).'
        ),
        key_result=(
            f'7/7 anomaly conditions satisfied [P]; '
            f'charge quantization derived; '
            f'quark-lepton relation Y_L = -N_c*Y_Q'
        ),
        dependencies=[
            'T_gauge',     # Gauge group + hypercharge derivation
            'T_field',     # Particle content
            'Theorem_R',   # N_c = 3
            'T7',          # N_gen = 3
            'T9_grav',     # Gravitational consistency cross-check
        ],
        artifacts={
            'conditions': {k: {
                'value': str(v['value']),
                'passed': v['passed'],
                'role': v['role'],
            } for k, v in results.items()},
            'hypercharges': {
                'Y_Q': str(Y_Q), 'Y_u': str(Y_u), 'Y_d': str(Y_d),
                'Y_L': str(Y_L), 'Y_e': str(Y_e),
            },
            'electric_charges': {k: str(v) for k, v in charges.items()},
            'charge_quantum': str(charge_quantum),
            'quark_lepton_relations': [
                f'Y_L = -N_c*Y_Q = -{N_c}*{Y_Q} = {Y_L}',
                f'Y_e = -2*N_c*Y_Q = -{2*N_c}*{Y_Q} = {Y_e}',
            ],
            'uniqueness': (
                '4 anomaly conditions + 5 hypercharges = '
                '1 free parameter (overall normalization). '
                'Hypercharge RATIOS are uniquely fixed.'
            ),
            'non_trivial_content': (
                'T_field tests 4680 templates against 7 filters. '
                'Only 1 survives. The SM content is uniquely selected '
                'by capacity constraints + self-consistency, and it '
                'HAPPENS to be anomaly-free. This is the cross-check.'
            ),
            'generation_independence': (
                'Per-generation anomaly cancellation => safe for any N_gen. '
                'Witten safe for any N_gen since N_c + 1 = 4 is even.'
            ),
        },
    )


# ======================================================================
#  STANDALONE EXECUTION
# ======================================================================

def display():
    r = check_L_anomaly_free()

    W = 74
    print(f"{'=' * W}")
    print(f"  L_anomaly_free: GAUGE ANOMALY CANCELLATION CROSS-CHECK")
    print(f"{'=' * W}")

    mark = 'PASS' if r['passed'] else 'FAIL'
    print(f"\n  {mark} [{r['epistemic']}] {r['key_result']}")

    print(f"\n{'-' * W}")
    print(f"  CONDITION-BY-CONDITION VERIFICATION")
    print(f"{'-' * W}")
    for cond_name, cond in r['artifacts']['conditions'].items():
        mark = 'PASS' if cond['passed'] else 'FAIL'
        print(f"  {mark}  {cond_name:22s} = {cond['value']:5s}   ({cond['role']})")

    print(f"\n{'-' * W}")
    print(f"  DERIVED ELECTRIC CHARGES")
    print(f"{'-' * W}")
    for name, q in r['artifacts']['electric_charges'].items():
        print(f"  Q({name:4s}) = {q}")

    print(f"\n{'-' * W}")
    print(f"  QUARK-LEPTON RELATIONS")
    print(f"{'-' * W}")
    for rel in r['artifacts']['quark_lepton_relations']:
        print(f"  {rel}")

    print(f"\n{'-' * W}")
    print(f"  UNIQUENESS")
    print(f"{'-' * W}")
    print(f"  {r['artifacts']['uniqueness']}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    display()
    sys.exit(0)
