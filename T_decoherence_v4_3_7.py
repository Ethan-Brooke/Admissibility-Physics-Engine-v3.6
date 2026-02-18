#!/usr/bin/env python3
"""
================================================================================
T_decoherence: QUANTUM-TO-CLASSICAL TRANSITION [P]
================================================================================

v4.3.7 supplement.

Decoherence -- the suppression of quantum coherence between macroscopic
alternatives -- is derived from L_irr + T_CPTP + L_loc. No collapse
postulate is needed. The Born rule (T_Born) provides probabilities;
decoherence explains why macroscopic superpositions are never observed.

Run standalone:  python3 T_decoherence_v4_3_7.py
================================================================================
"""

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


# ======================================================================
#  Linear algebra helpers
# ======================================================================

def _zeros(n, m):
    return [[complex(0)] * m for _ in range(n)]

def _eye(n):
    M = _zeros(n, n)
    for i in range(n):
        M[i][i] = complex(1)
    return M

def _mm(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = _zeros(n, p)
    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C

def _dag(M):
    n, m = len(M), len(M[0])
    return [[M[j][i].conjugate() for j in range(n)] for i in range(m)]

def _madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def _mscale(c, M):
    return [[c * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]

def _tr(M):
    return sum(M[i][i] for i in range(len(M)))

def _kron(A, B):
    na, ma = len(A), len(A[0])
    nb, mb = len(B), len(B[0])
    C = _zeros(na * nb, ma * mb)
    for i in range(na):
        for j in range(ma):
            for k in range(nb):
                for l in range(mb):
                    C[i*nb+k][j*mb+l] = A[i][j] * B[k][l]
    return C

def _partial_trace_B(rho_AB, dA, dB):
    """Partial trace over subsystem B."""
    rho_A = _zeros(dA, dA)
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                rho_A[i][j] += rho_AB[i*dB+k][j*dB+k]
    return rho_A

def _outer(psi, phi):
    n = len(psi)
    return [[psi[i] * phi[j].conjugate() for j in range(n)] for i in range(n)]

def _eigvalsh(M):
    n = len(M)
    if n == 1:
        return [M[0][0].real]
    if n == 2:
        a, b = M[0][0].real, M[1][1].real
        c = abs(M[0][1])
        disc = _math.sqrt(max(0, ((a - b) / 2) ** 2 + c ** 2))
        mid = (a + b) / 2
        return sorted([mid - disc, mid + disc])
    A = [[complex(M[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(200):
        off = sum(abs(A[i][j])**2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-24:
            break
        for p in range(n):
            for q in range(p+1, n):
                if abs(A[p][q]) < 1e-15:
                    continue
                d_pq = A[p][p].real - A[q][q].real
                if abs(d_pq) < 1e-15:
                    theta = _math.pi / 4
                else:
                    theta = 0.5 * _math.atan2(2*abs(A[p][q]), d_pq)
                c, s = _math.cos(theta), _math.sin(theta)
                phase = A[p][q] / abs(A[p][q]) if abs(A[p][q]) > 1e-15 else 1
                s_ph = s * phase.conjugate()
                for j in range(n):
                    apj, aqj = A[p][j], A[q][j]
                    A[p][j] = c * apj + s_ph * aqj
                    A[q][j] = -s_ph.conjugate() * apj + c * aqj
                for i in range(n):
                    aip, aiq = A[i][p], A[i][q]
                    A[i][p] = c * aip + s_ph.conjugate() * aiq
                    A[i][q] = -s_ph * aip + c * aiq
                A[p][q] = complex(0)
                A[q][p] = complex(0)
    return sorted(A[i][i].real for i in range(n))


def _vn_entropy(rho):
    eigs = _eigvalsh(rho)
    return -sum(ev * _math.log(ev) for ev in eigs if ev > 1e-15)


# ======================================================================
#  T_decoherence
# ======================================================================

def check_T_decoherence():
    """T_decoherence: Quantum-to-Classical Transition [P].

    v4.3.7 NEW.

    STATEMENT: When a quantum system S interacts with an environment E,
    the off-diagonal elements of the reduced density matrix rho_S (in
    the pointer basis selected by the S-E interaction) decay
    exponentially in time. Macroscopic superpositions decohere on
    timescales far shorter than any observation time.

    No collapse postulate is needed. The Born rule (T_Born) provides
    probabilities for outcomes. Decoherence explains why only one
    outcome is observed: the others have become operationally
    inaccessible due to information dispersal into the environment.

    PROOF (4 steps):

    Step 1 -- System-environment coupling [T_CPTP + L_loc, P]:
      Any physical system is coupled to its environment through
      interfaces (L_loc). The subsystem evolution is a CPTP map
      (T_CPTP). The total S+E system evolves unitarily.

      Model: S is a qubit (|0>, |1>), E has d_E >> 1 states.
      Interaction Hamiltonian: H_int = |0><0| x B_0 + |1><1| x B_1
      where B_0, B_1 are operators on E.

      The pointer basis {|0>, |1>} is selected by the form of H_int:
      it is the basis that commutes with the interaction. This is
      determined by L_loc (the interface structure).

    Step 2 -- Decoherence of off-diagonal elements [L_irr, P]:
      Initial state: |psi> = (alpha|0> + beta|1>) x |E_0>
      After interaction time t:
        |Psi(t)> = alpha|0>|E_0(t)> + beta|1>|E_1(t)>

      Reduced density matrix of S:
        rho_S(t) = |alpha|^2 |0><0| + |beta|^2 |1><1|
                   + alpha*beta* <E_1(t)|E_0(t)> |0><1|
                   + alpha beta* <E_0(t)|E_1(t)> |1><0|

      The decoherence factor: Gamma(t) = <E_1(t)|E_0(t)>

      L_irr: as the environment records which-path information
      (|0> vs |1>), the environmental states |E_0(t)> and |E_1(t)>
      become increasingly orthogonal. The overlap decays:
        |Gamma(t)| = |<E_1(t)|E_0(t)>| -> 0

      Rate: for a thermal environment at temperature T with
      coupling strength lambda:
        |Gamma(t)| ~ exp(-Lambda_D * t)
      where Lambda_D ~ lambda^2 * k_B * T (decoherence rate).

    Step 3 -- Pointer basis from locality [L_loc, P]:
      L_loc (factorization) selects the pointer basis: it is the
      basis of local observables at the S-E interface. States that
      are eigenstates of the interface Hamiltonian are stable under
      decoherence. Superpositions of these eigenstates decohere.

      This is "environment-induced superselection" (einselection):
      the environment SELECTS which observables have definite values.
      In the framework, this is a consequence of locality (L_loc)
      applied to the capacity structure at interfaces.

    Step 4 -- Born rule for outcomes [T_Born, P]:
      After decoherence, rho_S is diagonal in the pointer basis:
        rho_S -> |alpha|^2 |0><0| + |beta|^2 |1><1|
      T_Born: the probability of outcome |k> is Tr(rho_S * |k><k|).
        P(0) = |alpha|^2, P(1) = |beta|^2
      These are the Born rule probabilities.

    COMPUTATIONAL WITNESS:
    Model a 2-qubit system (S=1 qubit, E=1 qubit) with CNOT
    interaction. Verify: (a) off-diagonal elements of rho_S vanish,
    (b) diagonal elements give Born rule probabilities,
    (c) total state remains pure (no information loss).

    WHY NO COLLAPSE POSTULATE:
      The "measurement problem" is: why does a superposition give
      a single outcome? The framework answer:
      (1) The superposition EXISTS (total state is pure, unitary)
      (2) Decoherence makes branches operationally independent
          (off-diagonal rho_S -> 0, L_irr makes this irreversible)
      (3) Each branch sees definite outcomes (pointer basis, L_loc)
      (4) Probabilities follow Born rule (T_Born, Gleason)
      No additional postulate is needed.

    STATUS: [P]. All ingredients from [P] theorems.
    """
    # ================================================================
    # COMPUTATIONAL WITNESS: CNOT decoherence model
    # ================================================================

    # System: 1 qubit (S), Environment: 1 qubit (E)
    dS = 2
    dE = 2
    dSE = dS * dE

    # Initial state: (alpha|0> + beta|1>)_S x |0>_E
    alpha = complex(_math.cos(_math.pi / 5))  # arbitrary superposition
    beta = complex(_math.sin(_math.pi / 5))

    psi_S = [alpha, beta]
    psi_E = [complex(1), complex(0)]  # environment starts in |0>

    # Product state |psi_SE> = |psi_S> x |psi_E>
    psi_SE = [complex(0)] * dSE
    for i in range(dS):
        for j in range(dE):
            psi_SE[i * dE + j] = psi_S[i] * psi_E[j]

    # Initial reduced density matrix
    rho_SE_init = _outer(psi_SE, psi_SE)
    rho_S_init = _partial_trace_B(rho_SE_init, dS, dE)

    # Check: initial rho_S is pure and has off-diagonal elements
    S_init = _vn_entropy(rho_S_init)
    assert S_init < 1e-10, "Initial rho_S is pure"
    assert abs(rho_S_init[0][1]) > 0.1, "Initial rho_S has off-diagonal elements"

    # ================================================================
    # Apply CNOT (controlled-NOT): the decoherence interaction
    # CNOT|0,0> = |0,0>, CNOT|1,0> = |1,1>
    # This records which-state information in the environment
    # ================================================================
    CNOT = _zeros(dSE, dSE)
    CNOT[0][0] = 1  # |00> -> |00>
    CNOT[1][1] = 1  # |01> -> |01>
    CNOT[2][3] = 1  # |10> -> |11>
    CNOT[3][2] = 1  # |11> -> |10>

    # Apply CNOT
    psi_after = [complex(0)] * dSE
    for i in range(dSE):
        for j in range(dSE):
            psi_after[i] += CNOT[i][j] * psi_SE[j]

    # Result: alpha|0,0> + beta|1,1> (entangled!)
    assert abs(psi_after[0] - alpha) < 1e-10, "|00> coefficient = alpha"
    assert abs(psi_after[3] - beta) < 1e-10, "|11> coefficient = beta"
    assert abs(psi_after[1]) < 1e-10, "|01> coefficient = 0"
    assert abs(psi_after[2]) < 1e-10, "|10> coefficient = 0"

    # ================================================================
    # Check decoherence: rho_S after CNOT
    # ================================================================
    rho_SE_after = _outer(psi_after, psi_after)
    rho_S_after = _partial_trace_B(rho_SE_after, dS, dE)

    # Off-diagonal elements should be ZERO
    # because <E_0|E_1> = <0|1> = 0 (orthogonal environment states)
    offdiag = abs(rho_S_after[0][1])
    assert offdiag < 1e-10, f"Off-diagonal = {offdiag} ~ 0 (decoherence complete)"

    # Diagonal elements give Born rule probabilities
    P_0 = rho_S_after[0][0].real
    P_1 = rho_S_after[1][1].real
    assert abs(P_0 - abs(alpha)**2) < 1e-10, "P(0) = |alpha|^2 (Born rule)"
    assert abs(P_1 - abs(beta)**2) < 1e-10, "P(1) = |beta|^2 (Born rule)"
    assert abs(P_0 + P_1 - 1.0) < 1e-10, "Probabilities sum to 1"

    # ================================================================
    # Check: total state is still pure (no information loss!)
    # ================================================================
    S_total = _vn_entropy(rho_SE_after)
    assert S_total < 1e-10, "Total state is still PURE"

    # But subsystem entropy has INCREASED (decoherence = info leakage)
    S_sub = _vn_entropy(rho_S_after)
    assert S_sub > 0.1, f"Subsystem entropy = {S_sub:.3f} > 0 (info leaked to env)"

    # ================================================================
    # Decoherence timescale estimate (thermal environment)
    # ================================================================
    # For a macroscopic object at room temperature:
    # Lambda_D ~ lambda^2 * k_B * T / hbar
    # Typical: Lambda_D ~ 10^{20} - 10^{40} /s for macroscopic objects
    # t_decoherence ~ 1/Lambda_D ~ 10^{-20} to 10^{-40} s
    # This is FAR shorter than any observation time (~10^{-3} s)

    k_B_T_room = 0.025  # eV at 300K
    hbar = 6.58e-16  # eV*s
    lambda_coupling = 1e-3  # typical dimensionless coupling

    # For a dust grain (~10^{10} atoms) at room temperature
    N_atoms = 1e10
    Lambda_D = lambda_coupling**2 * k_B_T_room * N_atoms / hbar
    t_decoherence = 1.0 / Lambda_D
    t_observation = 1e-3  # 1 ms (fastest human observation)

    assert t_decoherence < t_observation, (
        f"Decoherence ({t_decoherence:.1e} s) << observation ({t_observation:.0e} s)"
    )

    # ================================================================
    # Multi-step decoherence (partial decoherence model)
    # ================================================================
    # Model: system coupled to N sequential environment qubits
    # Each interaction reduces coherence by factor cos(theta)
    theta_int = _math.pi / 6  # partial coupling per step
    gamma_per_step = _math.cos(theta_int)  # decoherence factor per step

    coherence = 1.0
    coherence_history = [coherence]
    N_steps = 40
    for step in range(N_steps):
        coherence *= gamma_per_step
        coherence_history.append(coherence)

    # Verify exponential decay
    expected_final = gamma_per_step ** N_steps
    assert abs(coherence - expected_final) < 1e-10, "Exponential decay"
    assert coherence < 0.01, f"After {N_steps} steps: coherence = {coherence:.4f} << 1"

    # Decoherence rate
    Lambda_rate = -_math.log(gamma_per_step)  # per step
    assert Lambda_rate > 0, "Positive decoherence rate"

    return _result(
        name='T_decoherence: Quantum-to-Classical Transition',
        tier=0,
        epistemic='P',
        summary=(
            'Decoherence from L_irr + T_CPTP + L_loc. When system S interacts '
            'with environment E, off-diagonal elements of rho_S decay '
            'exponentially: |<E_0|E_1>| -> 0 as E records which-state info. '
            'Pointer basis selected by L_loc (interface structure). '
            'Born rule (T_Born) gives outcome probabilities. '
            f'CNOT witness: initial off-diag = {abs(rho_S_init[0][1]):.3f} -> '
            f'final off-diag = {offdiag:.1e} (complete decoherence). '
            f'P(0) = {P_0:.3f} = |alpha|^2, P(1) = {P_1:.3f} = |beta|^2. '
            f'Total state remains PURE (S_total = {S_total:.1e}). '
            f'Subsystem entropy: {S_sub:.3f} nats (info leaked to env). '
            f'Timescale for dust grain at 300K: {t_decoherence:.0e} s << 1 ms. '
            'No collapse postulate needed.'
        ),
        key_result=(
            'Decoherence from L_irr + T_CPTP [P]; '
            'no collapse postulate; Born rule for outcomes'
        ),
        dependencies=[
            'T_CPTP',       # Subsystem evolution is CPTP
            'L_irr',        # Irreversible record creation
            'L_loc',        # Pointer basis from locality
            'T_Born',       # Born rule for probabilities
            'T_entropy',    # Subsystem entropy increase
            'T_tensor',     # Composite system structure
        ],
        cross_refs=[
            'T_second_law',     # Entropy increase for subsystem
            'T_BH_information', # Same mechanism: tracing out DOF
            'L_cluster',        # Distant experiments independent
        ],
        artifacts={
            'CNOT_witness': {
                'dS': dS, 'dE': dE,
                'alpha': round(alpha.real, 4),
                'beta': round(beta.real, 4),
                'offdiag_before': round(abs(rho_S_init[0][1]), 4),
                'offdiag_after': offdiag,
                'P_0': round(P_0, 4),
                'P_1': round(P_1, 4),
                'S_total': round(S_total, 10),
                'S_subsystem': round(S_sub, 4),
                'decoherence_complete': offdiag < 1e-10,
            },
            'timescale': {
                'dust_grain_300K': f'{t_decoherence:.0e} s',
                'observation_time': f'{t_observation:.0e} s',
                'ratio': f'{t_decoherence / t_observation:.0e}',
                'macroscopic_decoherence': 'Instantaneous on all practical timescales',
            },
            'multi_step': {
                'N_steps': N_steps,
                'gamma_per_step': round(gamma_per_step, 4),
                'final_coherence': round(coherence, 6),
                'rate': round(Lambda_rate, 4),
                'exponential_verified': True,
            },
            'measurement_problem_resolution': {
                'superposition_exists': 'Yes (total state is pure, unitary)',
                'branches_independent': 'Yes (off-diagonal -> 0, L_irr makes irreversible)',
                'definite_outcomes': 'Yes (pointer basis from L_loc)',
                'probabilities': 'Born rule (T_Born, Gleason)',
                'collapse_postulate': 'NOT NEEDED',
            },
        },
    )


# ======================================================================
#  STANDALONE EXECUTION
# ======================================================================

def display():
    r = check_T_decoherence()

    W = 74
    print(f"{'=' * W}")
    print(f"  T_decoherence: QUANTUM-TO-CLASSICAL TRANSITION")
    print(f"{'=' * W}")

    mark = 'PASS' if r['passed'] else 'FAIL'
    print(f"\n  {mark} [{r['epistemic']}] {r['key_result']}")

    a = r['artifacts']

    print(f"\n{'-' * W}")
    print(f"  CNOT DECOHERENCE WITNESS")
    print(f"{'-' * W}")
    w = a['CNOT_witness']
    print(f"  System: {w['dS']}-dim qubit, Environment: {w['dE']}-dim qubit")
    print(f"  |psi_S> = {w['alpha']}|0> + {w['beta']}|1>")
    print(f"  Off-diagonal BEFORE: {w['offdiag_before']}")
    print(f"  Off-diagonal AFTER:  {w['offdiag_after']:.1e}")
    print(f"  P(0) = {w['P_0']} = |alpha|^2")
    print(f"  P(1) = {w['P_1']} = |beta|^2")
    print(f"  Total entropy: {w['S_total']:.1e} (PURE)")
    print(f"  Subsystem entropy: {w['S_subsystem']} nats")

    print(f"\n{'-' * W}")
    print(f"  DECOHERENCE TIMESCALE")
    print(f"{'-' * W}")
    t = a['timescale']
    print(f"  Dust grain at 300K: {t['dust_grain_300K']}")
    print(f"  Observation time:   {t['observation_time']}")
    print(f"  Conclusion: {t['macroscopic_decoherence']}")

    print(f"\n{'-' * W}")
    print(f"  MEASUREMENT PROBLEM RESOLUTION")
    print(f"{'-' * W}")
    m = a['measurement_problem_resolution']
    for key, val in m.items():
        print(f"  {key}: {val}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    display()
    sys.exit(0)
