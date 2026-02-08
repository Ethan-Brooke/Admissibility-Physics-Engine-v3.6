#!/usr/bin/env python3
"""
================================================================================
THEOREM 0 — CANONICAL v4 (PRODUCTION RELEASE)
================================================================================
Foundational Constraint Framework: Logic of Finite Description

V4 Structure:
-------------
[D] = Deductive (general mathematical statement, paper proof)
[W] = Witnessed (finite-world executable certificate)
[C] = Countermodel (runnable finite world falsifying a claim/axiom-shape)

Theorem Components:
  T0.2a' [D] Chain-Increment Lower Bound → Capacity Non-Closure
         Pure deduction: if chain increments ≥ ε for n > C/ε elements, set inadmissible.
         No finite witness required/expected.
         
  T0.2b' [W] Interaction Blocks Additive Valuation
         Δ > 0 witnesses non-additivity of enforcement.
         
  T0.4'  [W] Operational Irreversibility via Path-Necessity
         BFS certificate: record-lock under constrained transitions.

Countermodels:
  CM_additive [C] → T0.2b' hypothesis fails (no Δ>0)
  CM_free_removal [C] → R4b/T0.4' fails (record freely removable)
================================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations, permutations
from collections import deque
from typing import Callable, Dict, Any, List, Set, FrozenSet, Tuple, Optional, Iterable

# =============================================================================
# LAYER 1: AXIOMATIC DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class Interface:
    name: str
    capacity: Fraction

    def __post_init__(self) -> None:
        assert self.capacity > 0, "Interface capacity must be positive."


@dataclass
class World:
    """
    A finite world W = (D, I, E):
      D  : finite set of distinctions
      I  : finite set of interfaces with finite capacities
      E  : enforcement functional E(S, Γ)
    """
    distinctions: Set[str]
    interfaces: List[Interface]
    E: Callable[[FrozenSet[str], str], Fraction]

    def enforcement(self, S: FrozenSet[str], gamma: Interface) -> Fraction:
        return self.E(S, gamma.name)

    def admissible(self, S: FrozenSet[str]) -> bool:
        """Admissible iff ∀Γ: E(S,Γ) ≤ C_Γ"""
        return all(self.enforcement(S, g) <= g.capacity for g in self.interfaces)

    def interaction(self, S1: FrozenSet[str], S2: FrozenSet[str], gamma: Interface) -> Fraction:
        """Δ(S1,S2) = E(S1∪S2) - E(S1) - E(S2). Superadditivity witness is Δ>0."""
        return (self.enforcement(S1 | S2, gamma) 
                - self.enforcement(S1, gamma) 
                - self.enforcement(S2, gamma))


# =============================================================================
# T0.2a' [D]: CHAIN-INCREMENT LOWER BOUND (Deductive - no witness required)
# =============================================================================

T02a_PRIME_STATEMENT = r"""
[D] T0.2a' (Capacity Non-Closure under Chain-Increment Lower Bound).
Fix an interface Γ with finite capacity C_Γ < ∞.
Fix ε > 0. Suppose there exists a finite set U ⊆ D and an ordering (d1,...,dn) of U such that:
  for all k=1..n:
     E_Γ({d1,...,dk}) - E_Γ({d1,...,d_{k-1}}) ≥ ε
and n > floor(C_Γ / ε).
Then U is not admissible at Γ, hence U ∉ Adm, so closure under union fails whenever
there exist admissible T1,T2 with T1∪T2 = U.
Proof: telescoping sum of increments gives E_Γ(U) ≥ nε > C_Γ.
"""


def CERT_T02a_prime(world: World, epsilon: Fraction, max_size: int = 10) -> Dict[str, Any]:
    """
    [Optional Certificate] Search for a chain-increment witness.
    
    NOTE: This is a supplementary check. T0.2a' is a deductive statement that holds
    in general; a finite witness may not exist within search bounds, and that's OK.
    """
    D = list(world.distinctions)
    limit = min(max_size, len(D))
    
    for g in world.interfaces:
        N_bound = int(g.capacity // epsilon)
        
        for k in range(N_bound + 1, limit + 1):  # need n > C/ε
            for tup in combinations(D, k):
                U = frozenset(tup)
                # Check if there exists an ordering with all increments ≥ ε
                for order in permutations(tup):
                    prev = frozenset()
                    increments = []
                    valid = True
                    for d in order:
                        curr = prev | {d}
                        inc = world.enforcement(curr, g) - world.enforcement(prev, g)
                        if inc < epsilon:
                            valid = False
                            break
                        increments.append((d, inc))
                        prev = curr
                    
                    if valid:
                        # Found a chain-increment witness
                        return {
                            "passed": True,
                            "witness": {
                                "interface": g.name,
                                "epsilon": str(epsilon),
                                "n": k,
                                "N_bound": N_bound,
                                "U": sorted(U),
                                "ordering": list(order),
                                "increments": [(d, str(v)) for d, v in increments],
                                "E_U": str(world.enforcement(U, g)),
                                "C": str(g.capacity)
                            }
                        }
    
    return {
        "passed": False,
        "note": "No T0.2a' certificate found within search limits."
    }


# =============================================================================
# T0.2b' [W]: INTERACTION BLOCKS ADDITIVE VALUATION
# =============================================================================

T02b_PRIME_STATEMENT = r"""
[D] T0.2b' (Interaction Blocks Additive Valuation / Context Independence).
If there exist disjoint S1,S2 and an interface Γ such that:
  Δ_Γ(S1,S2) := E_Γ(S1∪S2) - E_Γ(S1) - E_Γ(S2) > 0,
then there is no additive valuation v satisfying v(S1∪S2)=v(S1)+v(S2) that matches E_Γ
on {S1, S2, S1∪S2}. (Immediate.)
"""


def CERT_T02b_prime(world: World, max_size: int = 6) -> Dict[str, Any]:
    """
    [W] Witness for Δ > 0 (interaction surplus).
    
    This proves non-additivity of E at the witnessed triple.
    """
    D = list(world.distinctions)
    limit = min(max_size, len(D))
    
    # Enumerate non-empty sets up to limit
    sets: List[FrozenSet[str]] = []
    for k in range(1, limit + 1):
        for tup in combinations(D, k):
            sets.append(frozenset(tup))
    
    for g in world.interfaces:
        for S1 in sets:
            for S2 in sets:
                if not S1.isdisjoint(S2):
                    continue
                
                delta = world.interaction(S1, S2, g)
                
                if delta > 0:
                    E1 = world.enforcement(S1, g)
                    E2 = world.enforcement(S2, g)
                    EU = world.enforcement(S1 | S2, g)
                    
                    return {
                        "passed": True,
                        "witness": {
                            "interface": g.name,
                            "S1": sorted(S1),
                            "S2": sorted(S2),
                            "E(S1)": str(E1),
                            "E(S2)": str(E2),
                            "E(S1∪S2)": str(EU),
                            "delta": str(delta),
                            "note": "This witnesses Δ>0; therefore additive valuation matching E fails on this triple."
                        }
                    }
    
    return {
        "passed": False,
        "note": "No Δ>0 witness found within limits."
    }


# =============================================================================
# T0.4' [W]: OPERATIONAL IRREVERSIBILITY VIA PATH-NECESSITY (R4b)
# =============================================================================

T04_PRIME_STATEMENT = r"""
[D] T0.4' (Operational Irreversibility via Path-Necessity).
Given a TransitionSystem TS on admissible states, if there exists an admissible start state S0
containing a record r such that no admissible TS-path reaches any admissible state T with r ∉ T,
then r is operationally irreversible ("record-locked") relative to that transition semantics.
"""


@dataclass(frozen=True)
class TransitionSystem:
    world: World
    step: Callable[[FrozenSet[str]], Iterable[FrozenSet[str]]]

    def admissible(self, S: FrozenSet[str]) -> bool:
        return self.world.admissible(S)


def CERT_R4b_path_lock(
    ts: TransitionSystem,
    record: str,
    start_seeds: Optional[Iterable[FrozenSet[str]]] = None,
) -> Dict[str, Any]:
    """
    [W] BFS certificate for path-necessity of record-lock.
    
    Returns passed=True if ∃ admissible start S0 ∋ record such that NO admissible
    TS-path reaches any admissible state without record.
    """
    # Build closure of admissible states reachable from seeds
    admissible_states: Set[FrozenSet[str]] = set()
    q: deque = deque()
    
    # Seed with all admissible singletons if not provided
    if start_seeds is not None:
        seeds = list(start_seeds)
    else:
        seeds = []
        for d in ts.world.distinctions:
            S = frozenset({d})
            if ts.admissible(S):
                seeds.append(S)
    
    for s in seeds:
        if ts.admissible(s):
            admissible_states.add(s)
            q.append(s)
    
    # Expand reachable admissible states
    while q:
        s = q.popleft()
        for nxt in ts.step(s):
            nxt_f = frozenset(nxt)
            if ts.admissible(nxt_f) and nxt_f not in admissible_states:
                admissible_states.add(nxt_f)
                q.append(nxt_f)
    
    # For each admissible start containing record, BFS to check if record-free reachable
    for S0 in list(admissible_states):
        if record not in S0:
            continue
        
        visited = {S0}
        bfs: deque = deque([S0])
        reaches_record_free = False
        
        while bfs:
            s = bfs.popleft()
            if record not in s:
                reaches_record_free = True
                break
            for nxt in ts.step(s):
                nxt_f = frozenset(nxt)
                if nxt_f in admissible_states and nxt_f not in visited:
                    visited.add(nxt_f)
                    bfs.append(nxt_f)
        
        if not reaches_record_free:
            return {
                "passed": True,
                "witness": {
                    "record": record,
                    "start_state": sorted(S0),
                    "reachable_admissible_states": len(admissible_states),
                    "note": "No admissible TS-path reaches an admissible record-free state."
                }
            }
    
    return {
        "passed": False,
        "note": "Every admissible start containing record can reach an admissible record-free state."
    }


# =============================================================================
# WITNESS WORLD (T0.2b' and R4b/T0.4' certificates should pass)
# =============================================================================

def create_witness_world() -> Tuple[World, TransitionSystem, str]:
    """
    Witness world for V4:
      - T0.2b' [W]: Δ>0 at (a,b) on Γ1
      - R4b/T0.4' [W]: record 'r' locked under handoff semantics
    
    Note: T0.2a' is deductive; finite search may not find chain-increment witness.
    """
    D = {"a", "b", "c", "h", "r"}
    interfaces = [Interface("Γ1", Fraction(10)), Interface("Γ2", Fraction(10))]

    def E(S: FrozenSet[str], g: str) -> Fraction:
        cost = Fraction(0)
        if g == "Γ1":
            if "a" in S: cost += Fraction(2)
            if "b" in S: cost += Fraction(3)
            if "c" in S: cost += Fraction(2)
            if "r" in S: cost += Fraction(2)
            if "h" in S: cost += Fraction(9)  # {r,h} inadmissible (2+9>10)
            if "a" in S and "b" in S: cost += Fraction(4)  # Δ=4
        else:  # Γ2
            if "a" in S: cost += Fraction(2)
            if "b" in S: cost += Fraction(2)
            if "c" in S: cost += Fraction(3)
            if "r" in S: cost += Fraction(2)
            if "h" in S: cost += Fraction(9)
            if "b" in S and "c" in S: cost += Fraction(4)
        return cost

    world = World(distinctions=D, interfaces=interfaces, E=E)

    record = "r"
    free = {"a", "b", "c", "h"}

    def step(S: FrozenSet[str]) -> Iterable[FrozenSet[str]]:
        # Toggle free tokens
        for t in free:
            yield (S - {t}) if (t in S) else (S | {t})
        # Record removal requires handoff token (which makes state inadmissible)
        if (record in S) and ("h" in S):
            yield S - {record}
        # Record addition only when handoff absent
        if (record not in S) and ("h" not in S):
            yield S | {record}

    ts = TransitionSystem(world=world, step=step)
    return world, ts, record


# =============================================================================
# COUNTERMODELS
# =============================================================================

def countermodel_no_interaction_additive() -> World:
    """
    [C] Additive world: Δ = 0 always.
    T0.2b' hypothesis (Δ>0) should fail.
    """
    D = {f"d{i}" for i in range(6)}
    interfaces = [Interface("Γ", Fraction(10**6))]

    def E_additive(S: FrozenSet[str], _: str) -> Fraction:
        return Fraction(len(S))

    return World(distinctions=D, interfaces=interfaces, E=E_additive)


def countermodel_free_record_removal() -> Tuple[TransitionSystem, str]:
    """
    [C] Transition system where record is freely removable.
    R4b/T0.4' should fail (no path-lock).
    """
    D = {"a", "b", "r"}
    interfaces = [Interface("Γ", Fraction(100))]

    def E(S: FrozenSet[str], _: str) -> Fraction:
        return Fraction(0)  # everything admissible

    world = World(distinctions=D, interfaces=interfaces, E=E)

    def step(S: FrozenSet[str]) -> Iterable[FrozenSet[str]]:
        for t in D:
            yield (S - {t}) if (t in S) else (S | {t})

    return TransitionSystem(world=world, step=step), "r"


# =============================================================================
# V4 AUDIT RUN
# =============================================================================

def run_audit_v4(verbose: bool = True) -> Dict[str, Any]:
    """
    Execute all V4 certificates and countermodel checks.
    Returns structured report compatible with theorem_0_v4_report.json format.
    """
    world, ts, record = create_witness_world()

    # === Witness certificates ===
    cert_t02a = CERT_T02a_prime(world, epsilon=Fraction(1), max_size=8)
    cert_t02b = CERT_T02b_prime(world)
    cert_r4b = CERT_R4b_path_lock(ts, record)

    # === Countermodels ===
    cm_additive = countermodel_no_interaction_additive()
    cm_additive_res = CERT_T02b_prime(cm_additive)
    
    cm_ts, cm_r = countermodel_free_record_removal()
    cm_free_res = CERT_R4b_path_lock(cm_ts, cm_r)

    # === Build report ===
    report = {
        "meta": {
            "name": "theorem_0_canonical_v4",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "report": {
            "deductive_statements": {
                "T0.2a_prime": T02a_PRIME_STATEMENT,
                "T0.2b_prime": T02b_PRIME_STATEMENT,
                "T0.4_prime": T04_PRIME_STATEMENT
            },
            "witness_certificates": {
                "CERT_T0.2a_prime": cert_t02a,
                "CERT_T0.2b_prime": cert_t02b,
                "CERT_R4b_path_lock": cert_r4b
            },
            "countermodels": {
                "CM_no_interaction_additive__should_fail_T0.2b_prime_hypothesis": cm_additive_res,
                "CM_free_record_removal__should_fail_R4b_lock": cm_free_res
            }
        }
    }

    if verbose:
        print("=" * 72)
        print("THEOREM 0 — CANONICAL V4 AUDIT")
        print("=" * 72)
        
        print("\n[D] Deductive Statements:")
        print("    T0.2a' - Chain-Increment Lower Bound (general, no witness required)")
        print("    T0.2b' - Interaction Blocks Additive Valuation")
        print("    T0.4'  - Operational Irreversibility via Path-Necessity")
        
        print("\n[W] Witness Certificates:")
        print(f"    CERT_T0.2a_prime: {'PASSED' if cert_t02a['passed'] else 'NOT FOUND (expected for [D])'}")
        print(f"    CERT_T0.2b_prime: {'PASSED' if cert_t02b['passed'] else 'FAILED'}")
        if cert_t02b.get("witness"):
            w = cert_t02b["witness"]
            print(f"        S1={w['S1']}, S2={w['S2']}, Δ={w['delta']}")
        print(f"    CERT_R4b_path_lock: {'PASSED' if cert_r4b['passed'] else 'FAILED'}")
        if cert_r4b.get("witness"):
            w = cert_r4b["witness"]
            print(f"        start={w['start_state']}, record='{w['record']}'")
        
        print("\n[C] Countermodels (should fail their respective certificates):")
        print(f"    CM_additive (no Δ>0): {'CORRECTLY FAILS' if not cm_additive_res['passed'] else 'UNEXPECTED PASS'}")
        print(f"    CM_free_removal (no lock): {'CORRECTLY FAILS' if not cm_free_res['passed'] else 'UNEXPECTED PASS'}")
        
        print("\n" + "=" * 72)
        all_ok = (
            cert_t02b["passed"] and
            cert_r4b["passed"] and
            not cm_additive_res["passed"] and
            not cm_free_res["passed"]
        )
        print(f"V4 AUDIT STATUS: {'ALL CHECKS PASSED' if all_ok else 'ISSUES DETECTED'}")
        print("=" * 72)

    return report


def export_report_json(path: str = "theorem_0_v4_report.json") -> None:
    """Export audit report to JSON file."""
    report = run_audit_v4(verbose=False)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report exported to {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    report = run_audit_v4(verbose=True)
    
    # Also export JSON
    export_report_json("/home/claude/theorem_0_v4_report.json")
