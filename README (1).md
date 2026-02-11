# Admissibility Physics

**A constraint-first framework for deriving known physics from one axiom.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.admissibility_physics.svg)](https://zenodo.org/communities/admissibility_physics/)

---

## The Core Idea

Physical law is what survives finite enforceability. A distinction is physically meaningful if and only if the universe commits finite resources to enforce it. Start from this single constraint — *finite capacity* — and derive everything else.

No free parameters. No fitting. The framework outputs rational numbers from constraint logic and compares them to experiment after the fact.

## What It Derives

From one axiom (A1: Finite Capacity) plus two definitional postulates (M: Multiplicity, NT: Non-Triviality):

| Domain | Result | Status |
|--------|--------|--------|
| **Quantum mechanics** | Hilbert space, Born rule, CPTP dynamics, tensor products, von Neumann entropy | Derived [P] |
| **Gauge group** | SU(3)×SU(2)×U(1) — uniquely selected by capacity budget | Derived [P] |
| **Matter content** | {Q, L, u, d, e} — complete SM field content from anomaly cancellation + UV safety | Derived [P] |
| **Generations** | N_gen = 3 — from capacity saturation (N²+6 = 5N has unique positive integer solution) | Derived [P] |
| **Spacetime** | d = 4 with Lorentzian signature — from spinor-gauge compatibility + causality | Derived [P] |
| **Gravity** | Einstein field equations — from non-factorization of shared enforcement + Lovelock | Derived [P] |
| **Weinberg angle** | sin²θ_W = 3/13 ≈ 0.23077 (observed: 0.23122, error: 0.19%) | Derived [P] |
| **Dark energy** | Ω_Λ = 42/61 ≈ 0.6885 (observed: 0.6889, error: 0.05%) | Structural [P_s] |
| **Dark matter** | Ω_DM = 16/61 ≈ 0.2623 (observed: 0.2607, error: 0.61%) | Structural [P_s] |
| **Baryon fraction** | f_b = 3/19 ≈ 0.15789 (observed: 0.1571, error: 0.49%) | Structural [P_s] |
| **Charges** | Q_u = +2/3, Q_e = −1, Q_ν = 0, neutral atoms — all from anomaly cancellation | Derived [P] |
| **Higgs** | Massive scalar required by EW capacity pivot | Derived [P] |
| **Neutrinos** | Majorana predicted (C_total = 61 → no ν_R). Testable by 0νββ experiments. | Testable |

**Not yet derived:** Individual quark/lepton masses (Yukawa sector), absolute gravitational scale (Planck mass), CKM/PMNS matrix elements, strong CP phase.

## Axiom Hierarchy

```
Level 1:  One principle — Finite enforceability of distinction
Level 2:  Three orientational axioms (A, B, C) — pedagogical decomposition
Level 3:  Five operational axioms (A1–A5) — computational engine
          ↓ reduced by L_irr and L_loc to:
          One axiom (A1) + two postulates (M, NT)
```

The reduction chain:
- **A3** (locality) ← derived via L_loc from A1 + M + NT
- **L_nc** (non-closure) ← derived from A1 + A3
- **A4** (irreversibility) ← derived via L_irr from A1 + L_nc
- All 60 theorems follow from A1 alone.

## Repository Contents

### Papers (read in order, or start with Paper 13)

| # | Title | Layer |
|---|-------|-------|
| 0 | What Physics Permits | Orientation (non-technical) |
| 1 | The Enforceability of Distinction | SPINE |
| 2 | Finite Admissibility and the Failure of Global Description | STRUCTURE |
| 3 | Entropy, Time, and Accumulated Cost | LEDGERS |
| 4 | Admissibility Constraints and Structural Saturation | CONSTRAINTS |
| 5 | Quantum Structure from Finite Enforceability | QUANTUM |
| 6 | Dynamics and Geometry as Optimal Admissible Reallocation | DYNAMICS |
| 7 | A Minimal Quantum of Action | ACTION |
| 13 | The Minimal Admissibility Core v5.0 | Self-contained summary |
| 14 | The Enforcement Crystal v2 (Corrected) | Proof architecture analysis |
| 15 | Single-Axiom Reduction | L_irr + L_loc proofs |
| 59 | Executable Constraint Framework and Constants Map | Full technical reference |
| 60 | The Enforcement Crystal (original) | Superseded by Paper 14 |

### Code

| File | What it does |
|------|-------------|
| `Admissibility_Physics_Engine_V3_7.py` | Complete theorem bank. 60 theorems, 248 assertions, stdlib-only Python. Three modes: `display`, `audit-gaps`, `json`. |
| `L_irr_L_loc_single_axiom_reduction.py` | Witness worlds and countermodels for the single-axiom reduction. Runnable proof certificates. |
| `enforcement_crystal_v2.py` | Graph-theoretic analysis of the 63-node, 197-edge theorem dependency DAG. |
| `thedashboard_v38.html` | Interactive dashboard. Open in any browser. Self-contained except Three.js CDN. |

### Quick Start

```bash
# Run the theorem bank
python Admissibility_Physics_Engine_V3_7.py

# Run with gap audit
python Admissibility_Physics_Engine_V3_7.py --mode audit-gaps

# Export to JSON
python Admissibility_Physics_Engine_V3_7.py --mode json > output.json

# Run the single-axiom reduction proofs
python L_irr_L_loc_single_axiom_reduction.py

# View the dashboard
open thedashboard_v38.html
```

No external dependencies. Runs on Python 3.8+ with stdlib only.

## The Dashboard (v3.8)

The interactive dashboard provides:

- **Framework Status** — four animated radial gauges with expandable detail panels
- **Influence Treemap** — every theorem sized by downstream influence, with hover tooltips and structural analysis
- **Enforcement Crystal** — 3D WebGL visualization of the 63-node proof dependency graph (drag to rotate, hover to trace)
- **Cosmic Energy Budget** — the capacity ledger decomposition Ω_Λ + Ω_DM + Ω_b = 42/61 + 16/61 + 3/61 = 1
- **Complete Constants Map** — all 48 TOE parameters with derivation status, click-to-expand proof chains
- **Particle Content & Gravity** — SM field content and Einstein equation derivation chains
- **Quantum Foundations** — interactive derivation tree from A1 to Born rule
- **Featured Proof** — step-by-step Weinberg angle derivation
- **Proof River** — information flow visualization showing how one axiom becomes all of known physics
- **Audit Trail** — A01–A50 transparency log with severity and closure status

## Epistemic Status System

Every result carries an explicit tag:

| Tag | Meaning | Count |
|-----|---------|-------|
| **[P]** | Proved from axioms, all gates closed | 54 |
| **[P_structural]** | Structurally derived, identified bridge to close | 6 |
| **[testable]** | Awaiting experiment (e.g., Majorana neutrinos) | 2 |
| **[open]** | Not yet derived (Yukawa sector, absolute scale) | 14 |

## Falsifiability

The framework is maximally falsifiable. Every leaf prediction traces back to the axiom chain — a single failed prediction indicts everything above it. Key falsification targets:

- Discovery of a 4th-generation fermion at any mass
- Detection of additional gauge bosons not from SSB
- sin²θ_W deviating from 3/13 beyond measurement error
- Ω_Λ varying with redshift (quintessence)
- Non-detection of 0νββ at full experimental sensitivity (tests Majorana prediction)
- Detection of DM particles at direct detection experiments (DM = capacity overhead, not a particle)
- Detection of extra GW polarization modes or massive graviton

## How to Engage

This is an open program. Meaningful contributions include:

- Attempting to falsify specific derivations or consistency identities
- Proposing alternative regime assumptions and testing admissibility
- Extending the engine to additional physical sectors
- Challenging the enforceability principle itself with counterexamples
- Translating results into alternative mathematical formalisms

Contributions that demonstrate failure modes are as valuable as those that extend the framework.

## Links

- **Zenodo community:** https://zenodo.org/communities/admissibility_physics/
- **GitHub:** https://github.com/Ethan-Brooke/Admissibility-Physics-Engine-v3.6
- **Contact:** Via Zenodo community or GitHub issues

---

*Admissibility Physics: Canonical Release — February 2026*
