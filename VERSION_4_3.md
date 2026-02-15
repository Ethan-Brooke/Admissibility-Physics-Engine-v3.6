# VERSION 4.3.0 — Flavor Mixing

**Release date: February 2026**

## Summary

FCF v4.3.0 extends the framework into the flavor sector. The CKM quark mixing
matrix is now predicted from zero free parameters, with 6/6 magnitudes within 5%
of experiment. The Standard Model requires 4 free parameters to describe the same
observables.

## What's New

### Tier 3F: Flavor Mixing (10 new theorems)

| Theorem | Epistemic | Result |
|---|---|---|
| L_gen_path | **[P]** | Generation graph = path P₃ (Hasse diagram of total order) |
| T_capacity_ladder | **[P]** | FN charges q_B = (7, 4, 0) from quadratic capacity ladder |
| L_D2q | **[P]** | Universal second difference Δ²q = −ε (κ-independent) |
| L_H_curv | **[P]** | Higgs bump h = (0, 1, 0) — unique ℓ₁ solution on path graph |
| T_q_Higgs | [P_structural] | Combined q_H = (7, 5, 0) = q_B + h |
| L_holonomy_phase | [P_structural] | CP phase φ = π/4 from SU(2) orthogonal-generator holonomy |
| L_adjoint_sep | **[P]** | Mixer channel separation Δk = 3 = dim(adj SU(2)) |
| L_channel_crossing | [P_structural] | Up-type coupling c_Hu/c_Hd = x³ = 1/8 |
| **T_CKM** | [P_structural] | **CKM 6/6 within 5%, zero free parameters** |
| T_PMNS_partial | [P_structural \| open] | PMNS structural wall (θ₁₂ undetermined) |

### CKM Predictions

All inputs derived (x=1/2, κ=2, ε=1, q_B, q_H, φ=π/4, Δk=3, c_Hu=x³):

| Observable | FCF Prediction | Experiment (PDG 2024) | Error |
|---|---|---|---|
| θ₁₂ (Cabibbo angle) | 13.50° | 13.04° | +3.5% |
| θ₂₃ | 2.32° | 2.38° | −2.6% |
| θ₁₃ | 0.209° | 0.201° | +3.9% |
| \|V_us\| | 0.2334 | 0.2257 | +3.4% |
| \|V_cb\| | 0.0404 | 0.0410 | −1.4% |
| \|V_ub\| | 0.00364 | 0.00382 | −4.6% |
| J (Jarlskog) | 3.33×10⁻⁵ | 3.08×10⁻⁵ | +8.1% |

Known miss: δ_CP = 85° vs 68° (near-maximal CP violation correctly predicted,
magnitude overshoots by ~25%).

### PMNS Structural Wall

Extension to the lepton sector reveals a fundamental limitation of the
Froggatt-Nielsen mechanism for neutrinos:

- Neutrino FN charges q_ν = (0.5, 0, 0) give a rank-1 mass matrix
- Eigenvalue ratios ~10⁻¹⁶ (two eigenvalues numerically zero)
- θ₁₂ has 67° spread under 10⁻¹⁴ perturbations — **solver-dependent, not a prediction**
- θ₂₃ ≈ 44° and θ₁₃ ≈ 8° are solver-stable (correct order)

**Root cause**: Large PMNS angles require near-democratic neutrino mass matrix
→ small FN charges → rank deficiency. This is a fundamental tension between
the FN mechanism and large leptonic mixing, not a numerical issue.

**Conclusion**: The neutrino mass mechanism (Majorana/seesaw/Weinberg) likely
differs fundamentally from the quark/charged-lepton Yukawa structure. The
framework correctly identifies this distinction but cannot yet derive the PMNS.

Previous claim of "8/9 within 10%" was a numpy eigensolver artifact and has
been honestly retracted.

### Engine Updates

- **Engine version**: v10.0 → v11.0
- **Theorem bank**: v4.2.3 → v4.3.0
- **Entries**: 79 → 89 (+10)
- **[P] count**: 72 → 77 (+5)
- **[P_structural]**: 4 → 8 (+4)
- **[P_structural | open]**: 0 → 1 (+1: PMNS wall)
- **Assertions**: 380 → 462 (+82)
- **New sector**: `flavor_mixing` (10 theorems)
- **Predictions**: 20 → 27 (+7 CKM observables)
- **GAP_REGISTRY**: 4 → 9 entries (all new P_structural documented)
- **JSON export**: 6 → 13 prediction entries

### T4E Updated

CKM matrix elements (|V_us|, |V_cb|, |V_ub|) are no longer listed as "regime
parameters" in T4E. They are now **derived quantities** via T_CKM.

### Pure-Python Eigensolver

Added `_eigh` — a pure-Python Hermitian eigensystem solver (Jacobi method)
to keep the theorem bank free of numpy dependencies. Two-step approach:
1. Phase removal to make off-diagonal entries real
2. Real Givens rotation to diagonalize

Verified against numpy on both CKM (well-conditioned) and PMNS
(rank-deficient) test cases. The CKM results are solver-independent;
the PMNS θ₁₂ discrepancy between solvers is the structural wall itself.

## Counts

| Metric | v3.6 | v4.2.3 | v4.3.0 |
|---|---|---|---|
| Total entries | 49 | 79 | 89 |
| [P] proved | 43 | 72 | 77 |
| [P_structural] | 6 | 4 | 9 |
| Axioms | 5 (A1-A5) | 1 (A1) | 1 (A1) |
| Sectors | 7 | 10 | 11 |
| Predictions | 20 | 20 | 27 |
| Assertions | ~200 | ~380 | 462 |
| Gap registry items | 19 | 4 | 9 |

## Gap Registry (v4.3.0)

| Theorem | Type | Description |
|---|---|---|
| T4G | open_physics | Yukawa hierarchy (needs Majorana/Dirac) |
| T4G_Q31 | open_physics | Neutrino mass bound (follows T4G) |
| T6B | import | 1-loop β-coefficients from QFT |
| T10 | open_physics | Gravitational coupling (needs UV completion) |
| T_q_Higgs | structural | Higgs VEV location in M₂ channel |
| L_holonomy_phase | structural | Generation-channel correspondence |
| L_channel_crossing | structural | Conjugation cost (x vs x²) |
| T_CKM | structural | Inherits 3 P_structural dependencies |
| T_PMNS_partial | open | Rank-1 neutrino texture (FN wall) |

## Derivation Chain (Flavor Mixing Branch)

```
x = 1/2 (T27c [P]) + κ = 2 (T_kappa [P]) + ε = 1 (T_eta [P])
    |
    +-- Q(g) = gκ + g(g-1)ε/2  →  capacity ladder
    +-- q_B = Q(3) - Q(g) = (7, 4, 0)  [P]
    +-- Generation graph = path 1-2-3  (L_gen_path [P])
    +-- ℓ₁ minimization on path → h = (0, 1, 0)  (L_H_curv [P])
    +-- q_H = q_B + h = (7, 5, 0)  (T_q_Higgs [P_s])
    |
    +-- φ = π/4 from SU(2) holonomy  (L_holonomy_phase [P_s])
    +-- Δk = 3 = dim(adj SU(2))  (L_adjoint_sep [P])
    +-- c_Hu = x³ from channel crossings  (L_channel_crossing [P_s])
    |
    +== T_CKM: build M_u, M_d textures → diagonalize → V_CKM [P_s]
    |     6/6 within 5%, zero free parameters
    |
    +== T_PMNS_partial: extend to leptons → structural wall [open]
          θ₁₂ undetermined, θ₂₃ ~ 44°, θ₁₃ ~ 8° solver-stable
```

## File Changes

### New files
- `FCF_Theorem_Bank_v4_3.py` — Unified theorem bank (89 entries, ~7800 lines)
- `Admissibility_Physics_Engine_V4_3.py` — Master engine v11.0 (~900 lines)
- `VERSION_4_3.md` — This file

### Updated files
- `run.py` — Points to v4.3 engine
- `README.md` — Updated counts, added CKM section

### Legacy files (can be removed or kept as history)
- `Admissibility_Physics_Engine_V3_6.py`
- `Admissibility_Physics_Theorems_V3_6.py`
- `Admissibility_Physics_Gravity_V3_6.py`
- `Admissibility_Physics_BaryonFraction_V3_6.py`
- `Admissibility_Physics_DarkMatter_V3_6.py`
- `VERSION_3_5.md`, `VERSION_3_6.md`
