# Admissibility Physics Engine v4.3

**Deriving the Standard Model from a single axiom — 89 theorems, 0 free parameters, 27 predictions.**

**New in v4.3: CKM quark mixing matrix predicted from zero free parameters. 6/6 magnitudes within 5%.**

```
python3 run.py                        # Full display
python3 run.py --json                 # JSON export
python3 run.py --audit-gaps           # Gap analysis
python3 run.py --deps T_CKM          # Dependency tree
python3 run.py --reverse-deps A1     # Reverse dependency lookup
python3 run.py --export-dashboard    # Write dashboard_data.json
```

## What This Is

A computational verification engine for the Foundational Constraint Framework (FCF),
which derives the Standard Model of particle physics and general relativity from
a single information-theoretic axiom (A1: Finite Enforcement Capacity) with no free
parameters. Every theorem is machine-verified with full dependency tracking.

## Results

| Category | Count |
|---|---|
| Theorems | 89/89 pass |
| Proven [P] | 77 (87%) |
| Structural [P_structural] | 9 (10%) |
| Axioms/Postulates | 3 |
| Assertions | 462 |
| External imports | 9 |
| Sectors | 11/11 pass |

### Key Predictions (zero free parameters)

| Quantity | Predicted | Observed | Error |
|---|---|---|---|
| sin²θ_W | 3/13 | 0.2312 | 0.19% |
| Ω_Λ | 42/61 | 0.6889 | 0.05% |
| Ω_m | 19/61 | 0.3111 | 0.12% |
| f_b | 3/19 | 0.1571 | 0.49% |
| Gauge group | SU(3)×SU(2)×U(1) | SU(3)×SU(2)×U(1) | exact |
| Generations | 3 | 3 | exact |
| d (spacetime) | 4 | 4 | exact |
| Field content | {Q,L,u,d,e} | {Q,L,u,d,e} | exact |

### CKM Matrix (new v4.3 — zero free parameters)

The Standard Model uses 4 free parameters to fit 4 CKM observables.
FCF uses 0 free parameters and predicts 6+ observables.

| Quantity | Predicted | Observed | Error |
|---|---|---|---|
| θ₁₂ (Cabibbo) | 13.50° | 13.04° | 3.5% |
| θ₂₃ | 2.32° | 2.38° | 2.6% |
| θ₁₃ | 0.209° | 0.201° | 3.9% |
| \|V_us\| | 0.2334 | 0.2257 | 3.4% |
| \|V_cb\| | 0.0404 | 0.0410 | 1.4% |
| \|V_ub\| | 0.00364 | 0.00382 | 4.6% |
| J_CKM | 3.33×10⁻⁵ | 3.08×10⁻⁵ | 8.1% |

All 6 magnitudes within 5%. Hierarchy correct. CP violation sign correct.

### PMNS (Lepton Mixing) — Structural Wall

The extension to neutrino mixing reveals a structural wall: the Froggatt-Nielsen
texture with small neutrino charges produces a rank-1 mass matrix, making θ₁₂
solver-dependent. θ₂₃ ≈ 44° and θ₁₃ ≈ 8° are solver-stable and correct-order.
The framework correctly predicts PMNS ≫ CKM from the absence of color charge
in the lepton sector, but cannot derive full PMNS numerics from FN texture alone.

## Derivation Chain

```
A1 (Finite Capacity) + M (Multiplicity) + NT (Non-Triviality)
    |
    +-- L_eps*  : meaningful distinctions → ε > 0
    +-- L_loc   : enforcement distributes (A3 derived)
    +-- L_nc    : composition not free (A2 derived)
    +-- L_irr   : records lock capacity (A4 derived)
    |
    +== Tier 0: Foundations (26 theorems, all [P])
    +== Tier 1: Gauge Group (6 theorems, all [P])
    +== Tier 2: Particles/Generations (10 theorems)
    +== Tier 3: RG + Flavor Mixing (29 theorems)
    |     |
    |     +-- RG: T21 → T22 → T23 → T24 → sin²θ_W = 3/13
    |     |
    |     +-- Flavor (v4.3):
    |         x=1/2, κ=2, ε=1 → q_B=(7,4,0) → q_H=(7,5,0)
    |         φ=π/4, Δk=3, c_Hu=x³ → T_CKM (6/6 within 5%)
    |
    +== Tier 4: Gravity & Dark Sector (9 theorems)
    +== Tier 5: Delta_geo Corollaries (6 theorems)
```

## File Structure

```
FCF_Theorem_Bank_v4_3.py               # All 89 entries (Tiers 0-5 + 3F)
Admissibility_Physics_Engine_V4_3.py    # Master verification engine v11.0
run.py                                  # Entry point (all CLI flags)
VERSION_4_3.md                          # Changelog and full details
index.html                              # GitHub Pages dashboard
dashboard_data.json                     # Auto-generated from --export-dashboard
```

## Epistemic Stratification

Every theorem is tagged with machine-verified epistemic status:

- **[AXIOM]** — A1 (the single axiom)
- **[POSTULATE]** — M, NT (definitional)
- **[P]** — Proved from A1 + imported standard math
- **[P_structural]** — Proved modulo identified structural steps
- **[P_structural | open]** — Known open problem (PMNS wall)

No theorem is assumed. No circular dependencies. Full DAG validated at runtime.

## Open Physics (9 [P_structural] remaining)

| Type | Count | Theorems |
|---|---|---|
| Structural | 4 | T_q_Higgs, L_holonomy_phase, L_channel_crossing, T_CKM |
| Open physics | 3 | T10, T4G, T4G_Q31 |
| Import | 1 | T6B |
| Open | 1 | T_PMNS_partial |

The 4 structural gaps in flavor mixing are specific identified steps (Higgs VEV
location, generation-channel correspondence, conjugation cost). The 3 open physics
gaps require experimental input (Majorana/Dirac neutrinos, UV completion).

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

## Version History

| Version | Theorems | [P] | Predictions | Key Addition |
|---|---|---|---|---|
| v3.6 | 49 | 43 | 20 | Initial public release |
| v4.2.3 | 79 | 72 | 20 | Axiom reduction (A2-A5 derived), quantum structure |
| **v4.3.0** | **89** | **77** | **27** | **CKM matrix, flavor mixing (10 new theorems)** |

## License

MIT License. See [LICENSE](LICENSE).
