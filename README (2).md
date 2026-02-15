# Foundational Constraint Framework — Verification Engine

**Deriving the Standard Model from 1 axiom — 79 entries, 0 free parameters, 20+ predictions.**

```
python3 run.py                       # Full display (scorecard + derivation chain)
python3 run.py --json                # JSON export for CI integration
python3 run.py --audit-gaps          # Detailed gap analysis for every [P_structural]
python3 run.py --deps T24            # Dependency tree for any theorem
python3 run.py --reverse-deps A1     # What depends on a given theorem
```

Or call the engine directly:

```
python3 Admissibility_Physics_Engine_V4_2.py
python3 Admissibility_Physics_Engine_V4_2.py --json
```

## What This Is

A computational verification engine for the Foundational Constraint Framework (FCF),
which derives the Standard Model of particle physics and general relativity from
a single axiom (A1: Finite Enforcement Capacity) with two definitional postulates
(M: Multiplicity, NT: Non-Triviality) and zero free parameters.

The former five axioms (A1–A5) have been reduced to one. A2–A5 are now derived
lemmas: L_nc (non-closure), L_loc (locality), L_irr (irreversibility), and
minimality from A1's bounded complexity.

## Results

| Category | Count |
|---|---|
| Total entries | 79/79 pass |
| Axiom | 1 |
| Postulates | 2 |
| Proved [P] | 72 (91%) |
| Structural [P_structural] | 4 (5%) |
| External imports | 9 theorems use imports |
| Assertions | 432 |

### Key Predictions (zero free parameters)

| Quantity | Predicted | Observed | Error |
|---|---|---|---|
| sin²θ_W | 3/13 = 0.23077 | 0.23122 | 0.19% |
| Ω_Λ | 42/61 = 0.6885 | 0.6889 | 0.05% |
| Ω_m | 19/61 = 0.3115 | 0.3111 | 0.12% |
| f_b | 3/19 = 0.1579 | 0.1571 | 0.49% |
| Gauge group | SU(3)×SU(2)×U(1) | SU(3)×SU(2)×U(1) | exact |
| Generations | 3 | 3 | exact |
| d (spacetime) | 4 | 4 | exact |
| θ_QCD | 0 | < 10⁻¹⁰ | exact |
| Field content | {Q,L,u,d,e} | {Q,L,u,d,e} | exact |

All cosmological parameters within 1σ of Planck 2018. All discrete predictions exact.

## File Structure

```
run.py                                # Entry point (delegates to engine)
Admissibility_Physics_Engine_V4_2.py  # Master engine: DAG validation, scorecard,
                                      #   gap audit, dependency tracing, JSON export
FCF_Theorem_Bank_v4_2.py              # All 79 entries (Tiers 0-5), stdlib only
```

### Legacy files (from earlier versions)

```
FCF_Theorem_Bank_v4_0_THE_FOUNDATION.py  # v4.0 theorem bank (68 entries)
theorems.py                              # v3.8 theorem bank (64 entries)
Admissibility_Physics_Gravity_V3_6.py    # v3.6 gravity tier (standalone)
Admissibility_Physics_BaryonFraction_V3_6.py
Admissibility_Physics_DarkMatter_V3_6.py
enforcement_crystal_v3.py
theorem_0_v5.py                          # T0 axiom witnesses (standalone)
t_hermitian_from_axioms.py               # T_Hermitian (standalone)
t_S0_interface_schema.py                 # T_S0 (standalone)
theorem_sin2theta_v1.py                  # sin²θ_W chain (standalone)
```

All standalone theorem files are subsumed by `FCF_Theorem_Bank_v4_2.py`.
They are retained for reference but not required to run the engine.

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

## Tier Architecture

| Tier | Name | Entries | [P] | [P_s] |
|---|---|---|---|---|
| 0 | Axiom-Level Foundations | 26 | 26 | 0 |
| 1 | Gauge Group Selection | 6 | 6 | 0 |
| 2 | Particle Content / Generations | 10 | 8 | 2 |
| 3 | Continuous Constants / RG | 19 | 18 | 1 |
| 4 | Gravity & Dark Sector | 9 | 8 | 1 |
| 5 | Delta_geo Structural Corollaries | 6 | 6 | 0 |

## Epistemic Stratification

Every entry is tagged:

- **[P]** — Proved from axioms + imported math theorems. Mechanically verifiable.
- **[P_structural]** — Proof complete modulo identified open physics problems.
- **[I]** — Uses external mathematical theorem (cited, verified applicable).

No theorem is assumed. No circular dependencies. Full dependency DAG validated
at runtime. 432 assertions checked on every run.

## Open Physics (4 remaining [P_structural])

| Theorem | Gap | To Close |
|---|---|---|
| T4G | Yukawa hierarchy: y_f ~ exp(-E_f/T) | Majorana vs Dirac (experimental) |
| T4G_Q31 | Neutrino mass bound | Follows from T4G |
| T6B | RG running: sin²θ_W from 3/8 → 0.2312 | 1-loop coefficients (QFT import) |
| T10 | Gravitational coupling κ ~ 1/C_* | UV completion |

All four involve absolute mass scales — the hardest open problem in fundamental physics.

## Progress: v3.4 → v4.2.3

- **Entries:** 58 → 79
- **[P]:** 39 → 72
- **[P_structural]:** 19 → 4
- **Axioms:** 5 (A1–A5) → 1 (A1 only)
- **15 theorems upgraded** from [P_structural] to [P]

## Version History

| Version | Date | Entries | [P] | [P_s] | Key Change |
|---|---|---|---|---|---|
| v3.4 | Jan 2026 | 58 | 39 | 19 | 5 axioms, distributed theorem files |
| v3.6 | Feb 2026 | 49* | 43 | 6 | Dashboard, standalone scripts |
| v3.8 | Feb 2026 | 64 | 60 | 4 | L_equip, sin²θ_W chain closed |
| v4.0 | Feb 2026 | 68 | 63 | 4 | Single-axiom reduction, unified bank |
| v4.2.3 | Feb 2026 | 79 | 72 | 4 | T_LV, M_Ω, P_exhaust, L_cost, T_canonical |

*v3.6 used a different counting convention (excluded axioms/postulates/sub-theorems).

## Links

- **Interactive Dashboard:** [admissibilityphysics.com](https://admissibilityphysics.com)
- **Zenodo:** [zenodo.org/communities/admissibility_physics](https://zenodo.org/communities/admissibility_physics/)
- **Papers:** Available on Zenodo (Papers 1–7, 13, 61, Crystal)

## License

MIT. See [LICENSE](LICENSE).
