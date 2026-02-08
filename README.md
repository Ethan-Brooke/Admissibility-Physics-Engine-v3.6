# Admissibility Physics Engine v3.6

**Deriving the Standard Model from 5 axioms — 49 theorems, 0 free parameters, 20 predictions.**

```
python3 Admissibility_Physics_Engine_V3_6.py           # Full display
python3 Admissibility_Physics_Engine_V3_6.py --json     # JSON export
python3 Admissibility_Physics_Engine_V3_6.py --audit-gaps  # Gap analysis
python3 Admissibility_Physics_Engine_V3_6.py --export-dashboard  # Dashboard data
```

## What This Is

A computational verification engine for the Foundational Constraint Framework (FCF),
which derives the Standard Model of particle physics and general relativity from
five information-theoretic axioms (A1–A5) with no free parameters.

## Results

| Category | Count |
|----------|-------|
| Theorems | 49/49 pass |
| Proven [P] | 43 (88%) |
| Structural [P_structural] | 6 (12%) |
| Open physics | 3 |
| Predictions | 20 |

### Key Predictions

| Quantity | Predicted | Observed | Error |
|----------|-----------|----------|-------|
| sin²θ_W | 3/13 | 0.2312 | 0.19% |
| Ω_Λ | 42/61 | 0.6889 | 0.05% |
| Ω_m | 19/61 | 0.3111 | 0.12% |
| f_b | 3/19 | 0.1571 | 0.49% |
| Gauge group | SU(3)×SU(2)×U(1) | SU(3)×SU(2)×U(1) | exact |
| Generations | 3 | 3 | exact |
| d (spacetime) | 4 | 4 | exact |
| Field content | {Q,L,u,d,e} | {Q,L,u,d,e} | exact |

All 5 cosmological parameters within 1σ of Planck 2018. All discrete predictions exact.

## File Structure

```
Admissibility_Physics_Engine_V3_6.py      # Master engine (entry point)
Admissibility_Physics_Theorems_V3_6.py    # Tiers 0-3: gauge, particles, RG
Admissibility_Physics_Gravity_V3_6.py     # Tiers 4-5: gravity, dark sector
theorem_0_canonical_v4.py                 # T0 axiom witnesses
VERSION_3_6.md                            # Changelog and full prediction table
```

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

## Epistemic Stratification

Every theorem is tagged:
- **[P]** — Proven from axioms + imported math theorems
- **[P_structural]** — Proven modulo one identified structural step

No theorem is assumed. No circular dependencies. Full dependency DAG validated at runtime.

## Open Physics (3 remaining)

1. **T10** (gravitational coupling κ): needs C_total in absolute units
2. **T4G** (Yukawa hierarchy): needs Majorana vs Dirac determination
3. **T4G_Q31** (neutrino mass bound): follows from T4G

## License

Research use. See VERSION_3_6.md for full details.
