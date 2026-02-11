# Admissibility Physics Engine v3.8

**Deriving the Standard Model from 1 axiom — 60 theorems, 0 free parameters, 20 predictions.**

The framework operates on a single axiom (A1: Finite Capacity) plus two definitional postulates (M: Multiplicity, NT: Non-Triviality). Locality (A3) and irreversibility (A4) are themselves derived as lemmas L_loc and L_irr.

```bash
python3 fcf_theorem_bank.py                     # Full theorem bank (60 checks)
python3 enforcement_crystal_v3.py                # Crystal dependency analysis
python3 run_dashboard_export.py                  # Export dashboard_data.json
```

## What This Is

A computational verification engine for the Foundational Constraint Framework (FCF),
which derives the Standard Model of particle physics and general relativity from
information-theoretic axioms with no free parameters.

## Axiom Structure

| Element | Type | Statement |
|---------|------|-----------|
| **A1** | Axiom | Enforcement capacity is finite |
| M | Postulate | \|D\| ≥ 2 (multiplicity) |
| NT | Postulate | \|interfaces\| ≥ 2 (non-triviality) |
| A3 | Derived (L_loc) | Independent sectors factor (locality) |
| A4 | Derived (L_irr) | Record-creation is irreversible |
| L_nc | Derived | Non-closure under composition |

## Results

| Category | Count |
|----------|-------|
| Theorems | 60/60 pass |
| Proven [P] | 41 (68%) |
| Structural [P_structural] | 19 (32%) |
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

## Enforcement Crystal (v3)

The theorem dependency graph — the "enforcement crystal" — is auto-extracted from the theorem bank:

| Metric | 3-Axiom Mode | 1-Axiom Mode |
|--------|-------------|-------------|
| Nodes | 61 (3 ax + 58 derived) | 63 (3 ax + 60 derived) |
| Edges | 189 | 195 |
| Paths to sin²θ_W | 3,181 | 8,971 |
| Width-1 waists | depths 8, 17, 19 | depths 1, 4, 11, 20, 22 |
| Max depth | 19 | 22 |

Three structural bottlenecks in the 3-axiom crystal:
- **T_gauge** (depth 8): Gauge group selection — SU(3)×SU(2)×U(1) forced
- **T9_grav** (depth 17): Einstein field equations assembled
- **T12E** (depth 19): Baryon fraction f_b = 3/19

Axiom attribution for sin²θ_W: A1 46.7% · A3 42.1% · A4 11.1%

## File Structure

```
fcf_theorem_bank.py                  # Master theorem bank (60 checks, source of truth)
enforcement_crystal_v3.py            # Auto-extracting crystal analysis
run_dashboard_export.py              # Dashboard data generator
index.html                           # Interactive dashboard (GitHub Pages)
dashboard_data.json                  # Auto-generated dashboard data
Admissibility_Physics_Gravity_V3_6.py  # Gravity sector module
.github/workflows/update_dashboard.yml  # CI: auto-update dashboard_data.json
VERSION_3_8.md                       # Changelog
```

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

## Epistemic Stratification

Every theorem is tagged:

- **[P]** — Proven from axioms + imported math theorems
- **[P_structural]** — Proven modulo one identified structural step

No theorem is assumed. No circular dependencies. Full dependency DAG validated at runtime.

## Auto-Extraction Pipeline

The enforcement crystal (v3) auto-extracts all dependency data by:
1. Running every `check_*()` function in the theorem bank
2. Reading the `dependencies` field from each result
3. Cleaning annotated strings to canonical node IDs
4. Building the DAG and running 12 graph analyses

This eliminates hardcoded dependency maps — the crystal always reflects the current theorem bank.

## License

MIT. See LICENSE.
