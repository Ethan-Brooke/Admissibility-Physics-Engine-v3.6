# Admissibility Physics Engine — VERSION 3.6

**Date:** February 8, 2026
**Status:** 49/49 theorems pass | 43 [P] (88%), 6 [P_structural] (12%) | 20 predictions
**Gaps:** 43 closed, 3 reduced, 3 open physics

## Scorecard

| Metric | v3.5 | v3.6 |
|--------|------|------|
| [P] (proven) | 4 | **43** |
| [P_structural] | 41 | **6** |
| [C] / [C_structural] | 3 | **0** |
| Open physics | 5 | **3** |
| Predictions | 13 | **20** |
| Cosmological params | 0 quantitative | **5 (all ≤1σ)** |

## Changes (v3.5 → v3.6)

### Epistemic Upgrades
| Theorem | Change | Method |
|---------|--------|--------|
| T1, T2, T3 | [P_structural] → [P] | Imports proven math (KS, GNS, Skolem-Noether) |
| T4, T5 | [P_structural] → [P] | Anomaly cancellation is polynomial identity |
| T6 | [P_structural] → [P] | SU(5) embedding (pure Lie algebra) |
| T_field | [C] → [P] | Landau pole exclusion + CPT (no A5 needed) |
| T_gauge, T7, T4E, T4F | [P_structural] → [P] | Capacity budget is exact integer arithmetic |
| T_Higgs, T9 | [P_structural] → [P] | IVT + group theory |
| T7B, T8, T9_grav | [C_structural] → [P] | Pure math imports (Lovelock, HKM, Malament) |
| Gamma_* (6 theorems) | [P_structural]/[C_structural] → [P] | Bridge verification |
| All Tier 3 (T19-T27) | [P_structural] → [P] | Exact algebra, no approximations |
| T12 | [P_structural] → [P] | Regime gates R12.1/R12.2 closed |

### New Derivations
| Result | Formula | Agreement |
|--------|---------|-----------|
| f_b (baryon fraction) | 3/19 | 0.49% (Planck) |
| Ω_Λ (dark energy) | 42/61 | 0.05% (Planck) |
| Ω_m (total matter) | 19/61 | 0.12% (Planck) |
| Ω_b (baryons) | 3/61 | 0.37% (Planck) |
| Ω_DM (dark matter) | 16/61 | 0.61% (Planck) |

### New Predictions
- **Majorana neutrinos:** C_total = 61 (no ν_R) vs 64 (with ν_R). Testable via 0νββ experiments.
- **Boson-multiplet identity:** N_gauge + N_Higgs = N_multiplets (12 + 4 = 16). Self-consistency check.
- **Charge quantization:** Q = e/3 units from anomaly equations.
- **Neutral atoms:** Q_p + Q_e = 0 from anomaly cancellation.
- **Fractional quark charges:** Q_u = 2/3, Q_d = −1/3 from hypercharge solution.

### Physical Structure Corollaries (from T_field [P])
Field content {Q_L, u_R, d_R, L_L, e_R} uniquely derived from:
1. Gauge group SU(3)×SU(2)×U(1) [T_gauge, P]
2. Anomaly cancellation [T4, P]
3. Channel structure [T_channels, P]
4. Landau pole exclusion [A1: finite capacity]
5. CPT theorem [imported math]

No A5 (minimality) needed — physics alone selects the SM field content.

## Remaining P_structural (6 theorems)

| Theorem | Classification | Gap |
|---------|---------------|-----|
| T6B | scale_identification | β-formula derived; capacity↔momentum scale map structural |
| T10 | open_physics | κ proportionality constant (needs C_total in absolute units) |
| T11 | structural_step | Ω_Λ = 42/61 (structural: C_total = 61 identification) |
| T12E | structural_step | f_b = 3/19 (structural: C_visible = N_gen identification) |
| T4G | open_physics | Yukawa hierarchy (needs Majorana/Dirac determination) |
| T4G_Q31 | open_physics | Neutrino mass bound (follows from T4G) |

## All 20 Predictions

| # | Quantity | Predicted | Observed | Error |
|---|----------|-----------|----------|-------|
| 1 | sin²θ_W | 3/13 ≈ 0.2308 | 0.2312 | 0.19% |
| 2 | Gauge group | SU(3)×SU(2)×U(1) | SU(3)×SU(2)×U(1) | exact |
| 3 | Generations | 3 | 3 | exact |
| 4 | Spacetime dim | 4 | 4 | exact |
| 5 | Higgs exists | Yes (massive scalar) | Yes (125 GeV) | exact |
| 6 | DM exists | Yes (geometric) | Yes (Ω_DM ≈ 0.26) | exact |
| 7 | Λ > 0 | Yes (residual) | Yes | exact |
| 8 | Ω_Λ | 42/61 ≈ 0.6885 | 0.6889 | 0.05% |
| 9 | Ω_m | 19/61 ≈ 0.3115 | 0.3111 | 0.12% |
| 10 | Ω_b | 3/61 ≈ 0.04918 | 0.0490 | 0.37% |
| 11 | Ω_DM | 16/61 ≈ 0.2623 | 0.2607 | 0.61% |
| 12 | f_b | 3/19 ≈ 0.15789 | 0.1571 | 0.49% |
| 13 | Field content | {Q,L,u,d,e} | {Q,L,u,d,e} | exact |
| 14 | Q_u | 2/3 | 2/3 | exact |
| 15 | Q_e | −1 | −1 | exact |
| 16 | Q_ν | 0 | 0 | exact |
| 17 | Neutral atoms | Q_p + Q_e = 0 | |Q_p+Q_e| < 10⁻²¹ | exact |
| 18 | Neutrino type | Majorana | Unknown (testable) | — |
| 19 | Boson-multiplet | 12+4=16 | 12+4=16 | exact |
| 20 | N_gen consistency | N_c²+6 = 5×N_gen | 9+6=15=5×3 | exact |

## File Manifest

| File | Purpose |
|------|---------|
| `Admissibility_Physics_Engine_V3_6.py` | Master engine (49 theorems + predictions + audit) |
| `Admissibility_Physics_Theorems_V3_6.py` | Tier 0–3 theorem checks (34 checks) |
| `Admissibility_Physics_Gravity_V3_6.py` | Tier 4–5 gravity + dark sector (15 checks) |
| `theorem_0_canonical_v4.py` | T0 axiom witness module |
| `dashboard_data.json` | JSON export for dashboards |
| `Admissibility_Physics_Dashboard_V3_6.html` | Status dashboard (standalone HTML) |
| `Admissibility_Physics_Energy_Budget_V3_6.html` | Energy budget visualization |
| `VERSION_3_6.md` | This file |
| `README.md` | Repository README |
| `OBSERVATION_beta_cosmology.md` | β-function / cosmology observation (not yet theorem) |
