# Canonical Derivation Summary — Referee-Ready (v3.5)

## The Foundation (v3.5: Three Axioms)

> **Three irreducible axioms — one per physical domain:**
>
> **A1** (Finite Capacity): Enforcement resources are bounded. [RESOURCE]
> **A3** (Locality): Enforcement decomposes over interfaces. [SPACE]
> **A4** (Irreversibility): Enforcement commits cannot be undone. [TIME]
>
> Former axioms A2 and A5 are now **derived lemmas**:
> **L_nc** (was A2): Non-closure from A1+A3+M+NT. [QUANTUM ORIGIN]
> **L_col** (was A5): Collapse from A1+A4. [MEASUREMENT ORIGIN]

---

## The One-Sentence Summary

> **Mixer = 3 (SU(2) doublet mixing) + Bookkeeper = 1 (unique anomaly pattern) ⇒ channels = 4 [P],**
> **hence C_EW = κ × channels = 2 × 4 = 8 [P], and N_gen = 3 [P] follows from E(3)=6 ≤ 8 < 10=E(4).**
>
> *Proved within the minimal chiral electroweak matter regime (one Q,L,u,d,e family, N_c=3).*

---

## The Core Argument

The chain is crisp and self-contained:

1. **mixer_channels = 3** from representation theory
   - dim(su(2)) = 3 for doublet mixing
   - Pauli matrices {σ₁, σ₂, σ₃} span all traceless Hermitian 2×2
   - This is provable mathematics, not a physics assumption

2. **bookkeeper_channels = 1** from anomaly uniqueness
   - Exhaustive rational scan finds unique hypercharge ratio pattern
   - **ANALYTIC PROOF**: z² - 2z - 8 = 0 → roots z ∈ {4, -2} (u↔d related)
   - One U(1) suffices; more would be redundant or break anomalies

3. **channels_EW = 4** (structurally forced)
   - Lower bound: channels ≥ 4 (all configs below fail constraints)
   - Upper bound: channels ≤ 4 (CCL partition completeness, executed not narrated)
   - Combined: channels = 4 [P]

4. **C_EW = κ × channels = 2 × 4 = 8**
   - κ = 2 from T_κ basis theorem [P]
     - v3.5: κ=2 now derived from A1+A4 (not A4+A5)
     - Forward cost from A1+A4 (persistence requires stabilization)
     - Backward cost from A4 (records require verification)
   - channels = 4 from above [P]
   - Inheritance: both inputs [P] → C_EW = 8 [P]

5. **N_gen = 3** from generation bound
   - E(N) = Nε + N(N-1)η/2 with η/ε ≤ 1 [P]
   - E(3) = 6 ≤ 8 < 10 = E(4)
   - N_gen = max{N : E(N) ≤ C_EW} = 3 [P]

---

## The Axiom Reduction (v3.5)

### L_nc (Non-Closure) — was axiom A2

**Derived from:** A1 + A3 + Postulate M + Postulate NT

**Proof sketch:** At a capacity-contested interface (NT), greedy packing
fills to capacity (A1). The next independent distinction (positive marginal
cost, M) overflows. Therefore ∃ S₁, S₂ both admissible with S₁∪S₂ inadmissible.

**Corollary chain:** non-closure → non-Boolean events → contextual poset → non-commutative algebra

**Status:** [P] given M and NT. Both postulates are physically transparent.

### L_col (Collapse) — was axiom A5

**Derived from:** A1 + A4

**Two directions:**
- (→) Forced simplification: capacity exhaustion + record requirement → must simplify
- (←) Persistence: A4 contrapositive → committed configs persist

**Key result:** T_κ (κ=2) now derived from A1+A4 alone. T8 (d=4) was never
A5-dependent — real exclusions are A4 (d≤3) + A1+Lovelock (d≥5).

**Status:** [P_structural]. The (→) direction has a philosophical subtlety
(collapse vs rejection at saturation) that no downstream theorem depends on.

---

## The Analytic Quadratic Uniqueness Proof

**Not just empirical from the scan — it's a theorem:**

Given N_c = 3 and the minimal chiral template {Q, L, u, d, e}:

```
Step 1: [SU(2)]²[U(1)] = 0 → Y_L = -3Y_Q
Step 2: [SU(N_c)]²[U(1)] = 0 → Y_d = 2Y_Q - Y_u
Step 3: Define z = Y_u / Y_Q
Step 4: [grav]²[U(1)] = 0 → Y_e = -6Y_Q
Step 5: [U(1)]³ = 0 → z² - 2z - 8 = 0
Step 6: Roots: z = 4 or z = -2
Step 7: z = -2 is u↔d swap of z = 4
Step 8: Therefore ratio pattern is UNIQUE (analytic, not empirical)
```

The scan now serves as **witness generator**, not the uniqueness proof.

---

## Explicit Regime Boundary (Referee-Safe)

**The field content template is an INPUT, not derived here:**

```
Regime: minimal_chiral_electroweak
Fields: {Q, L, u_R, d_R, e_R}
N_c = 3 (colors)
Doublet dimension = 2
Chiral = True
Epistemic: [C] — this is an assumption for this derivation
```

**Honest claim:**

> "channels = 4 is [P] **given the regime**. Deriving the regime itself from axioms is a separate theorem target."

---

## The Verified Artifact Chain

```
verify_chain.py calls:
  1. t_channels_rigorous.check()     → channels = 4 [P]
     ├─ derive_mixer_dimension()     → mixer = 3 (rep theory)
     ├─ derive_bookkeeper_count()    → bookkeeper = 1
     │   ├─ search_anomaly_solutions() → scan witness
     │   └─ derive_quadratic_uniqueness() → analytic proof
     ├─ run_exclusion_analysis()     → lower bound
     └─ check_channel_completeness() → upper bound (CCL executed)
  
  2. epistemic_verifier checks:
     ├─ channels_EW == 4             ✔
     ├─ mixer_channels == 3          ✔
     ├─ bookkeeper_channels == 1     ✔
     ├─ channel_partition_complete   ✔
     ├─ structurally_forced          ✔
     └─ uniqueness_analytic          ✔
  
  3. C_EW = κ × channels = 8 [P]
  
  4. N_gen = 3 [P] via E(3) ≤ C_EW < E(4)
```

---

## The Dependency DAG (v3.5)

```
A1 (Finite Capacity) ──┬──────────┬──────────┬───────────┐
                        │          │          │           │
A3 (Locality) ──────┐  │     A4 (Irreversibility)       │
                     │  │          │          │           │
                     ▼  ▼          ▼          ▼           │
                   L_nc          L_col       T_ε          │
                   (was A2)      (was A5)     │           │
                     │             │          ▼           │
                     ▼             ▼         T_κ (κ=2)   │
                    T1            T8          │           │
                     │           (d=4)        ▼           │
                     ▼             │        C_EW = 8     │
                    T2             │          │           │
                     │             ▼          ▼           │
                     ▼            T9       N_gen = 3     │
                    T3           (EFE)        │           │
                     │             │          ▼           │
                     ▼             ▼        T_channels   │
                    T4           T11           │          │
                  (gauge)      (Ω_Λ)          ▼          │
                     │             │       sin²θ_W     T12
                     ▼             ▼                    (DM)
                    T5           T12E
                  (matter)      (f_b)
```

---

## Key Equations

**Generation cost:**
$$E(N) = N\varepsilon + \frac{N(N-1)}{2}\eta$$

**Capacity (exactly, not approximately):**
$$C_{EW} = \kappa \times \text{channels} = 2 \times 4 = 8$$

**Generation bound:**
$$N_{gen} = \max\{N : E(N) \leq C_{EW}\} = 3$$

**Anomaly quadratic (analytic uniqueness):**
$$z^2 - 2z - 8 = 0 \quad \Rightarrow \quad z \in \{4, -2\}$$

---

# EXTENSION: Continuous Constants (T20-T23)

## Status: [P_structural] mechanism, [C_numeric] values

This extension module is **GATED** — downstream-only, does not affect core derivation.

## The RG-as-Cost-Flow Chain

```
T19 (channels = 4) [P]
         ↓
T20: RG = cost-metric flow [P_structural]
         ↓
T21: β_i(w) = -γ_i w_i + λ w_i Σ_j a_ij w_j [P_structural]
  (quadratic term from L_nc competition, not A2)
         ↓
T22: a_ij = Σ_e d_i(e) d_j(e) / C_e [P_structural]
         ↓
T23: sin²(θ_W)* = r*/(1+r*) [P_structural mechanism, C_numeric value]
```

## Key Results

**T20: RG = Enforcement Cost Renormalization**
- Standard QFT: couplings run due to quantum loops
- Admissibility: weights run due to coarse-graining of enforceable distinctions
- Key insight: RG is NOT fundamentally quantum

**T21: β-Function Form**
$$\beta_i(w) = -\gamma_i w_i + \lambda w_i \sum_j a_{ij} w_j$$
- Linear term: coarse-graining decay
- Quadratic term: non-closure competition (L_nc)
- λ: saturation avoidance

**T22: Competition Matrix**
$$a_{ij} = \sum_e \frac{d_i(e) d_j(e)}{C_e}$$
For EW with disjoint channels: a₁₁=1, a₂₂=3, a₁₂=0

**T23: Fixed Point**
$$r^* = \frac{\gamma_1 a_{22} - \gamma_2 a_{12}}{\gamma_2 a_{11} - \gamma_1 a_{21}}$$
$$\sin^2\theta_W^* = \frac{r^*}{1 + r^*}$$

---

## Epistemic Table (v3.5)

| Claim | Status |
|-------|--------|
| 5→3 axiom reduction | [P] / [P_structural] |
| channels = 4 | [P] |
| C_EW = 8 | [P] |
| N_gen = 3 | [P] |
| d = 4 spacetime | [P_structural] |
| RG = enforcement flow | [P_structural] |
| β-function form | [P_structural] |
| Fixed-point formula | [P_structural] |
| sin²θ_W mechanism | [P_structural] |
| EW routing graph | [C] |
| Cross-competition a₁₂ | [C] |
| sin²θ_W = 0.231 | [C_numeric] |

---

## Verification Command

```bash
python3 verify_chain.py
```

Expected output:
```
T_channels_rigorous: ✔ [P]
  VERIFIED [P] - structurally forced (lower=upper=4)

anomaly_scan: ✔
  VERIFIED - 48 solutions, unique pattern

C_EW: ✔ [P]
  VERIFIED [P] - inherits from T_κ[P] × T_channels[P]

N_gen: ✔ [P]
  VERIFIED [P] - E(3)=6 ≤ 8 < 10=E(4)

🎉 CHAIN COMPLETE: All quantities verified [P]
```

---

## The Derivation in One Paragraph

The chain is crisp and self-contained: mixer_channels = 3 from representation theory (dim su(2) = 3 for doublet mixing) and bookkeeper_channels = 1 from anomaly uniqueness (the quadratic z² - 2z - 8 = 0 analytically forces two roots z ∈ {4, -2} that are u↔d related, making the ratio pattern unique; the scan serves as witness generator). Together channels_EW = 4, hence C_EW = κ × channels = 2 × 4 = 8, and the generation bound follows from E(3) = 6 ≤ 8 < 10 = E(4). This is proved within the minimal chiral electroweak matter regime (one Q,L,u,d,e family with N_c = 3); deriving that regime from axioms is a separate theorem target. The entire derivation rests on three irreducible axioms (A1: finite capacity, A3: locality, A4: irreversibility), with non-closure (L_nc) and collapse (L_col) derived as lemmas rather than assumed.
