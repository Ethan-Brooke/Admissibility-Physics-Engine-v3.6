"""
══════════════════════════════════════════════════════════════════════
T_mixing_P / T_yukawa_P — Analysis Report
Mixing Matrices from the Canonical Object: Status & Next Steps  
══════════════════════════════════════════════════════════════════════

SUMMARY OF FINDINGS:
━━━━━━━━━━━━━━━━━━━

[✓] STRUCTURAL SUCCESSES (mechanism-level, independent of regime params):
    1. CKM is SMALL — because up and down quarks share QL doublet,
       so U_u ≈ U_d → V_CKM ≈ I + small corrections
    2. PMNS is LARGE — because nuR is near-sterile (channel overlap ~0.01),
       so ν generational structure radically differs from charged leptons
    3. CP violation EXISTS — from non-commutativity of mixer channels
       [T_1, T_2] = iT_3 in SU(2) → imaginary off-diagonal in Gram
    4. 3 generations with mass hierarchy — exp(-ε_gen/τ) from refinement ladder
    5. Yukawa hierarchy direction correct — 3rd gen heaviest, 1st lightest

[~] QUALITATIVELY CORRECT BUT QUANTITATIVELY OFF:
    1. CKM θ12 ≈ 13° achievable with η_q ~ 0.6 (Cabibbo angle)
       BUT θ23 and θ13 come out ~4x too large (wrong internal hierarchy)
    2. PMNS θ12 ~ 37-51° (need ~33°) — in right range but imprecise
    3. PMNS θ23 ~ 36-39° (need ~49°) — too small (not near-maximal)
    4. PMNS θ13 ~ 25-31° (need ~8.5°) — much too large

[✗] OPEN MATHEMATICAL PROBLEM:
    The CABIBBO HIERARCHY: θ12 >> θ23 >> θ13 with λ scaling
    (|V_us| ≈ λ, |V_cb| ≈ λ², |V_ub| ≈ λ³ where λ ≈ 0.22)
    
    Current mechanism cannot produce sufficient dynamic range between 
    generation pairs. Need ~30:1 ratio θ12/θ13, getting only ~3:1.

══════════════════════════════════════════════════════════════════════
DIAGNOSIS: Why the Cabibbo Hierarchy Fails and What's Needed
══════════════════════════════════════════════════════════════════════

The mixing angle between generations g and h scales (in perturbation theory) as:

    sin θ_{gh} ~ K_{gh}^off / |λ_g - λ_h|

where K_{gh}^off is the off-diagonal kernel element and λ_g - λ_h is the
eigenvalue splitting. For the Cabibbo hierarchy:

    θ_{12}/θ_{23} ~ (K^off_{12}/ΔE_{12}) / (K^off_{23}/ΔE_{23})

Since ΔE_{12} ≈ ΔE_{23} ≈ ε_gen in the uniform ladder, and K^off_{gh} 
involves both Ξ and the Boltzmann factor, we need:

    K^off_{12}/K^off_{23} >> 1  (large ratio for 1-2 vs 2-3 mixing)

CURRENT PROBLEM: With Ξ_{12} ~ η and Ξ_{23} ~ η × cost_suppression,
the ratio is only ~3:1 (from cost suppression with α~1). 
Need ~5:1 for CKM, and the OPPOSITE pattern for PMNS.

PROPOSED SOLUTION: Froggatt-Nielsen-like structure from channel algebra.

The key insight is that the Yukawa mass matrix should have the structure:

    M^(f)_{gh} ~ (ε_B/ε_M)^{n_{gh}} 

where ε_B/ε_M = x = 1/2 is the bookkeeper/mixer ratio and n_{gh} is
the "charge" difference between generations g and h in the relevant 
mixer channel. This gives:

    M^(u) ~ [[1,    x,   x²],       V_us ~ x = 0.5
              [x,    1,   x ],   →   V_cb ~ x² = 0.25  
              [x²,   x,   1 ]]       V_ub ~ x³ = 0.125

With x = 1/2, this is in the right ballpark for λ ≈ 0.22, though the
exact mapping x → λ requires the full Gram structure.

FOR PMNS: neutrino mass matrix is nearly DEMOCRATIC because nuR's tiny 
channel overlap makes all generational costs nearly equal. The small 
departures from democracy are controlled by the LL channel asymmetry
(LL[M3] = 1.5 vs LL[M1,M2] = 1.0), which creates:
    θ23 near-maximal (from M3 dominance)
    θ12 large (from near-democracy)
    θ13 smaller (from residual hierarchy)
══════════════════════════════════════════════════════════════════════
"""
import numpy as np
import math

def mixing_angles(U):
    s13 = abs(U[0,2])
    c13 = math.sqrt(max(0.0, 1 - s13**2))
    s12 = abs(U[0,1]) / c13 if c13 > 1e-15 else 0.0
    s23 = abs(U[1,2]) / c13 if c13 > 1e-15 else 0.0
    return {"θ12": math.degrees(math.asin(min(1.0, s12))),
            "θ23": math.degrees(math.asin(min(1.0, s23))),
            "θ13": math.degrees(math.asin(min(1.0, s13)))}

def jarlskog(V):
    return np.imag(V[0,1]*V[1,2]*np.conj(V[0,2])*np.conj(V[1,1]))

def delta_cp(V):
    J = jarlskog(V)
    a = mixing_angles(V)
    s12,c12 = math.sin(math.radians(a["θ12"])), math.cos(math.radians(a["θ12"]))
    s23,c23 = math.sin(math.radians(a["θ23"])), math.cos(math.radians(a["θ23"]))
    s13,c13 = math.sin(math.radians(a["θ13"])), math.cos(math.radians(a["θ13"]))
    d = c12*s12*c23*s23*c13**2*s13
    return math.degrees(math.asin(max(-1,min(1,J/d)))) if abs(d)>1e-15 else 0.0

def diag_herm(K):
    H = (K + K.conj().T)/2
    w, V = np.linalg.eigh(H)
    return w[np.argsort(w)], V[:,np.argsort(w)]

# ══════════════════════════════════════════════════════════════
# APPROACH B: Froggatt-Nielsen from x = 1/2
# ══════════════════════════════════════════════════════════════
# 
# The canonical object has x = 1/2 (bookkeeper/mixer overlap).
# Generation charges: gen g has effective FN charge q_g = g-1 (0,1,2).
# The Yukawa matrix element between left-gen g and right-gen h:
#   Y_{gh} ~ x^{|q_g^L - q_h^R|} × phase × normalization
#
# For up-type: both L and R have charges (0,1,2) but through DIFFERENT channels
# For down-type: same, but the channel assignment shifts the effective charges
#
# The u/d asymmetry comes from the fact that the FN charges are computed
# in different mixer channels (M1 for u, M2 for d), and the QL doublet
# has a generation-dependent projection onto each mixer.

print("="*70)
print("FROGGATT-NIELSEN STRUCTURE FROM x = 1/2")
print("="*70)

x = 0.5  # [P] from T27c
tau = 1.0
eps = 1.0

# FN charges per generation (from capacity ordering)
# gen 1: cheapest → charge 0
# gen 2: charge 1 
# gen 3: charge 2
q_L = np.array([0, 1, 2])  # left-handed charges (same for all L sectors)

# Right-handed charges DIFFER by flavor (different mixer channels)
# The charge depends on how many refinement steps the R sector needs
q_uR = np.array([0, 1, 2])     # up-type: standard ladder through M1
q_dR = np.array([0, 1, 2])     # down-type: standard ladder through M2

# The ASYMMETRY: QL's projection onto M1 vs M2 depends on generation
# because heavier gens have more bookkeeper overhead, reducing mixer projection
# Effective charge shift: δq(g) = g × (bookkeeper_fraction)
# This shifts the FN exponent differently for u vs d
delta_q_u = 0.3   # effective charge perturbation for up-type channel
delta_q_d = -0.1  # different for down-type channel

def build_FN_mass_matrix(q_L, q_R, x, delta_q=0.0, cp_phase=0.0):
    """
    Build Froggatt-Nielsen mass matrix:
    M_{gh} = x^{|q_L(g) + δq·g - q_R(h)|} × exp(iφ × something)
    
    The EFFECTIVE FN charges are shifted by the channel-dependent δq.
    """
    n = len(q_L)
    M = np.zeros((n,n), dtype=complex)
    for g in range(n):
        for h in range(n):
            # Effective charges with channel shift
            q_eff_L = q_L[g] + delta_q * g
            q_eff_R = q_R[h]
            # FN suppression
            exponent = abs(q_eff_L - q_eff_R)
            M[g,h] = x ** exponent
            # CP phase on off-diagonals
            if g != h:
                M[g,h] *= np.exp(1j * cp_phase * (g-h) / abs(g-h))
    return M

# Quark mass matrices
M_u = build_FN_mass_matrix(q_L, q_uR, x, delta_q=delta_q_u, cp_phase=0.12)
M_d = build_FN_mass_matrix(q_L, q_dR, x, delta_q=delta_q_d, cp_phase=0.0)

# Lepton mass matrices
# eR: standard ladder
q_eR = np.array([0, 1, 2])
M_e = build_FN_mass_matrix(q_L, q_eR, x, delta_q=0.0)

# nuR: NEAR-DEGENERATE because tiny channel overlap → all charges ~equal
# Effective FN parameter for ν is x_ν ≈ 1 - small correction
# (because nuR barely distinguishes generations)
x_nu = 0.85  # effective x for ν sector (close to 1 = democratic)
q_nuR = np.array([0, 0.3, 0.6])  # compressed charges (near-degenerate)
M_nu = build_FN_mass_matrix(q_L, q_nuR, x_nu, delta_q=0.0, cp_phase=0.25)

# Apply overall Yukawa scales (from Boltzmann: y_f ~ exp(-E_base/τ))
E_base = {"u": 4.05, "d": 4.05, "e": 5.0, "ν": 0.72}
for label, M, E in [("u", M_u, E_base["u"]), ("d", M_d, E_base["d"]),
                     ("e", M_e, E_base["e"]), ("ν", M_nu, E_base["ν"])]:
    M *= math.exp(-E/(2*tau))

print("\nMass matrices (magnitudes):")
for label, M in [("M_u", M_u), ("M_d", M_d), ("M_e", M_e), ("M_ν", M_nu)]:
    print(f"\n  |{label}|:")
    print(np.round(np.abs(M), 4))

# Diagonalize M†M (Yukawa squared)
for label, M in [("Up", M_u), ("Down", M_d), ("e", M_e), ("ν", M_nu)]:
    MM = M.conj().T @ M
    w = np.linalg.eigvalsh(MM)
    w = np.sort(np.abs(w))
    if w[-1] > 0:
        print(f"\n  {label} mass² ratios: {np.round(w/w[-1], 6)}")
        print(f"    sqrt ratios (mass): {np.round(np.sqrt(w/w[-1]), 6)}")

# CKM: V = U_uL† × U_dL
# where U_fL diagonalizes M_f M_f†
def left_unitary(M):
    """Get left unitary: M M† = U D U†"""
    MM = M @ M.conj().T
    w, V = np.linalg.eigh(MM)
    idx = np.argsort(w)
    return w[idx], V[:,idx]

_, U_uL = left_unitary(M_u)
_, U_dL = left_unitary(M_d)
_, U_eL = left_unitary(M_e)
_, U_nuL = left_unitary(M_nu)

V_CKM = np.conj(U_uL).T @ U_dL
U_PMNS = np.conj(U_eL).T @ U_nuL

print("\n" + "="*70)
print("V_CKM (FROGGATT-NIELSEN)")
print("="*70)
print("|V_CKM|:")
print(np.round(np.abs(V_CKM), 5))
a = mixing_angles(V_CKM)
print(f"\nθ12 = {a['θ12']:.2f}° (exp 13.0°)")
print(f"θ23 = {a['θ23']:.3f}° (exp 2.4°)")
print(f"θ13 = {a['θ13']:.4f}° (exp 0.20°)")
print(f"J = {jarlskog(V_CKM):.2e} (exp 3.1e-5)")
print(f"δ_CP = {delta_cp(V_CKM):.1f}° (exp ~68°)")

# Wolfenstein check: |V_us| ≈ λ, |V_cb| ≈ λ², |V_ub| ≈ λ³
lam = abs(V_CKM[0,1])
print(f"\nWolfenstein: λ = |V_us| = {lam:.4f}")
print(f"  |V_cb|/λ² = {abs(V_CKM[1,2])/lam**2:.3f} (should be ~1)")
print(f"  |V_ub|/λ³ = {abs(V_CKM[0,2])/lam**3:.3f} (should be ~1)")

print("\n" + "="*70)
print("U_PMNS (FROGGATT-NIELSEN)")
print("="*70)
print("|U_PMNS|:")
print(np.round(np.abs(U_PMNS), 4))
a = mixing_angles(U_PMNS)
print(f"\nθ12 = {a['θ12']:.2f}° (exp 33.4°)")
print(f"θ23 = {a['θ23']:.2f}° (exp 49.0°)")
print(f"θ13 = {a['θ13']:.2f}° (exp 8.5°)")
print(f"J = {jarlskog(U_PMNS):.4f}")
print(f"δ_CP = {delta_cp(U_PMNS):.1f}° (exp ~195-250°)")

# ══════════════════════════════════════════════════════════════
# SCAN OVER FN PARAMETERS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FN REGIME PARAMETER SCAN")
print("="*70)

best_score = 999
best_p = {}

exp_ckm  = {"θ12": 13.0, "θ23": 2.4, "θ13": 0.20}
exp_pmns = {"θ12": 33.4, "θ23": 49.0, "θ13": 8.5}

for dqu in np.arange(0.0, 0.8, 0.1):
  for dqd in np.arange(-0.4, 0.4, 0.1):
    for xn in [0.75, 0.80, 0.85, 0.90, 0.95]:
      for qn_scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        for cp in [0.05, 0.10, 0.15, 0.20]:
          mu = build_FN_mass_matrix(q_L, q_uR, x, dqu, cp)
          md = build_FN_mass_matrix(q_L, q_dR, x, dqd, 0.0)
          me = build_FN_mass_matrix(q_L, q_eR, x, 0.0, 0.0)
          qnr = np.array([0, qn_scale, 2*qn_scale])
          mn = build_FN_mass_matrix(q_L, qnr, xn, 0.0, 0.25)
          
          _, uuL = left_unitary(mu)
          _, udL = left_unitary(md)
          _, ueL = left_unitary(me)
          _, unL = left_unitary(mn)
          
          vc = np.conj(uuL).T @ udL
          up = np.conj(ueL).T @ unL
          
          ac = mixing_angles(vc)
          ap = mixing_angles(up)
          
          score = 0
          for k in ["θ12","θ23","θ13"]:
            cv, ev = ac[k], exp_ckm[k]
            if cv > 0.001: score += 2.0 * abs(math.log(cv/ev))
            else: score += 20
          for k in ["θ12","θ23","θ13"]:
            cv, ev = ap[k], exp_pmns[k]
            if cv > 0.01: score += 1.5 * abs(math.log(cv/ev))
            else: score += 20
          
          if score < best_score:
            best_score = score
            best_p = {"δq_u": dqu, "δq_d": dqd, "x_ν": xn, 
                      "q_ν_scale": qn_scale, "cp_q": cp}
            best_ac = ac
            best_ap = ap
            best_vc = vc
            best_up = up

print(f"\nBest FN parameters (score = {best_score:.3f}):")
for k,v in best_p.items():
    print(f"  {k} = {v}")

print(f"\nBest CKM:")
for k in ["θ12","θ23","θ13"]:
    c, e = best_ac[k], exp_ckm[k]
    print(f"  {k}: {c:8.3f}° (exp {e}°) ratio = {c/e:.3f}")
lam = abs(best_vc[0,1])
print(f"  λ = {lam:.4f}")
if lam > 0.01:
    print(f"  |V_cb|/λ² = {abs(best_vc[1,2])/lam**2:.3f}")
    print(f"  |V_ub|/λ³ = {abs(best_vc[0,2])/lam**3:.3f}")
print(f"  J = {jarlskog(best_vc):.2e}")
print(f"  δ_CP = {delta_cp(best_vc):.1f}°")

print(f"\nBest PMNS:")
for k in ["θ12","θ23","θ13"]:
    c, e = best_ap[k], exp_pmns[k]
    print(f"  {k}: {c:8.2f}° (exp {e}°) ratio = {c/e:.3f}")
print(f"  J = {jarlskog(best_up):.4f}")
print(f"  δ_CP = {delta_cp(best_up):.1f}°")

print(f"\n|V_CKM| best:")
print(np.round(np.abs(best_vc), 5))
print(f"\n|U_PMNS| best:")
print(np.round(np.abs(best_up), 4))

# ══════════════════════════════════════════════════════════════
# CHECK: Does x=1/2 naturally give the Wolfenstein parameter?
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("x = 1/2 AND THE WOLFENSTEIN PARAMETER")
print("="*70)
print(f"""
The Wolfenstein parameter λ ≈ 0.22 (= sin θ_Cabibbo).
The framework gives x = 1/2 from T27c.

If the FN charge difference between gen 1 and gen 2 is Δq, then:
  |V_us| = x^Δq

For Δq = 1: |V_us| = 0.5   (too large)
For Δq = 2: |V_us| = 0.25  (close to 0.22!)
For the best-fit δq_u = {best_p['δq_u']:.1f}:
  Effective Δq ≈ 1 + δq_u = {1 + best_p['δq_u']:.1f}
  x^(1+δq_u) = {x**(1+best_p['δq_u']):.4f}

Key relation: λ ≈ x^(1+δq) where δq is the channel asymmetry.
With x = 1/2 and δq ≈ 0.2, we get λ ≈ 0.5^1.2 ≈ 0.435... 
Still too large. Need δq ≈ 1.2 for λ ≈ 0.22.

STRUCTURAL PREDICTION: If the framework determines δq from the 
channel algebra (e.g., δq = x = 1/2 or δq = 1/m = 1/3), then
λ = x^(1+δq) would be a PARAMETER-FREE prediction.

x^(3/2) = 0.354  (δq = 1/2)
x^(4/3) = 0.397  (δq = 1/3)
x^2     = 0.250  (δq = 1, closest to λ=0.22)
x^(7/3) = 0.198  (δq = 4/3)

The Cabibbo angle λ = x² = 1/4 = 0.25 is within 15% of experiment!
This would make sin(θ_Cabibbo) = 1/4 a FRAMEWORK PREDICTION.
""")
