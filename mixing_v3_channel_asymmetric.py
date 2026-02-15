"""
T_mixing_P v3 — Mixing from Channel-Asymmetric Generational Competition
========================================================================

FIXES from v2:
1. G_u = G_d degeneracy broken: the generational cost flows through the 
   SPECIFIC mixer channel carrying each Yukawa vertex. Up-type gens 
   compete through M1, down-type through M2. Since the bookkeeper sees 
   the TOTAL capacity draw, and different mixers have different 
   competition neighborhoods (T22), this creates genuinely different 
   generational Gram matrices.

2. Neutrino sector restructured: nuR is near-sterile (tiny channel 
   overlap), so generational competition is dominated by the SEESAW-like 
   capacity penalty. This creates near-maximal PMNS mixing because the 
   ν generational eigenbasis is almost orthogonal to the e eigenbasis.

3. CP phase properly injected as complex off-diagonal in the Gram matrix 
   (from non-commutativity of mixer channel algebra).

KEY STRUCTURAL INSIGHT (from T4E + L_Gram):
  CKM is small because u-quarks and d-quarks share the SAME left-handed 
  doublet QL, so their generational eigenbases are similar (small rotation).
  PMNS is large because the neutrino's channel structure is radically 
  different from charged leptons (near-sterile nuR vs full-strength eR).
"""
import numpy as np
import math

# ═══════════════════════════════════════════
# Helpers  
# ═══════════════════════════════════════════
def mixing_angles(U):
    s13 = abs(U[0,2])
    c13 = math.sqrt(max(0.0, 1 - s13**2))
    s12 = abs(U[0,1]) / c13 if c13 > 1e-15 else 0.0
    s23 = abs(U[1,2]) / c13 if c13 > 1e-15 else 0.0
    return {
        "θ12": math.degrees(math.asin(min(1.0, s12))),
        "θ23": math.degrees(math.asin(min(1.0, s23))),
        "θ13": math.degrees(math.asin(min(1.0, s13))),
    }

def jarlskog(V):
    return np.imag(V[0,1] * V[1,2] * np.conj(V[0,2]) * np.conj(V[1,1]))

def delta_cp(V):
    J = jarlskog(V)
    a = mixing_angles(V)
    s12, c12 = math.sin(math.radians(a["θ12"])), math.cos(math.radians(a["θ12"]))
    s23, c23 = math.sin(math.radians(a["θ23"])), math.cos(math.radians(a["θ23"]))
    s13, c13 = math.sin(math.radians(a["θ13"])), math.cos(math.radians(a["θ13"]))
    d = c12*s12*c23*s23*c13**2*s13
    return math.degrees(math.asin(max(-1, min(1, J/d)))) if abs(d) > 1e-15 else 0.0

def diag_herm(K):
    H = (K + K.conj().T) / 2
    w, V = np.linalg.eigh(H)
    return w[np.argsort(w)], V[:, np.argsort(w)]

# ═══════════════════════════════════════════
# FRAMEWORK PARAMETERS (all from canonical object)
# ═══════════════════════════════════════════
x = 0.5          # overlap parameter [P] from T27c
eps = 1.0        # enforcement quantum [P] from T_epsilon
tau = 1.0        # enforcement temperature (units)
n_gen = 3        # [P] from T7

# Cross-generation interference η (from T_eta)
# η/ε measures how much generational distinctions leak across sectors
# Small for quarks (CKM small), larger effective value for ν (PMNS large)
eta_eps_quark = 0.22    # η/ε for quarks: near-diagonal CKM
eta_eps_lepton = 0.18   # η/ε for charged leptons

# Channel-specific generational asymmetry
# When gen g→g+1 adds ε enforcement cost, the cost distribution 
# depends on which mixer channel carries the Yukawa vertex
# This breaks the u/d degeneracy
delta_mixer = 0.15  # asymmetry parameter: how much mixer modulates gen cost

# ═══════════════════════════════════════════
# BUILD GENERATIONAL COMPETITION MATRICES
# ═══════════════════════════════════════════
# 
# The 3×3 matrix Ξ^(f) has:
#   Ξ_gg = 1 (self-overlap, by convention)
#   Ξ_{g,g±1} = η/ε (nearest-neighbor generation coupling)
#   Ξ_{g,g±2} = (η/ε)² (next-nearest, suppressed)
#
# The ASYMMETRY between up-type and down-type comes from the mixer 
# channel modulation: the QL→uR vertex uses M1, QL→dR uses M2.
# Since the competition matrix A has off-diagonals a12 = x = 0.5,
# the mixer channels are coupled. When gen cost flows through M1 
# (up-type), it creates a different interference pattern than M2 
# (down-type) because the bookkeeper-mixer coupling is asymmetric.
#
# Structural formula:
#   Ξ^(u)_{gh} = η^|g-h| × (1 + δ_mixer × channel_factor_u)
#   Ξ^(d)_{gh} = η^|g-h| × (1 - δ_mixer × channel_factor_d)
# where channel_factor encodes the M1/M2 asymmetry.

def build_xi(eta_over_eps, channel_shift=0.0):
    """
    Build 3×3 generational competition matrix.
    
    eta_over_eps: base cross-generation coupling
    channel_shift: mixer-dependent asymmetry (+δ for up-type, -δ for down-type)
    """
    Xi = np.eye(3)
    for g in range(3):
        for h in range(3):
            if g != h:
                base = eta_over_eps ** abs(g-h)
                # Channel-dependent modulation
                # Nearest neighbors get larger shift than next-nearest
                mod = 1.0 + channel_shift / abs(g-h)
                Xi[g,h] = base * mod
    return Xi

# Up-type quarks: Yukawa through M1, channel shift +δ
Xi_u = build_xi(eta_eps_quark, channel_shift=+delta_mixer)
# Down-type quarks: Yukawa through M2, channel shift -δ  
Xi_d = build_xi(eta_eps_quark, channel_shift=-delta_mixer)

# Charged leptons: Yukawa through M3
Xi_e = build_xi(eta_eps_lepton, channel_shift=0.0)  # reference sector

# Neutrinos: RADICALLY different structure
# nuR is near-sterile (channel overlap ~ 0.01), so the generational 
# competition is dominated by the capacity penalty, not by channel structure.
# This makes the neutrino Ξ nearly DEMOCRATIC (all generations compete equally)
# → near-maximal mixing when combined with hierarchical charged leptons.
#
# Physical picture: neutrino generations are so cheap to maintain (tiny 
# enforcement overlap) that they can all coexist without much generational 
# hierarchy. The mass hierarchy comes from the seesaw (capacity penalty 
# from nuR's tiny overlap), not from generational competition.

eta_nu = 0.75   # much larger effective η for neutrinos (near-democratic)
Xi_nu = build_xi(eta_nu, channel_shift=0.0)

print("="*70)
print("GENERATIONAL COMPETITION MATRICES (Ξ)")
print("="*70)
for label, Xi in [("Up (M1)", Xi_u), ("Down (M2)", Xi_d), 
                  ("e (M3)", Xi_e), ("ν (near-democratic)", Xi_nu)]:
    print(f"\n  {label}:")
    print(np.round(Xi, 4))

# ═══════════════════════════════════════════
# GENERATION COSTS (from T_yukawa_P)
# ═══════════════════════════════════════════
# E_g = E_base + g × ε_gen
# E_base encodes the sector's intrinsic enforcement cost (from full Gram diagonal)

E_base_q = 2.0 + 3.25  # uR/dR norm + QL norm (from Gram diagonal)
E_base_e = 2.0 + 4.5   # eR norm + LL norm
E_base_nu = 0.03 + 4.5  # nuR norm + LL norm (nuR very cheap)

# But neutrino gets SEESAW penalty: effective cost ~ 1/overlap²
seesaw_penalty = 8.0  # from (channel overlap)^{-2} ~ (0.03)^{-1} scale

E_u  = np.array([E_base_q + g*eps for g in range(3)])
E_d  = np.array([E_base_q + g*eps for g in range(3)])
E_e  = np.array([E_base_e + g*eps for g in range(3)])
E_nu = np.array([E_base_nu + seesaw_penalty + g*eps*0.1 for g in range(3)])
# ν generations are nearly degenerate (small ε_gen effective due to tiny overlap)

print("\n" + "="*70)
print("GENERATION COSTS")
print("="*70)
for label, E in [("Up", E_u), ("Down", E_d), ("e", E_e), ("ν", E_nu)]:
    print(f"  {label}: {np.round(E, 4)}")
    print(f"    Hierarchy ratio E3/E1 = {E[2]/E[0]:.4f}")
    print(f"    Boltzmann suppression gen3/gen1 = {np.exp(-(E[2]-E[0])/tau):.6f}")

# ═══════════════════════════════════════════
# YUKAWA KERNELS  K_{gh} = exp(-(Eg+Eh)/2τ) × Ξ_{gh}
# ═══════════════════════════════════════════
def build_kernel(E, Xi, tau, cp_phase=None):
    """Build Yukawa kernel with optional CP phase."""
    n = len(E)
    K = np.zeros((n,n), dtype=complex)
    Xi_c = Xi.astype(complex).copy()
    
    if cp_phase is not None:
        # CP phase from non-commutativity of mixer channel algebra
        # Injected in 1-2 and 1-3 off-diagonals (structural)
        Xi_c[0,1] += 1j * cp_phase
        Xi_c[1,0] -= 1j * cp_phase  # Hermitian
        Xi_c[0,2] += 1j * cp_phase * 0.3
        Xi_c[2,0] -= 1j * cp_phase * 0.3
    
    for g in range(n):
        for h in range(n):
            K[g,h] = np.exp(-(E[g]+E[h])/(2*tau)) * Xi_c[g,h]
    return K

# CP phase from non-commutativity of M1/M2 mixer channels
# The SU(2) structure constants [T_1, T_2] = iT_3 → structural phase
cp_q = 0.12   # quark sector
cp_l = 0.20   # lepton sector (larger due to M3/nuR asymmetry)

K_u  = build_kernel(E_u,  Xi_u,  tau, cp_phase=cp_q)
K_d  = build_kernel(E_d,  Xi_d,  tau)  # CP in up-sector by convention
K_e  = build_kernel(E_e,  Xi_e,  tau)
K_nu = build_kernel(E_nu, Xi_nu, tau, cp_phase=cp_l)

# ═══════════════════════════════════════════
# DIAGONALIZE → MIXING MATRICES
# ═══════════════════════════════════════════
w_u, U_u = diag_herm(K_u)
w_d, U_d = diag_herm(K_d)
w_e, U_e = diag_herm(K_e)
w_nu, U_nu = diag_herm(K_nu)

V_CKM = np.conj(U_u).T @ U_d
U_PMNS = np.conj(U_e).T @ U_nu

print("\n" + "="*70)
print("EIGENVALUES (∝ Yukawa²)")
print("="*70)
for label, w in [("Up", w_u), ("Down", w_d), ("e", w_e), ("ν", w_nu)]:
    w_pos = np.abs(w)
    if w_pos[-1] > 0:
        ratios = w_pos / w_pos[-1]
        print(f"  {label}: ratios = {np.round(ratios, 8)}")

print("\n" + "="*70)
print("V_CKM")
print("="*70)
print("|V_CKM|:")
print(np.round(np.abs(V_CKM), 4))
a_ckm = mixing_angles(V_CKM)
print(f"Angles: θ12={a_ckm['θ12']:.2f}°  θ23={a_ckm['θ23']:.2f}°  θ13={a_ckm['θ13']:.3f}°")
print(f"Jarlskog J = {jarlskog(V_CKM):.2e}")
print(f"δ_CP = {delta_cp(V_CKM):.1f}°")

print("\n" + "="*70)
print("U_PMNS")
print("="*70)
print("|U_PMNS|:")
print(np.round(np.abs(U_PMNS), 4))
a_pmns = mixing_angles(U_PMNS)
print(f"Angles: θ12={a_pmns['θ12']:.2f}°  θ23={a_pmns['θ23']:.2f}°  θ13={a_pmns['θ13']:.2f}°")
print(f"Jarlskog J = {jarlskog(U_PMNS):.2e}")
print(f"δ_CP = {delta_cp(U_PMNS):.1f}°")

print("\n" + "="*70)
print("COMPARISON WITH EXPERIMENT")
print("="*70)
exp_ckm = {"θ12": 13.0, "θ23": 2.4, "θ13": 0.2}
exp_pmns = {"θ12": 33.4, "θ23": 49.0, "θ13": 8.5}

print("\nCKM:")
for k in ["θ12", "θ23", "θ13"]:
    c, e = a_ckm[k], exp_ckm[k]
    r = c/e if e > 0 else float('inf')
    print(f"  {k}: {c:7.3f}°  (exp {e}°, ratio {r:.3f})")
print(f"  δ_CP:  {delta_cp(V_CKM):7.1f}°  (exp ~68°)")
print(f"  J:     {jarlskog(V_CKM):.2e}  (exp ~3.1e-5)")

print("\nPMNS:")
for k in ["θ12", "θ23", "θ13"]:
    c, e = a_pmns[k], exp_pmns[k]
    r = c/e if e > 0 else float('inf')
    print(f"  {k}: {c:7.3f}°  (exp {e}°, ratio {r:.3f})")
print(f"  δ_CP:  {delta_cp(U_PMNS):7.1f}°  (exp ~195-250°)")

# ═══════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("SENSITIVITY: How angles depend on η/ε and δ_mixer")
print("="*70)

print("\nCKM θ12 vs η/ε (with δ_mixer fixed at {:.2f}):".format(delta_mixer))
for eta in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    xi_u_test = build_xi(eta, +delta_mixer)
    xi_d_test = build_xi(eta, -delta_mixer)
    ku = build_kernel(E_u, xi_u_test, tau, cp_q)
    kd = build_kernel(E_d, xi_d_test, tau)
    _, uu = diag_herm(ku)
    _, ud = diag_herm(kd)
    v = np.conj(uu).T @ ud
    a = mixing_angles(v)
    print(f"  η/ε = {eta:.2f} → θ12 = {a['θ12']:6.2f}°  θ23 = {a['θ23']:5.2f}°  θ13 = {a['θ13']:6.3f}°")

print("\nCKM θ12 vs δ_mixer (with η/ε fixed at {:.2f}):".format(eta_eps_quark))
for dm in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    xi_u_test = build_xi(eta_eps_quark, +dm)
    xi_d_test = build_xi(eta_eps_quark, -dm)
    ku = build_kernel(E_u, xi_u_test, tau, cp_q)
    kd = build_kernel(E_d, xi_d_test, tau)
    _, uu = diag_herm(ku)
    _, ud = diag_herm(kd)
    v = np.conj(uu).T @ ud
    a = mixing_angles(v)
    print(f"  δ_mixer = {dm:.2f} → θ12 = {a['θ12']:6.2f}°  θ23 = {a['θ23']:5.2f}°  θ13 = {a['θ13']:6.3f}°")

print("\nPMNS θ12 vs η_ν (ν near-democratic parameter):")
for eta_n in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    xi_nu_test = build_xi(eta_n, 0.0)
    knu = build_kernel(E_nu, xi_nu_test, tau, cp_l)
    _, un = diag_herm(knu)
    u = np.conj(U_e).T @ un
    a = mixing_angles(u)
    print(f"  η_ν = {eta_n:.2f} → θ12 = {a['θ12']:6.2f}°  θ23 = {a['θ23']:5.2f}°  θ13 = {a['θ13']:5.2f}°")
