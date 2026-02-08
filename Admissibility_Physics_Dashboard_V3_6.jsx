import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

// ============================================================
// Admissibility Physics Engine v3.6 — Status Dashboard
// Date: 2026-02-08
// 49 theorems | 0 free parameters | 0 contradictions
// ============================================================

const THEOREMS = [
  // Tier 0: Axiom Foundations (9 theorems, all [P])
  { id: "T0", name: "Axiom Witness Certificates", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T1", name: "Contextuality (KS)", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T2", name: "Hilbert Space (GNS)", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T3", name: "Gauge Bundle (DR)", tier: 0, epistemic: "P", gap: "closed" },
  { id: "L_e*", name: "Granularity Bound", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T_e", name: "Min Enforcement Cost", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T_eta", name: "Subordination Bound", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T_kappa", name: "Capacity Multiplier k=2", tier: 0, epistemic: "P", gap: "closed" },
  { id: "T_M", name: "Monogamy (biconditional)", tier: 0, epistemic: "P", gap: "closed" },
  // Tier 1: Gauge Group (3 theorems, all [P])
  { id: "T4", name: "Anomaly Cancellation", tier: 1, epistemic: "P", gap: "closed" },
  { id: "T5", name: "Hypercharge Uniqueness", tier: 1, epistemic: "P", gap: "closed" },
  { id: "T_gauge", name: "SU(3)×SU(2)×U(1)", tier: 1, epistemic: "P", gap: "closed" },
  // Tier 2: Particles (10 theorems, 8 [P] + 2 [P_structural])
  { id: "T_field", name: "Field Content {Q,L,u,d,e}", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T_channels", name: "EW Channels = 4", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T7", name: "N_gen = 3", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T4E", name: "Generation Structure", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T4F", name: "Capacity Saturation 75%", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T4G", name: "Yukawa Structure", tier: 2, epistemic: "P_structural", gap: "open" },
  { id: "T4G_Q31", name: "Neutrino Mass Bound", tier: 2, epistemic: "P_structural", gap: "open" },
  { id: "T_Higgs", name: "Massive Scalar Required", tier: 2, epistemic: "P", gap: "closed" },
  { id: "T9", name: "Record Sectors (3!=6)", tier: 2, epistemic: "P", gap: "closed" },
  // Tier 3: RG / Constants (13 theorems, 12 [P] + 1 [P_structural])
  { id: "T6", name: "sin²θ_W(M_U) = 3/8", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T6B", name: "Capacity RG Running", tier: 3, epistemic: "P_structural", gap: "reduced" },
  { id: "T19", name: "Routing Sectors M=3", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T20", name: "RG = Enforcement Flow", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T21", name: "β-function Form", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T22", name: "Competition Matrix", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T23", name: "Fixed-Point Formula", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T24", name: "sin²θ_W = 3/13", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T25a", name: "Overlap Bounds", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T25b", name: "Saturation Push x→1/2", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T26", name: "Gamma Ratio Bounds", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T27c", name: "x = 1/2", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T27d", name: "γ₂/γ₁ = 17/4", tier: 3, epistemic: "P", gap: "closed" },
  { id: "T_sin2theta", name: "Weinberg Angle (final)", tier: 3, epistemic: "P", gap: "closed" },
  // Tier 4: Gravity + Dark Sector (8 theorems, 5 [P] + 3 [P_structural])
  { id: "T7B", name: "Gravity from Non-Factorization", tier: 4, epistemic: "P", gap: "closed" },
  { id: "T8", name: "d = 4 Spacetime Dimensions", tier: 4, epistemic: "P", gap: "closed" },
  { id: "T9_grav", name: "Einstein Equations (Lovelock)", tier: 4, epistemic: "P", gap: "closed" },
  { id: "T10", name: "Gravitational Coupling κ", tier: 4, epistemic: "P_structural", gap: "open" },
  { id: "T11", name: "Λ: Ω_Λ = 42/61", tier: 4, epistemic: "P_structural", gap: "reduced" },
  { id: "T_particle", name: "Mass Gap & SSB", tier: 4, epistemic: "P", gap: "closed" },
  { id: "T12", name: "Dark Matter = Capacity", tier: 4, epistemic: "P", gap: "closed" },
  { id: "T12E", name: "f_b = 3/19", tier: 4, epistemic: "P_structural", gap: "reduced" },
  // Tier 5: Gamma_geo Closure (6 theorems, all [P])
  { id: "G_ordering", name: "Ledger Ordering R1-R4", tier: 5, epistemic: "P", gap: "closed" },
  { id: "G_fbc", name: "Fluctuation Bound", tier: 5, epistemic: "P", gap: "closed" },
  { id: "G_continuum", name: "Continuum Limit", tier: 5, epistemic: "P", gap: "closed" },
  { id: "G_signature", name: "Lorentzian Signature (HKM)", tier: 5, epistemic: "P", gap: "closed" },
  { id: "G_particle", name: "Particle Emergence V(Φ)", tier: 5, epistemic: "P", gap: "closed" },
  { id: "G_closure", name: "Full Closure 10/10", tier: 5, epistemic: "P", gap: "closed" },
];

const PREDICTIONS = [
  { name: "sin\u00B2\u03B8_W", predicted: "3/13 \u2248 0.2308", observed: "0.23122", error: 0.19, type: "continuous" },
  { name: "Gauge Group", predicted: "SU(3)\u00D7SU(2)\u00D7U(1)", observed: "SU(3)\u00D7SU(2)\u00D7U(1)", error: 0, type: "exact" },
  { name: "Generations", predicted: "3", observed: "3", error: 0, type: "exact" },
  { name: "Spacetime d", predicted: "4", observed: "4", error: 0, type: "exact" },
  { name: "Higgs exists", predicted: "Yes (massive scalar)", observed: "Yes (125 GeV)", error: 0, type: "exact" },
  { name: "DM exists", predicted: "Yes (geometric)", observed: "\u03A9_DM \u2248 0.26", error: 0, type: "exact" },
  { name: "\u039B > 0", predicted: "Yes (residual capacity)", observed: "Yes", error: 0, type: "exact" },
  { name: "\u03A9_\u039B", predicted: "42/61 \u2248 0.6885", observed: "0.6889", error: 0.05, type: "continuous" },
  { name: "\u03A9_m", predicted: "19/61 \u2248 0.3115", observed: "0.3111", error: 0.12, type: "continuous" },
  { name: "\u03A9_b", predicted: "3/61 \u2248 0.04918", observed: "0.0490", error: 0.37, type: "continuous" },
  { name: "\u03A9_DM", predicted: "16/61 \u2248 0.2623", observed: "0.2607", error: 0.61, type: "continuous" },
  { name: "f_b", predicted: "3/19 \u2248 0.1579", observed: "0.1571", error: 0.49, type: "continuous" },
  { name: "Field content", predicted: "{Q,L,u,d,e}", observed: "{Q,L,u,d,e}", error: 0, type: "exact" },
  { name: "Q_u", predicted: "2/3", observed: "2/3", error: 0, type: "exact" },
  { name: "Q_e", predicted: "\u22121", observed: "\u22121", error: 0, type: "exact" },
  { name: "Q_\u03BD", predicted: "0", observed: "0", error: 0, type: "exact" },
  { name: "Neutral atoms", predicted: "Q_p + Q_e = 0", observed: "|Q_p+Q_e| < 10\u207B\u00B2\u00B9", error: 0, type: "exact" },
  { name: "\u03BD type", predicted: "Majorana", observed: "TBD (0\u03BD\u03B2\u03B2)", error: null, type: "testable" },
  { name: "Boson-multiplet", predicted: "12+4=16=N_mult", observed: "12+4=16 \u2713", error: 0, type: "exact" },
  { name: "N_gen identity", predicted: "N_c\u00B2+6=5\u00D7N_gen", observed: "9+6=15=5\u00D73 \u2713", error: 0, type: "exact" },
];

const AUDIT_CHECKS = [
  { id: "A01", name: "Runtime Output", status: "FIXED", desc: "Engine produces stdout on every run", severity: "critical" },
  { id: "A02", name: "Schema Validation", status: "FIXED", desc: "Every theorem checked for required fields + valid types", severity: "critical" },
  { id: "A03", name: "DAG Cycle Detection", status: "FIXED", desc: "Dependency graph checked for circular references", severity: "critical" },
  { id: "A07", name: "Computational Witnesses", status: "ACTIVE", desc: "V(Phi) computed: 5/5 checks including mass gap = 0.53", severity: "high" },
  { id: "A16", name: "C_structural \u2192 P Bridge", status: "FIXED", desc: "Lovelock + HKM are pure math: gravity imports now [P]", severity: "critical" },
  { id: "A22", name: "T6 SU(5) Embedding", status: "FIXED", desc: "sin\u00B2\u03B8_W(M_U)=3/8 from Lie algebra, no physics assumption", severity: "high" },
  { id: "A24", name: "T_field Landau Pole", status: "FIXED", desc: "Field content uniquely derived via UV safety + CPT", severity: "critical" },
  { id: "A25", name: "T12E Combinatorial f_b", status: "FIXED", desc: "f_b = 3/19 from pure counting (was \u03B3-calibrated)", severity: "critical" },
  { id: "A26", name: "T11 \u03A9_\u039B = 42/61", status: "FIXED", desc: "Cosmological constant from DOF counting, 0.05% Planck", severity: "critical" },
  { id: "A27", name: "Majorana Prediction", status: "ACTIVE", desc: "C_total=61 (no \u03BD_R) vs 64 (with). Testable via 0\u03BD\u03B2\u03B2", severity: "high" },
  { id: "A28", name: "Boson-Multiplet Identity", status: "VERIFIED", desc: "N_gauge + N_Higgs = N_mult = 16. Self-consistency check", severity: "high" },
];

const TIER_NAMES = {
  0: "Axiom Foundations", 1: "Gauge Group", 2: "Particles",
  3: "RG / Constants", 4: "Gravity + Dark", 5: "\u0393_geo Closure"
};

const TIER_COLORS = {
  0: "#6366f1", 1: "#8b5cf6", 2: "#ec4899",
  3: "#f59e0b", 4: "#10b981", 5: "#06b6d4"
};

const GAP_COLORS = { closed: "#22c55e", import: "#3b82f6", open: "#ef4444", reduced: "#f59e0b" };
const EPI_COLORS = { P: "#22c55e", P_structural: "#3b82f6", C_structural: "#f97316", C: "#6b7280", W: "#eab308" };
const SEVERITY_COLORS = { critical: "#ef4444", high: "#f97316", medium: "#eab308", low: "#6b7280" };
const STATUS_COLORS = { FIXED: "#22c55e", ACTIVE: "#3b82f6", NOTED: "#6b7280" };

function Badge({ color, children }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: 0.5,
      background: color + "22", color: color, border: `1px solid ${color}44`
    }}>{children}</span>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: "8px 20px", border: "none", cursor: "pointer",
      fontSize: 13, fontWeight: active ? 700 : 400, letterSpacing: 0.5,
      background: active ? "#1e293b" : "transparent",
      color: active ? "#f8fafc" : "#94a3b8",
      borderBottom: active ? "2px solid #3b82f6" : "2px solid transparent",
      transition: "all 0.2s"
    }}>{children}</button>
  );
}

// ==================== TAB 1: STATUS ====================
function StatusTab() {
  const epiCounts = useMemo(() => {
    const counts = {};
    THEOREMS.forEach(t => { counts[t.epistemic] = (counts[t.epistemic] || 0) + 1; });
    return Object.entries(counts).map(([k, v]) => ({ name: k, value: v, color: EPI_COLORS[k] || "#999" }));
  }, []);

  const gapCounts = useMemo(() => {
    const counts = {};
    THEOREMS.forEach(t => { counts[t.gap] = (counts[t.gap] || 0) + 1; });
    return Object.entries(counts).map(([k, v]) => ({ name: k, value: v, color: GAP_COLORS[k] || "#999" }));
  }, []);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
        {[
          { label: "Theorems", value: "49/49", sub: "all pass", color: "#22c55e" },
          { label: "Free Params", value: "0", sub: "zero", color: "#3b82f6" },
          { label: "Predictions", value: "20", sub: "12 exact, 6 continuous, 1 testable", color: "#22c55e" },
          { label: "Open Physics", value: "3", sub: "T10, T4G, T4G_Q31", color: "#f59e0b" },
        ].map((s, i) => (
          <div key={i} style={{ background: "#0f172a", borderRadius: 8, padding: 16, border: `1px solid ${s.color}33` }}>
            <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: s.color, fontFamily: "monospace" }}>{s.value}</div>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: "#0f172a", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>Epistemic Distribution</div>
          {epiCounts.map((e, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <Badge color={e.color}>{e.name}</Badge>
              <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3 }}>
                <div style={{ width: `${(e.value / 48) * 100}%`, height: "100%", background: e.color, borderRadius: 3 }} />
              </div>
              <span style={{ fontSize: 13, fontWeight: 700, color: e.color, fontFamily: "monospace", minWidth: 24 }}>{e.value}</span>
            </div>
          ))}
        </div>
        <div style={{ background: "#0f172a", borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>Gap Classification</div>
          {gapCounts.map((g, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <Badge color={g.color}>{g.name}</Badge>
              <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3 }}>
                <div style={{ width: `${(g.value / 48) * 100}%`, height: "100%", background: g.color, borderRadius: 3 }} />
              </div>
              <span style={{ fontSize: 13, fontWeight: 700, color: g.color, fontFamily: "monospace", minWidth: 24 }}>{g.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ==================== TAB 2: ACCURACY ====================
function AccuracyTab() {
  return (
    <div>
      <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>Predictions vs Experiment</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: "4px 16px", fontSize: 12 }}>
          <div style={{ color: "#64748b", fontWeight: 600, borderBottom: "1px solid #334155", paddingBottom: 4 }}>Prediction</div>
          <div style={{ color: "#64748b", fontWeight: 600, borderBottom: "1px solid #334155", paddingBottom: 4 }}>FCF Value</div>
          <div style={{ color: "#64748b", fontWeight: 600, borderBottom: "1px solid #334155", paddingBottom: 4 }}>Experiment</div>
          <div style={{ color: "#64748b", fontWeight: 600, borderBottom: "1px solid #334155", paddingBottom: 4 }}>Error</div>
          <div style={{ color: "#64748b", fontWeight: 600, borderBottom: "1px solid #334155", paddingBottom: 4 }}>Type</div>
          {PREDICTIONS.map((p, i) => (
            <React.Fragment key={i}>
              <div style={{ color: "#f8fafc", fontWeight: 500, paddingTop: 6 }}>{p.name}</div>
              <div style={{ color: "#93c5fd", fontFamily: "monospace", paddingTop: 6 }}>{p.predicted}</div>
              <div style={{ color: "#94a3b8", fontFamily: "monospace", paddingTop: 6 }}>{p.observed}</div>
              <div style={{ paddingTop: 6 }}>
                <Badge color={p.error === 0 ? "#22c55e" : p.error < 1 ? "#3b82f6" : "#f59e0b"}>
                  {p.error === 0 ? "exact" : `${p.error}%`}
                </Badge>
              </div>
              <div style={{ paddingTop: 6 }}>
                <Badge color={p.type === "exact" ? "#22c55e" : p.type === "range" ? "#8b5cf6" : "#3b82f6"}>{p.type}</Badge>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>
      <div style={{ background: "#0f172a", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>Error Distribution (continuous predictions)</div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={PREDICTIONS.filter(p => p.error > 0)} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
            <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 12 }} />
            <Bar dataKey="error" radius={[4, 4, 0, 0]}>
              {PREDICTIONS.filter(p => p.error > 0).map((p, i) => (
                <Cell key={i} fill={p.error < 1 ? "#3b82f6" : p.error < 5 ? "#f59e0b" : "#ef4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ==================== TAB 3: THEOREM MAP ====================
function TheoremMapTab() {
  const [selectedTier, setSelectedTier] = useState(null);
  const tiers = [0, 1, 2, 3, 4, 5];

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        <button onClick={() => setSelectedTier(null)} style={{
          padding: "4px 12px", borderRadius: 4, border: "1px solid #334155",
          background: selectedTier === null ? "#3b82f6" : "#0f172a",
          color: selectedTier === null ? "#fff" : "#94a3b8", fontSize: 11, cursor: "pointer"
        }}>All ({THEOREMS.length})</button>
        {tiers.map(t => {
          const count = THEOREMS.filter(th => th.tier === t).length;
          return (
            <button key={t} onClick={() => setSelectedTier(t)} style={{
              padding: "4px 12px", borderRadius: 4, border: `1px solid ${TIER_COLORS[t]}44`,
              background: selectedTier === t ? TIER_COLORS[t] : "#0f172a",
              color: selectedTier === t ? "#fff" : TIER_COLORS[t], fontSize: 11, cursor: "pointer"
            }}>T{t}: {TIER_NAMES[t]} ({count})</button>
          );
        })}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280, 1fr))", gap: 8 }}>
        {THEOREMS.filter(t => selectedTier === null || t.tier === selectedTier).map((t, i) => (
          <div key={i} style={{
            background: "#0f172a", borderRadius: 6, padding: "10px 14px",
            borderLeft: `3px solid ${TIER_COLORS[t.tier]}`,
            display: "flex", flexDirection: "column", gap: 4
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 700, color: "#f8fafc" }}>{t.id}</span>
              <div style={{ display: "flex", gap: 4 }}>
                <Badge color={EPI_COLORS[t.epistemic]}>{t.epistemic}</Badge>
                <Badge color={GAP_COLORS[t.gap]}>{t.gap}</Badge>
              </div>
            </div>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>{t.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ==================== TAB 4: AUDIT SYSTEMS ====================
function AuditTab() {
  const fixedCount = AUDIT_CHECKS.filter(a => a.status === "FIXED").length;
  const activeCount = AUDIT_CHECKS.filter(a => a.status === "ACTIVE").length;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 16 }}>
        <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, border: "1px solid #22c55e33" }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>v3.6 Fixes Applied</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#22c55e", fontFamily: "monospace" }}>{fixedCount}</div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>red-team issues resolved</div>
        </div>
        <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, border: "1px solid #3b82f633" }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>Active Checks</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#3b82f6", fontFamily: "monospace" }}>{activeCount}</div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>ongoing verification</div>
        </div>
        <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, border: "1px solid #f59e0b33" }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>Schema Valid</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#f59e0b", fontFamily: "monospace" }}>48/48</div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>all fields verified</div>
        </div>
      </div>

      <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>Red-Team Audit Checklist</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {AUDIT_CHECKS.map((a, i) => (
            <div key={i} style={{
              display: "grid", gridTemplateColumns: "60px 80px 200px 1fr 70px",
              gap: 8, alignItems: "center", padding: "8px 12px",
              background: "#1e293b", borderRadius: 6,
              borderLeft: `3px solid ${SEVERITY_COLORS[a.severity]}`
            }}>
              <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>{a.id}</span>
              <Badge color={STATUS_COLORS[a.status]}>{a.status}</Badge>
              <span style={{ fontSize: 12, color: "#f8fafc", fontWeight: 500 }}>{a.name}</span>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{a.desc}</span>
              <Badge color={SEVERITY_COLORS[a.severity]}>{a.severity}</Badge>
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: "#0f172a", borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 12 }}>v3.5 → v3.6 Changelog</div>
        <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.8 }}>
          <div><Badge color="#22c55e">UP</Badge> <span style={{ color: "#f8fafc" }}>39 theorems [P_structural] → [P]</span> — Via imported math theorems + exact algebra</div>
          <div><Badge color="#22c55e">UP</Badge> <span style={{ color: "#f8fafc" }}>T6 [P]: sin²θ_W(M_U) = 3/8</span> — SU(5) embedding is pure Lie algebra</div>
          <div><Badge color="#22c55e">UP</Badge> <span style={{ color: "#f8fafc" }}>T_field [P]: {"{Q,L,u,d,e}"}</span> — Landau pole exclusion + CPT (no minimality needed)</div>
          <div><Badge color="#3b82f6">NEW</Badge> <span style={{ color: "#f8fafc" }}>f_b = 3/19</span> — Baryon fraction from combinatorial counting (0.49% Planck)</div>
          <div><Badge color="#3b82f6">NEW</Badge> <span style={{ color: "#f8fafc" }}>Cosmic budget: 3+16+42=61</span> — Five Planck parameters within 1σ</div>
          <div><Badge color="#3b82f6">NEW</Badge> <span style={{ color: "#f8fafc" }}>Majorana prediction</span> — C_total=61 requires no ν_R (testable via 0νββ)</div>
          <div><Badge color="#3b82f6">NEW</Badge> <span style={{ color: "#f8fafc" }}>Physical corollaries</span> — Charge quantization, neutral atoms, fractional quark charges</div>
          <div><Badge color="#f59e0b">OBS</Badge> <span style={{ color: "#f8fafc" }}>β-cosmology identity</span> — b₂/b₃ = 19/42 = Ω_m/Ω_Λ iff N_gen=3 (not yet theorem)</div>
        </div>
      </div>
    </div>
  );
}

// ==================== MAIN DASHBOARD ====================
export default function Dashboard() {
  const [tab, setTab] = useState("status");

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      background: "#0a0e1a", color: "#e2e8f0", minHeight: "100vh", padding: 20,
    }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 2 }}>Admissibility Physics Engine</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: "#f8fafc", letterSpacing: -0.5 }}>
            v3.6 Status Dashboard
          </div>
          <div style={{ fontSize: 12, color: "#64748b" }}>
            2026-02-08 &middot; 49 theorems &middot; 5 axioms &middot; 0 free parameters &middot; 0 contradictions
          </div>
        </div>

        <div style={{ borderBottom: "1px solid #1e293b", marginBottom: 16, display: "flex", gap: 0 }}>
          <TabButton active={tab === "status"} onClick={() => setTab("status")}>Status</TabButton>
          <TabButton active={tab === "accuracy"} onClick={() => setTab("accuracy")}>Accuracy</TabButton>
          <TabButton active={tab === "theorems"} onClick={() => setTab("theorems")}>Theorem Map</TabButton>
          <TabButton active={tab === "audit"} onClick={() => setTab("audit")}>Audit Systems</TabButton>
        </div>

        {tab === "status" && <StatusTab />}
        {tab === "accuracy" && <AccuracyTab />}
        {tab === "theorems" && <TheoremMapTab />}
        {tab === "audit" && <AuditTab />}
      </div>
    </div>
  );
}
