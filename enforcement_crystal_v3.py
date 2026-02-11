#!/usr/bin/env python3
"""
================================================================================
ENFORCEMENT CRYSTAL v3 — Auto-Extracting Analysis
================================================================================

KEY CHANGE FROM v2: No hardcoded DEPENDENCY_MAP. Instead, this script:
  1. Imports and runs the theorem bank (fcf_theorem_bank.py)
  2. Extracts the 'dependencies' field from every theorem's check() result
  3. Builds the DAG dynamically
  4. Runs all graph-theoretic analyses

This means the crystal ALWAYS reflects the current state of the theorem bank.
No more stale data. If you update a theorem's dependencies, just re-run this.

Source of truth: fcf_theorem_bank.py THEOREM_REGISTRY
  - Every check_*() function returns {'dependencies': [...], ...}
  - The bank speaks in 5-axiom vocabulary (A1, A2, A3, A4, A5)
  - Mode-aware remapping converts to 3-axiom or 1-axiom form

Usage:
  python3 enforcement_crystal_v3.py                    # full comparative run
  python3 enforcement_crystal_v3.py --mode 3-axiom     # single mode
  python3 enforcement_crystal_v3.py --diff             # show what changed vs v2

================================================================================
"""

import json
import re
import sys
from collections import defaultdict, deque, Counter
from math import log2
from typing import Dict, Set, List, Tuple, Any

# ============================================================================
# STEP 1: EXTRACT DEPENDENCIES FROM THEOREM BANK (the single source of truth)
# ============================================================================

def extract_dependencies_from_bank() -> Dict[str, List[str]]:
    """
    Run every theorem check in the bank and extract declared dependencies.
    Returns {theorem_id: [dep1, dep2, ...]} in the bank's native vocabulary.
    """
    from fcf_theorem_bank import THEOREM_REGISTRY, run_all

    results = run_all()
    dep_map = {}

    for tid, res in results.items():
        raw_deps = res.get('dependencies', [])
        cleaned = [_clean_dep_string(d) for d in raw_deps]
        dep_map[tid] = cleaned

    return dep_map


def _clean_dep_string(d: str) -> str:
    """
    Normalize dependency strings to bare IDs.
    Handles annotated forms like 'L_nc (from A1 + A3, via A45)'
    and variant names like 'L_epsilon' -> 'L_epsilon*'.
    """
    d = d.strip()
    # Extract leading identifier
    m = re.match(r'^([A-Za-z0-9_*]+)', d)
    if m:
        name = m.group(1)
        # Normalize L_epsilon variants
        if name in ('L_epsilon', 'L_e', 'L_e*'):
            return 'L_epsilon*'
        # Normalize Δ variants (in case of unicode issues)
        return name
    return d


# ============================================================================
# STEP 2: KNOWN STRUCTURAL FACTS (things the bank doesn't encode)
# ============================================================================

# Physical tiers — these are interpretive, not dependency data.
# Could also be extracted if the bank encoded them, but they're stable.
TIER_MAP = {
    # Reduction layer
    'L_loc': 0, 'L_irr': 0, 'L_nc': 0,
    # Tier 0: Axiom-level foundations
    'T0': 0, 'T1': 0, 'L_T2': 0, 'T2': 0, 'T3': 0,
    'L_epsilon*': 0, 'T_epsilon': 0, 'T_eta': 0, 'T_kappa': 0,
    'T_M': 0, 'T_Hermitian': 0,
    # Tier 1: Gauge group selection
    'T4': 1, 'T5': 1, 'T_gauge': 1,
    # Tier 2: Particle content
    'T_field': 2, 'T_channels': 2, 'T7': 2, 'T4E': 2, 'T4F': 2,
    'T4G': 2, 'T4G_Q31': 2, 'T_Higgs': 2, 'T9': 2,
    # Tier 3: Continuous constants / RG
    'T6': 3, 'T6B': 3, 'T19': 3, 'T20': 3, 'T21': 3, 'T22': 3,
    'T23': 3, 'T24': 3, 'T25a': 3, 'T25b': 3, 'T26': 3,
    'T27c': 3, 'T27d': 3, 'T_sin2theta': 3, 'T_S0': 3,
    # Quantum chain
    'T_Born': 0, 'T_CPTP': 0, 'T_tensor': 0, 'T_entropy': 0,
    # Tier 4: Gravity
    'T7B': 4, 'T_particle': 4, 'T8': 4, 'T9_grav': 4,
    'T10': 4, 'T11': 4, 'T12': 4, 'T12E': 4, 'T_Bek': 4,
    # Tier 5: Δ_geo closure
    'Delta_ordering': 5, 'Delta_fbc': 5, 'Delta_particle': 5,
    'Delta_continuum': 5, 'Delta_signature': 5, 'Delta_closure': 5,
}

# Physical sectors for dominance analysis
SECTOR_MAP = {
    'quantum_spine': ['T1', 'L_T2', 'T2', 'T_Hermitian', 'T_Born', 'T_CPTP', 'T_tensor', 'T_entropy'],
    'gauge_selection': ['T4', 'T5', 'T_gauge', 'T6'],
    'particle_content': ['T_field', 'T_channels', 'T7', 'T4E', 'T4F', 'T4G', 'T4G_Q31', 'T_Higgs', 'T9'],
    'ew_constants': ['T6B', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25a', 'T25b', 'T26', 'T27c', 'T27d', 'T_sin2theta', 'T_S0'],
    'gravity': ['T7B', 'T8', 'T9_grav', 'T10', 'T11', 'T_particle'],
    'dark_sector': ['T12', 'T12E', 'T_Bek'],
    'geo_closure': ['Delta_ordering', 'Delta_fbc', 'Delta_particle', 'Delta_continuum', 'Delta_signature', 'Delta_closure'],
    'enforcement_core': ['T0', 'L_epsilon*', 'T_epsilon', 'T_eta', 'T_kappa', 'T_M', 'L_nc', 'L_irr', 'L_loc'],
}

# Known cycles in the EW sector (self-consistency conditions).
# These must be broken to get a DAG. The paper should document these.
KNOWN_CYCLES = {
    ('T27c', 'T_S0'),   # T27c's dep on T_S0 (mutual constraint)
    ('T26', 'T21'),      # T26's dep on T21 (RG feedback)
    ('T21', 'T27c'),     # T21's dep on T27c (mixing feedback)
    ('T21', 'T27d'),     # T21's dep on T27d (mixing feedback)
}

# Mode-specific dependency overrides.
#
# WHY THIS IS NEEDED: The theorem bank declares dependencies in 5-axiom
# vocabulary. When we reduce to 3 axioms, naive remapping of A2→L_nc
# creates a self-loop (L_nc depends on A2, A2 maps to L_nc → circular).
# The actual 3-axiom derivation of L_nc comes from A1+A3 (not A1+A2).
# Similarly for 1-axiom mode where L_nc comes from A1+L_loc.
#
# These overrides capture the actual derivation paths in each mode.
# They are the ONE place where human judgment enters the auto-extraction.

MODE_OVERRIDES = {
    '3-axiom': {
        # No overrides needed — L_nc now natively declares ['A1', 'A3']
    },
    '1-axiom': {
        # L_nc comes from A1 + L_loc (since A3 → L_loc in 1-axiom mode)
        'L_nc': ['A1', 'L_loc'],
    },
    '5-axiom': {},
}


# ============================================================================
# STEP 3: GRAPH CONSTRUCTION
# ============================================================================

class EnforcementCrystal:
    """Graph-theoretic analysis of the FCF dependency DAG."""

    def __init__(self, dep_map: Dict[str, List[str]], axiom_mode='3-axiom',
                 cycle_breaks=None):
        """
        dep_map:      {theorem_id: [dep1, dep2, ...]} extracted from bank
        axiom_mode:   '1-axiom', '3-axiom', or '5-axiom'
        cycle_breaks: set of (child, parent) edges to break for DAG-ification
        """
        self.axiom_mode = axiom_mode
        self.adj = defaultdict(set)      # parent -> set of children
        self.radj = defaultdict(set)     # child -> set of parents
        self.nodes = set()
        self.axiom_nodes = set()
        self.cycle_breaks = cycle_breaks or KNOWN_CYCLES
        self.raw_dep_map = dep_map

        self._build(dep_map, axiom_mode)

    def _build(self, dep_map, mode):
        """Build the DAG from extracted dependency map with mode-aware remapping."""
        # Determine source axiom set
        if mode == '1-axiom':
            self.axiom_nodes = {'A1', 'M', 'NT'}
        elif mode == '3-axiom':
            self.axiom_nodes = {'A1', 'A3', 'A4'}
        elif mode == '5-axiom':
            self.axiom_nodes = {'A1', 'A2', 'A3', 'A4', 'A5'}
        else:
            raise ValueError(f"Unknown axiom_mode: {mode}")

        self.nodes = set(self.axiom_nodes)

        # Apply mode-specific overrides (e.g., L_nc -> [A1, A3] in 3-axiom mode)
        overrides = MODE_OVERRIDES.get(mode, {})
        effective_map = dict(dep_map)
        for thm, override_deps in overrides.items():
            effective_map[thm] = override_deps

        for thm, deps in effective_map.items():
            # In 3-axiom mode, skip L_loc and L_irr (A3/A4 are given)
            if mode == '3-axiom' and thm in ('L_loc', 'L_irr'):
                continue
            # In 5-axiom mode, skip reduction lemmas
            if mode == '5-axiom' and thm in ('L_loc', 'L_irr'):
                continue

            self.nodes.add(thm)

            for dep_raw in deps:
                dep = self._remap_dep(dep_raw, mode)
                if dep is None:
                    continue  # dependency absorbed/skipped in this mode

                # Break known cycles
                if (thm, dep) in self.cycle_breaks:
                    continue

                # Only add edge if dep is a known node (axiom or theorem)
                if dep in self.axiom_nodes or dep in effective_map:
                    self.nodes.add(dep)
                    self.adj[dep].add(thm)
                    self.radj[thm].add(dep)

    def _remap_dep(self, dep: str, mode: str) -> str:
        """
        Remap a dependency name based on axiom mode.
        The bank speaks in 5-axiom vocabulary. This converts.
        Returns None if the dep should be dropped in this mode.
        """
        if mode == '3-axiom':
            if dep == 'A2':
                return 'L_nc'      # A2 (non-closure) is now derived as L_nc
            if dep == 'A5':
                return 'L_col'     # A5 (collapse) is now derived as L_col
            if dep in ('M', 'NT'):
                return None         # postulates absorbed into operational axioms
            # Remap L_nc's own dependencies: in 3-axiom form, it comes from A1+A3
            # (The bank says A1+A2, but A2 IS L_nc — circular. The actual
            # derivation is: A1+A3 → L_nc via the non-closure proof.)
            return dep

        elif mode == '1-axiom':
            if dep == 'A3':
                return 'L_loc'     # A3 derived via L_loc
            if dep == 'A4':
                return 'L_irr'     # A4 derived via L_irr
            if dep == 'A2':
                return 'L_nc'
            if dep == 'A5':
                return 'L_col'
            return dep

        else:  # 5-axiom
            return dep

    # ----- Basic metrics -----

    def node_count(self):
        return len(self.nodes)

    def edge_count(self):
        return sum(len(ch) for ch in self.adj.values())

    def theorem_nodes(self):
        return self.nodes - self.axiom_nodes

    # ----- Depth / topological sort -----

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm with cycle detection."""
        in_deg = defaultdict(int)
        for node in self.nodes:
            in_deg[node]  # ensure present
        for node in self.nodes:
            for child in self.adj.get(node, set()):
                if child in self.nodes:
                    in_deg[child] += 1

        queue = deque(n for n in self.nodes if in_deg[n] == 0)
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for ch in self.adj.get(n, set()):
                if ch in self.nodes:
                    in_deg[ch] -= 1
                    if in_deg[ch] == 0:
                        queue.append(ch)

        remaining = self.nodes - set(order)
        if remaining:
            if not hasattr(self, '_cycle_warned'):
                self._cycle_warned = True
                print(f"  ⚠ CYCLE DETECTED: {sorted(remaining)}")
                print(f"    These nodes were not fully linearized.")
            order.extend(sorted(remaining))
        return order

    def compute_depths(self) -> Dict[str, int]:
        """Compute depth of each node (longest path from any axiom)."""
        depths = {a: 0 for a in self.axiom_nodes}
        topo = self._topological_sort()
        for node in topo:
            if node in self.axiom_nodes:
                continue
            parents = self.radj.get(node, set())
            if parents:
                depths[node] = max(depths.get(p, 0) for p in parents) + 1
            else:
                depths[node] = 0
        return depths

    def width_profile(self) -> Dict[int, List[str]]:
        """Width (number of nodes) at each depth level."""
        depths = self.compute_depths()
        profile = defaultdict(list)
        for node, d in depths.items():
            profile[d].append(node)
        return dict(sorted(profile.items()))

    # ----- Betweenness centrality -----

    def betweenness_centrality(self) -> Dict[str, float]:
        """Betweenness centrality (Brandes algorithm for DAG)."""
        cb = {v: 0.0 for v in self.nodes}
        for s in self.nodes:
            S = []
            pred = {v: [] for v in self.nodes}
            sigma = {v: 0 for v in self.nodes}
            sigma[s] = 1
            dist = {v: -1 for v in self.nodes}
            dist[s] = 0
            Q = deque([s])
            while Q:
                v = Q.popleft()
                S.append(v)
                for w in self.adj.get(v, set()):
                    if w not in self.nodes:
                        continue
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        Q.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = {v: 0.0 for v in self.nodes}
            while S:
                w = S.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        n = len(self.nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            cb = {v: c * norm for v, c in cb.items()}
        return cb

    # ----- Cascade failure -----

    def cascade_failure(self, node: str) -> Set[str]:
        """
        Nodes that die if `node` is removed.
        Model: a node dies if ALL its parents have died (AND-dependency).
        """
        failed = {node}
        changed = True
        while changed:
            changed = False
            for n in self.nodes - failed:
                if n in self.axiom_nodes:
                    continue
                parents = self.radj.get(n, set()) & self.nodes
                if not parents:
                    continue
                alive = parents - failed
                if len(alive) == 0:
                    failed.add(n)
                    changed = True
        return failed - {node}

    def all_cascades(self) -> Dict[str, int]:
        """Cascade failure size for every non-axiom node."""
        return {n: len(self.cascade_failure(n)) for n in self.theorem_nodes()}

    # ----- Ancestry / descendants -----

    def ancestors(self, node: str) -> Set[str]:
        """All nodes reachable going UP from node."""
        visited = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for p in self.radj.get(n, set()):
                if p in self.nodes and p not in visited:
                    visited.add(p)
                    queue.append(p)
        return visited

    def descendants(self, node: str) -> Set[str]:
        """All nodes reachable going DOWN from node."""
        visited = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for c in self.adj.get(n, set()):
                if c in self.nodes and c not in visited:
                    visited.add(c)
                    queue.append(c)
        return visited

    def axiom_loads(self) -> Dict[str, int]:
        """Number of descendants for each axiom."""
        return {a: len(self.descendants(a)) for a in self.axiom_nodes}

    # ----- Path counting -----

    def count_paths(self, target: str) -> int:
        """Count distinct directed paths from any axiom to target."""
        topo = self._topological_sort()
        paths = {}
        for node in topo:
            if node in self.axiom_nodes:
                paths[node] = 1
            else:
                parents = self.radj.get(node, set()) & self.nodes
                paths[node] = sum(paths.get(p, 0) for p in parents)
        return paths.get(target, 0)

    def count_paths_from(self, source: str, target: str) -> int:
        """Count paths from a specific source to target."""
        topo = self._topological_sort()
        paths = {}
        for node in topo:
            if node == source:
                paths[node] = 1
            elif node in self.axiom_nodes:
                paths[node] = 0
            else:
                parents = self.radj.get(node, set()) & self.nodes
                paths[node] = sum(paths.get(p, 0) for p in parents)
        return paths.get(target, 0)

    def axiom_attribution(self, target: str) -> Dict[str, float]:
        """Fraction of paths to target from each axiom."""
        axiom_paths = {}
        total = 0
        for a in self.axiom_nodes:
            c = self.count_paths_from(a, target)
            axiom_paths[a] = c
            total += c
        if total == 0:
            return {a: 0.0 for a in self.axiom_nodes}
        return {a: c / total for a, c in axiom_paths.items()}

    # ----- Axiom fingerprint -----

    def axiom_fingerprint(self, node: str) -> Dict[str, bool]:
        """Which axioms are in the ancestry of this node?"""
        anc = self.ancestors(node) | {node}
        return {a: a in anc for a in sorted(self.axiom_nodes)}

    # ----- Sector dominance -----

    def sector_dominance(self) -> Dict[str, Dict[str, str]]:
        """For each sector, fraction of theorems requiring each axiom."""
        results = {}
        for sector, theorems in SECTOR_MAP.items():
            active = [t for t in theorems if t in self.nodes]
            if not active:
                continue
            counts = {}
            for a in sorted(self.axiom_nodes):
                c = sum(1 for t in active if self.axiom_fingerprint(t).get(a, False))
                counts[a] = f"{c}/{len(active)}"
            results[sector] = counts
        return results

    # ----- Entropy measures -----

    def degree_entropy(self, direction='out') -> Tuple[float, float, float]:
        """Shannon entropy of degree distribution. Returns (entropy, max, evenness)."""
        if direction == 'out':
            degrees = [len(self.adj.get(n, set())) for n in self.nodes]
        else:
            degrees = [len(self.radj.get(n, set()) & self.nodes) for n in self.nodes]
        counts = Counter(degrees)
        total = sum(counts.values())
        if total == 0:
            return (0.0, 0.0, 0.0)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        max_ent = log2(len(counts)) if len(counts) > 1 else 1.0
        evenness = entropy / max_ent if max_ent > 0 else 0.0
        return (round(entropy, 2), round(max_ent, 2), round(evenness, 2))

    def axiom_load_entropy(self) -> Tuple[float, float, float]:
        """Shannon entropy of axiom load distribution."""
        loads = self.axiom_loads()
        total = sum(loads.values())
        if total == 0:
            return (0.0, 0.0, 0.0)
        probs = [v / total for v in loads.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        max_ent = log2(len(loads)) if len(loads) > 1 else 1.0
        evenness = entropy / max_ent if max_ent > 0 else 0.0
        return (round(entropy, 2), round(max_ent, 2), round(evenness, 2))

    # ----- Hourglass / waist detection -----

    def find_waists(self, threshold=3) -> List[Tuple[int, List[str]]]:
        """Find depths where width <= threshold."""
        profile = self.width_profile()
        return [(d, nodes) for d, nodes in sorted(profile.items())
                if len(nodes) <= threshold]

    # ----- Independence analysis -----

    def check_independence(self, set_a: List[str], set_b: List[str]) -> Dict[str, Any]:
        """Check if two sets of theorems have independent derivation chains."""
        anc_a = set()
        for t in set_a:
            if t in self.nodes:
                anc_a |= self.ancestors(t) | {t}
        anc_b = set()
        for t in set_b:
            if t in self.nodes:
                anc_b |= self.ancestors(t) | {t}
        shared = (anc_a & anc_b) - self.axiom_nodes
        return {
            'set_a_ancestors': len(anc_a),
            'set_b_ancestors': len(anc_b),
            'shared_non_axiom': len(shared),
            'shared_nodes': sorted(shared),
            'independent': len(shared) == 0,
        }


# ============================================================================
# STEP 4: DIFF AGAINST v2 HARDCODED MAP
# ============================================================================

def diff_against_v2(extracted_map: Dict[str, List[str]]):
    """Compare extracted deps against the v2 hardcoded map."""
    # Import old hardcoded map
    try:
        # We inline the v2 map here for comparison
        from enforcement_crystal_v2_deps import DEPENDENCY_MAP as V2_MAP
    except ImportError:
        V2_MAP = None

    if V2_MAP is None:
        print("  (v2 dependency map not available for diff)")
        return

    all_thms = sorted(set(extracted_map.keys()) | set(V2_MAP.keys()))
    diffs = []
    for tid in all_thms:
        v2_deps = sorted(V2_MAP.get(tid, []))
        v3_deps = sorted(extracted_map.get(tid, []))
        if v2_deps != v3_deps:
            diffs.append((tid, v2_deps, v3_deps))

    if not diffs:
        print("  ✓ No differences between extracted and v2 hardcoded maps")
    else:
        print(f"  ✗ {len(diffs)} differences found:")
        for tid, v2, v3 in diffs:
            print(f"    {tid}:")
            print(f"      v2: {v2}")
            print(f"      v3: {v3}")


# ============================================================================
# STEP 5: RUN FULL ANALYSIS
# ============================================================================

def run_crystal_analysis(dep_map: Dict[str, List[str]], mode='3-axiom') -> Dict[str, Any]:
    """Run the complete crystal analysis for a given axiom mode."""
    crystal = EnforcementCrystal(dep_map, axiom_mode=mode)

    print(f"\n{'='*74}")
    print(f"  ENFORCEMENT CRYSTAL v3 — {mode.upper()} MODE")
    print(f"  (dependencies auto-extracted from theorem bank)")
    print(f"{'='*74}")

    # 1. Basic counts
    n_nodes = crystal.node_count()
    n_edges = crystal.edge_count()
    n_axioms = len(crystal.axiom_nodes)
    n_theorems = len(crystal.theorem_nodes())
    print(f"\n  Nodes: {n_nodes} ({n_axioms} axioms + {n_theorems} derived)")
    print(f"  Edges: {n_edges}")

    # 2. Width profile
    profile = crystal.width_profile()
    max_depth = max(profile.keys()) if profile else 0
    print(f"\n  WIDTH PROFILE (depth → width):")
    for d in range(max_depth + 1):
        nodes = profile.get(d, [])
        bar = '█' * len(nodes)
        names = ', '.join(sorted(nodes)[:5])
        if len(nodes) > 5:
            names += f", ... (+{len(nodes)-5})"
        marker = ' ★' if len(nodes) == 1 else ''
        print(f"  Depth {d:2d}: {len(nodes):3d}  {bar}  [{names}]{marker}")

    # 3. Width-1 waists
    waists_1 = [(d, n) for d, n in profile.items() if len(n) == 1]
    if waists_1:
        print(f"\n  WIDTH-1 WAISTS (structural bottlenecks):")
        for d, nodes in sorted(waists_1):
            print(f"    Depth {d}: {nodes[0]}")

    # 4. Betweenness centrality
    bc = crystal.betweenness_centrality()
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])[:15]
    print(f"\n  BETWEENNESS CENTRALITY (top 15):")
    for node, score in bc_sorted:
        print(f"    {node:18s}  {score:.4f}")

    # 5. Cascade failure
    cascades = crystal.all_cascades()
    cas_sorted = sorted(cascades.items(), key=lambda x: -x[1])[:10]
    print(f"\n  CASCADE FAILURE (top 10 — nodes killed if removed):")
    for node, count in cas_sorted:
        parents = crystal.radj.get(node, set()) & crystal.nodes
        risk = "HIGH" if len(parents) <= 1 else "low"
        print(f"    {node:18s}  cascade={count:3d}  parents={len(parents)}  risk={risk}")

    # 6. Axiom loads
    loads = crystal.axiom_loads()
    print(f"\n  AXIOM LOADS (descendants):")
    for a in sorted(loads, key=lambda x: -loads[x]):
        pct = round(100 * loads[a] / n_theorems) if n_theorems > 0 else 0
        print(f"    {a:6s}: {loads[a]:3d} descendants ({pct}%)")

    # 7. Path counting to sin²θ_W
    sin2theta_data = None
    if 'T_sin2theta' in crystal.nodes:
        paths = crystal.count_paths('T_sin2theta')
        ancestors = crystal.ancestors('T_sin2theta')
        attr = crystal.axiom_attribution('T_sin2theta')
        print(f"\n  SIN²θ_W PREDICTION FUNNEL:")
        print(f"    Total paths to T_sin2theta: {paths:,}")
        print(f"    Ancestor count: {len(ancestors)}")
        print(f"    Axiom attribution weights:")
        for a in sorted(attr, key=lambda x: -attr[x]):
            print(f"      {a:6s}: {attr[a]*100:.1f}%")
        sin2theta_data = {
            'paths': paths,
            'ancestors': len(ancestors),
            'attribution': {a: round(v, 4) for a, v in attr.items()},
        }

    # 8. Entropy measures
    out_ent = crystal.degree_entropy('out')
    in_ent = crystal.degree_entropy('in')
    ax_ent = crystal.axiom_load_entropy()
    print(f"\n  ENTROPY MEASURES:")
    print(f"    {'Measure':30s}  {'Entropy':>8s}  {'Maximum':>8s}  {'Evenness':>8s}")
    print(f"    {'Out-degree distribution':30s}  {out_ent[0]:8.2f}  {out_ent[1]:8.2f}  {out_ent[2]:8.2f}")
    print(f"    {'In-degree distribution':30s}  {in_ent[0]:8.2f}  {in_ent[1]:8.2f}  {in_ent[2]:8.2f}")
    print(f"    {'Axiom load distribution':30s}  {ax_ent[0]:8.2f}  {ax_ent[1]:8.2f}  {ax_ent[2]:8.2f}")

    # 9. Sector dominance
    sec_dom = crystal.sector_dominance()
    print(f"\n  SECTOR DOMINANCE (axiom required by fraction of sector):")
    axiom_list = sorted(crystal.axiom_nodes)
    header = f"    {'Sector':22s}" + "".join(f"  {a:>6s}" for a in axiom_list)
    print(header)
    for sector, counts in sec_dom.items():
        row = f"    {sector:22s}"
        for a in axiom_list:
            row += f"  {counts.get(a, '0/0'):>6s}"
        print(row)

    # 10. QM/GR independence
    qm_nodes = ['T1', 'T2', 'T_Born', 'T_CPTP', 'T_tensor', 'T_entropy']
    gr_nodes = ['T7B', 'T8', 'T9_grav']
    indep = crystal.check_independence(qm_nodes, gr_nodes)
    print(f"\n  QM / GR INDEPENDENCE:")
    print(f"    QM chain ancestors: {indep['set_a_ancestors']}")
    print(f"    GR chain ancestors: {indep['set_b_ancestors']}")
    print(f"    Shared (non-axiom): {indep['shared_non_axiom']}")
    if indep['shared_nodes']:
        print(f"    Shared nodes: {indep['shared_nodes']}")
    print(f"    Independent: {'YES' if indep['independent'] else 'NO'}")

    return {
        'mode': mode,
        'node_count': n_nodes,
        'edge_count': n_edges,
        'axiom_count': n_axioms,
        'theorem_count': n_theorems,
        'axiom_nodes': sorted(crystal.axiom_nodes),
        'width_profile': {d: sorted(nodes) for d, nodes in profile.items()},
        'width_1_waists': [(d, sorted(n)) for d, n in profile.items() if len(n) == 1],
        'betweenness_top15': [(n, round(s, 4)) for n, s in bc_sorted],
        'cascade_top10': [(n, c) for n, c in cas_sorted],
        'axiom_loads': loads,
        'sin2theta': sin2theta_data,
        'entropy': {
            'out_degree': out_ent,
            'in_degree': in_ent,
            'axiom_load': ax_ent,
        },
        'sector_dominance': sec_dom,
        'qm_gr_independence': indep,
    }


# ============================================================================
# STEP 6: COMPARATIVE ANALYSIS
# ============================================================================

def compare_crystal_versions():
    """Extract dependencies, run both modes, compare."""
    print("\n" + "█" * 74)
    print("  ENFORCEMENT CRYSTAL v3: AUTO-EXTRACTING ANALYSIS")
    print("  Dependencies extracted live from theorem bank")
    print("█" * 74)

    # Step 1: Extract
    print("\n  Extracting dependencies from theorem bank...")
    dep_map = extract_dependencies_from_bank()
    print(f"  ✓ Extracted {len(dep_map)} theorem dependency lists")

    # Step 2: Run both modes
    results = {}
    for mode in ['3-axiom', '1-axiom']:
        results[mode] = run_crystal_analysis(dep_map, mode)

    # Step 3: Comparison summary
    r3 = results['3-axiom']
    r1 = results['1-axiom']

    print(f"\n\n{'='*74}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*74}")

    print(f"\n  {'Metric':35s}  {'Original (5-ax)':>15s}  {'Current (3-ax)':>15s}  {'Reduced (1-ax)':>15s}")
    print(f"  {'─'*35}  {'─'*15}  {'─'*15}  {'─'*15}")
    print(f"  {'Axiom nodes':35s}  {'5':>15s}  {r3['axiom_count']:>15d}  {r1['axiom_count']:>15d}")
    print(f"  {'Theorem nodes':35s}  {'51':>15s}  {r3['theorem_count']:>15d}  {r1['theorem_count']:>15d}")
    print(f"  {'Total nodes':35s}  {'56':>15s}  {r3['node_count']:>15d}  {r1['node_count']:>15d}")
    print(f"  {'Total edges':35s}  {'164':>15s}  {r3['edge_count']:>15d}  {r1['edge_count']:>15d}")

    if r3.get('sin2theta') and r1.get('sin2theta'):
        print(f"  {'Paths to sin²θ_W':35s}  {'1,398':>15s}  {r3['sin2theta']['paths']:>15,}  {r1['sin2theta']['paths']:>15,}")
        print(f"  {'sin²θ_W ancestors':35s}  {'32':>15s}  {r3['sin2theta']['ancestors']:>15d}  {r1['sin2theta']['ancestors']:>15d}")

    if r3['betweenness_top15']:
        print(f"  {'Highest betweenness node':35s}  {'L_ε*':>15s}  {r3['betweenness_top15'][0][0]:>15s}  {r1['betweenness_top15'][0][0]:>15s}")

    ax3 = r3['entropy']['axiom_load']
    ax1 = r1['entropy']['axiom_load']
    print(f"  {'Axiom load evenness':35s}  {'0.98':>15s}  {ax3[2]:>15.2f}  {ax1[2]:>15.2f}")

    return results, dep_map


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    mode_arg = None
    do_diff = False

    for arg in sys.argv[1:]:
        if arg == '--diff':
            do_diff = True
        elif arg.startswith('--mode='):
            mode_arg = arg.split('=', 1)[1]
        elif arg == '--mode' and len(sys.argv) > sys.argv.index(arg) + 1:
            mode_arg = sys.argv[sys.argv.index(arg) + 1]

    # Always extract fresh
    print("  Extracting dependencies from theorem bank...")
    dep_map = extract_dependencies_from_bank()
    print(f"  ✓ Extracted {len(dep_map)} theorem dependency lists\n")

    if mode_arg:
        results = run_crystal_analysis(dep_map, mode_arg)
    else:
        results, dep_map = compare_crystal_versions()

    # Export JSON
    output = {
        'analysis': 'Enforcement Crystal v3 (auto-extracted)',
        'source': 'fcf_theorem_bank.py THEOREM_REGISTRY',
        'extracted_theorems': len(dep_map),
        'cycle_breaks': [list(cb) for cb in sorted(KNOWN_CYCLES)],
    }
    if isinstance(results, dict) and 'mode' in results:
        output['results'] = {results['mode']: results}
    elif isinstance(results, tuple):
        output['results'] = results[0]
    else:
        output['results'] = results

    with open('crystal_v3_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results exported to crystal_v3_analysis.json")
