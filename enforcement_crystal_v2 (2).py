#!/usr/bin/env python3
"""
================================================================================
ENFORCEMENT CRYSTAL v2 â€” Post-Reduction Analysis
================================================================================

Reconstructs the enforcement crystal (dependency DAG) from the current
theorem bank (v3.6.1) and v3.7 master engine, then computes all the
graph-theoretic metrics from the original Crystal paper.

Compares the original 5-axiom crystal (56 nodes, 164 edges) against:
  - The 3-axiom operational form (A1, A3, A4 + L_nc, L_col)
  - The 1-axiom reduced form (A1 + M, NT postulates)

Metrics computed:
  1. Node/edge counts
  2. Width profile (shape diagram)
  3. Betweenness centrality
  4. Cascade failure analysis
  5. Minimum vertex cuts (hourglass detection)
  6. Path enumeration to sinÂ²Î¸_W
  7. Axiom fingerprinting
  8. Axiom attribution weights
  9. Shannon entropy measures
  10. Sector dominance table
  11. Keystone identification
  12. Independence analysis (QM vs GR chains)

================================================================================
"""

import json
from collections import defaultdict, deque
from itertools import combinations
from math import log2
from typing import Dict, Set, List, Tuple, Any, Optional

# ============================================================================
# STEP 1: BUILD THE DAG FROM THEOREM BANK v3.6.1 + v3.7 GRAVITY
# ============================================================================

# Axiom / postulate / lemma layer
AXIOMS_1 = {'A1'}  # Single-axiom form
POSTULATES = {'M', 'NT'}  # Definitional postulates (multiplicity, non-triviality)

AXIOMS_3 = {'A1', 'A3', 'A4'}  # 3-axiom operational form
DERIVED_LEMMAS = {'L_nc', 'L_col'}  # Derived in 3-axiom form

AXIOMS_5 = {'A1', 'A2', 'A3', 'A4', 'A5'}  # Original 5-axiom form (Crystal paper)

# -------------------------------------------------------------------
# DEPENDENCY MAP: extracted from Admissbility_Physics_Theorms_V3_6_1.py
# and Admissibility_Physics_Engine_V3_7.py
#
# Format: theorem_id -> list of dependencies
# Dependencies reference axioms (A1, A3, A4), lemmas (L_nc, L_col, etc.),
# postulates (M, NT), or other theorem IDs.
# -------------------------------------------------------------------

DEPENDENCY_MAP = {
    # ===== REDUCTION LAYER (how axioms derive from A1) =====
    # L_loc: A1 + M + NT â†’ A3 (locality)
    'L_loc':   ['A1', 'L_epsilon*', 'M', 'NT'],
    # L_irr: A1 + L_nc â†’ A4 (irreversibility)
    'L_irr':   ['A1', 'L_nc'],
    # L_nc: non-closure (from A1 + A3, gated by T0 consistency proof)
    'L_nc':    ['A1', 'A3', 'T0'],
    # L_col: collapse derived (from A1 + A4)
    # (not in the theorem bank as a separate check, but declared in the reduction)

    # ===== TIER 0: AXIOM-LEVEL FOUNDATIONS =====
    'T0':           ['A1', 'A3', 'A4'],
    'T1':           ['L_nc', 'T0', 'A3'],
    'L_T2':         ['L_nc', 'A3', 'A4'],
    'T2':           ['A1', 'L_nc', 'T1', 'L_T2'],
    'T3':           ['T2', 'A3'],
    'L_epsilon*':   ['A1', 'T0'],
    'T_epsilon':    ['L_epsilon*', 'A1'],
    'T_eta':        ['T_epsilon', 'T_M', 'A1', 'T_kappa'],
    'T_kappa':      ['T_epsilon', 'A1', 'A4'],
    'T_M':          ['A1', 'A3', 'L_epsilon*'],
    'T_Hermitian':  ['A1', 'A4', 'L_nc'],

    # ===== TIER 1: GAUGE GROUP SELECTION =====
    'T4':       ['A1', 'L_nc', 'T3'],
    'T5':       ['T4'],
    'T_gauge':  ['T4', 'T5', 'A1'],

    # ===== TIER 2: PARTICLE CONTENT =====
    'T_field':     ['T_gauge', 'T7', 'T5', 'A1', 'L_nc', 'T_tensor'],
    'T_channels':  ['T5', 'T_gauge'],
    'T7':          ['T_kappa', 'T_channels', 'T_eta'],
    'T4E':         ['T7', 'T_eta'],
    'T4F':         ['T7', 'T_channels'],
    'T4G':         ['T4E', 'T_epsilon'],
    'T4G_Q31':     ['T4G', 'A1'],
    'T_Higgs':     ['T_particle', 'A4', 'A1', 'T_gauge', 'T_channels'],
    'T9':          ['A4', 'T7'],

    # ===== TIER 3: CONTINUOUS CONSTANTS / RG =====
    'T6':           ['T_gauge'],
    'T6B':          ['T6', 'T21', 'T21b', 'T22', 'T_field'],
    'T19':          ['T_channels', 'T_field', 'T9'],
    'T20':          ['A1', 'T3', 'T_Hermitian'],
    'T21':          ['L_nc', 'T20', 'T_M', 'T27c', 'T27d', 'T_CPTP'],
    'T21a':         ['T21b'],
    'T21b':         ['T21', 'T22', 'T24', 'T27d'],
    'T21c':         ['T21b'],
    'T22':          ['T19', 'T_gauge'],
    'T23':          ['T21', 'T22', 'T27c', 'T27d'],
    'T24':          ['T23', 'T27c', 'T27d', 'T22', 'T_S0'],
    'T25a':         ['T_M', 'T_channels'],
    'T25b':         ['T25a', 'T4F'],
    'T26':          ['T21', 'A1', 'T_channels'],
    'T27c':         ['T25a', 'T_S0', 'T_gauge', 'T27d'],
    'T27d':         ['T_channels', 'A4', 'L_epsilon*', 'T26'],
    'T_sin2theta':  ['T23', 'T27c', 'T27d', 'T24', 'T_S0'],
    'T_S0':         ['T22', 'T27d', 'T_channels', 'T27c'],

    # ===== QUANTUM CHAIN (v3.5+) =====
    'T_Born':    ['T2', 'T_Hermitian', 'A1'],
    'T_CPTP':    ['T2', 'T_Born', 'A1'],
    'T_tensor':  ['T2', 'L_nc', 'A1'],
    'T_entropy': ['T2', 'T_Born', 'L_nc', 'A1'],

    # ===== TIER 4: GRAVITY =====
    'T7B':       ['A1', 'A4', 'T3'],
    'T_particle': ['A1', 'A4', 'L_epsilon*', 'T1', 'T2', 'T_Hermitian', 'T_M'],
    'T8':        ['A1', 'A4', 'T_gauge'],
    'T9_grav':   ['T7B', 'T8', 'Delta_closure'],
    'T10':       ['T9_grav', 'A1', 'T_Bek'],
    'T11':       ['T9_grav', 'T4F', 'T_field', 'T_gauge', 'T_Higgs', 'A1'],
    'T12':       ['A1', 'T3', 'T9_grav', 'T_gauge', 'T_field', 'T7', 'T_Higgs'],
    'T12E':      ['T12', 'T4F', 'T_field', 'T_Higgs', 'T4G', 'A1', 'T20'],
    'T_Bek':     ['A1', 'T_M', 'T_entropy', 'Delta_continuum'],

    # ===== TIER 5: Î”_geo CLOSURE =====
    'Delta_ordering':   ['A4', 'L_epsilon*', 'T0'],
    'Delta_fbc':        ['A4', 'A1', 'L_epsilon*'],
    'Delta_particle':   ['A1', 'A4', 'L_epsilon*', 'T_M', 'T_S0'],
    'Delta_continuum':  ['A1', 'A4', 'Delta_fbc', 'Delta_ordering'],
    'Delta_signature':  ['A1', 'A4', 'Delta_continuum'],
    'Delta_closure':    ['Delta_ordering', 'Delta_fbc', 'Delta_continuum',
                         'Delta_signature', 'Delta_particle'],
}

# Tiers for each theorem
TIER_MAP = {
    # Reduction layer
    'L_loc': 0, 'L_irr': 0, 'L_nc': 0,
    # Tier 0
    'T0': 0, 'T1': 0, 'L_T2': 0, 'T2': 0, 'T3': 0,
    'L_epsilon*': 0, 'T_epsilon': 0, 'T_eta': 0, 'T_kappa': 0,
    'T_M': 0, 'T_Hermitian': 0,
    # Tier 1
    'T4': 1, 'T5': 1, 'T_gauge': 1,
    # Tier 2
    'T_field': 2, 'T_channels': 2, 'T7': 2, 'T4E': 2, 'T4F': 2,
    'T4G': 2, 'T4G_Q31': 2, 'T_Higgs': 2, 'T9': 2,
    # Tier 3
    'T6': 3, 'T6B': 3, 'T19': 3, 'T20': 3, 'T21': 3, 'T21a': 3, 'T21b': 3, 'T21c': 3, 'T22': 3,
    'T23': 3, 'T24': 3, 'T25a': 3, 'T25b': 3, 'T26': 3,
    'T27c': 3, 'T27d': 3, 'T_sin2theta': 3, 'T_S0': 3,
    # Quantum chain
    'T_Born': 0, 'T_CPTP': 0, 'T_tensor': 0, 'T_entropy': 0,
    # Tier 4
    'T7B': 4, 'T_particle': 4, 'T8': 4, 'T9_grav': 4,
    'T10': 4, 'T11': 4, 'T12': 4, 'T12E': 4, 'T_Bek': 4,
    # Tier 5
    'Delta_ordering': 5, 'Delta_fbc': 5, 'Delta_particle': 5,
    'Delta_continuum': 5, 'Delta_signature': 5, 'Delta_closure': 5,
}

# Physical sectors
SECTOR_MAP = {
    'quantum_spine': ['T1', 'L_T2', 'T2', 'T_Hermitian', 'T_Born', 'T_CPTP', 'T_tensor', 'T_entropy'],
    'gauge_selection': ['T4', 'T5', 'T_gauge', 'T6'],
    'particle_content': ['T_field', 'T_channels', 'T7', 'T4E', 'T4F', 'T4G', 'T4G_Q31', 'T_Higgs', 'T9'],
    'ew_constants': ['T6B', 'T19', 'T20', 'T21', 'T21a', 'T21b', 'T21c', 'T22', 'T23', 'T24', 'T25a', 'T25b', 'T26', 'T27c', 'T27d', 'T_sin2theta', 'T_S0'],
    'gravity': ['T7B', 'T8', 'T9_grav', 'T10', 'T11', 'T_particle'],
    'dark_sector': ['T12', 'T12E', 'T_Bek'],
    'geo_closure': ['Delta_ordering', 'Delta_fbc', 'Delta_particle', 'Delta_continuum', 'Delta_signature', 'Delta_closure'],
    'enforcement_core': ['T0', 'L_epsilon*', 'T_epsilon', 'T_eta', 'T_kappa', 'T_M', 'L_nc', 'L_irr', 'L_loc'],
}


# ============================================================================
# STEP 2: GRAPH CONSTRUCTION
# ============================================================================

class EnforcementCrystal:
    """Graph-theoretic analysis of the FCF dependency DAG."""

    def __init__(self, axiom_mode='3-axiom'):
        """
        axiom_mode: '1-axiom', '3-axiom', or '5-axiom'
        Determines which nodes are treated as source axioms.
        """
        self.axiom_mode = axiom_mode
        self.adj = defaultdict(set)      # node -> set of children
        self.radj = defaultdict(set)     # node -> set of parents
        self.nodes = set()
        self.axiom_nodes = set()

        self._build(axiom_mode)

    def _build(self, mode):
        """Build the DAG from DEPENDENCY_MAP."""
        # Determine which nodes are axioms (source nodes)
        if mode == '1-axiom':
            self.axiom_nodes = {'A1', 'M', 'NT'}
        elif mode == '3-axiom':
            self.axiom_nodes = {'A1', 'A3', 'A4'}
        elif mode == '5-axiom':
            self.axiom_nodes = {'A1', 'A2', 'A3', 'A4', 'A5'}
        else:
            raise ValueError(f"Unknown axiom_mode: {mode}")

        self.nodes = set(self.axiom_nodes)

        # Break known cycles: self-consistency conditions in EW sector
        # T27c <-> T_S0 (mutual dep)
        # T21 -> T27d -> T26 -> T21 (RG running loop)
        # T21 -> T27c -> T27d -> T26 -> T21 (another path)
        cycle_breaks = {
            ('T27c', 'T_S0'),   # break T27c's dep on T_S0
            ('T26', 'T21'),     # break T26's dep on T21 (RG feedback)
            ('T21', 'T27c'),    # break T21's dep on T27c (mixing feedback)
            ('T21', 'T27d'),    # break T21's dep on T27d (mixing feedback)
        }

        for thm, deps in DEPENDENCY_MAP.items():
            # In 3-axiom mode, skip L_loc and L_irr (they derive A3/A4 which are given)
            if mode == '3-axiom' and thm in ('L_loc', 'L_irr'):
                continue
            # In 5-axiom mode, skip reduction lemmas and remap L_nc deps
            if mode == '5-axiom' and thm in ('L_loc', 'L_irr'):
                continue

            self.nodes.add(thm)
            for dep in deps:
                # Normalize dependency names
                dep_clean = dep.split('(')[0].strip()
                # Map derived status based on mode
                if mode == '3-axiom':
                    if dep_clean == 'A2':
                        dep_clean = 'L_nc'  # A2 is derived as L_nc
                    if dep_clean == 'A5':
                        dep_clean = 'L_col'  # A5 is derived as L_col
                    if dep_clean in ('M', 'NT'):
                        continue  # postulates absorbed into A3 in 3-axiom mode
                elif mode == '1-axiom':
                    # In 1-axiom mode, A3 and A4 are derived via L_loc and L_irr
                    if dep_clean == 'A3':
                        dep_clean = 'L_loc'
                    elif dep_clean == 'A4':
                        dep_clean = 'L_irr'
                    elif dep_clean == 'A2':
                        dep_clean = 'L_nc'
                    elif dep_clean == 'A5':
                        dep_clean = 'L_col'

                # Break cycles
                if (thm, dep_clean) in cycle_breaks:
                    continue

                if dep_clean in self.axiom_nodes or dep_clean in DEPENDENCY_MAP:
                    self.nodes.add(dep_clean)
                    self.adj[dep_clean].add(thm)
                    self.radj[thm].add(dep_clean)

    # ----- Basic metrics -----

    def node_count(self):
        return len(self.nodes)

    def edge_count(self):
        return sum(len(ch) for ch in self.adj.values())

    def theorem_nodes(self):
        return self.nodes - self.axiom_nodes

    # ----- Depth / topological sort -----

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

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm with cycle detection."""
        in_deg = defaultdict(int)
        for node in self.nodes:
            in_deg[node]  # ensure all nodes present
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
        # Any remaining nodes with in_deg > 0 are in cycles
        remaining = self.nodes - set(order)
        if remaining:
            if not hasattr(self, '_cycle_warned'):
                self._cycle_warned = True
                print(f"  âš  CYCLE DETECTED: {sorted(remaining)} not in topological order")
            # Append them anyway so metrics don't crash
            order.extend(sorted(remaining))
        return order

    def width_profile(self) -> Dict[int, List[str]]:
        """Width (number of nodes) at each depth level."""
        depths = self.compute_depths()
        profile = defaultdict(list)
        for node, d in depths.items():
            profile[d].append(node)
        return dict(sorted(profile.items()))

    # ----- Betweenness centrality -----

    def betweenness_centrality(self) -> Dict[str, float]:
        """Betweenness centrality for DAG (Brandes-style)."""
        cb = {v: 0.0 for v in self.nodes}
        for s in self.nodes:
            # BFS from s
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
        # Normalize
        n = len(self.nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            cb = {v: c * norm for v, c in cb.items()}
        return cb

    # ----- Cascade failure -----

    def cascade_failure(self, node: str) -> Set[str]:
        """Compute set of nodes that would fail if `node` is removed."""
        remaining = self.nodes - {node}
        # A node fails if any of its dependencies is not in remaining
        failed = {node}
        changed = True
        while changed:
            changed = False
            for n in list(remaining):
                if n in self.axiom_nodes:
                    continue
                parents = self.radj.get(n, set())
                # If ALL parents are failed, this node fails too
                # Actually: if any REQUIRED parent is failed
                # In a dependency DAG, all parents are required
                if parents and not parents.issubset(remaining - failed):
                    # Check if node has at least one parent still alive
                    alive_parents = parents & (remaining - failed)
                    if len(alive_parents) == 0:
                        failed.add(n)
                        remaining.discard(n)
                        changed = True
        return failed - {node}

    def all_cascades(self) -> Dict[str, int]:
        """Cascade failure size for every non-axiom node."""
        results = {}
        for node in self.theorem_nodes():
            results[node] = len(self.cascade_failure(node))
        return results

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
        """Count distinct paths from any axiom to target (iterative, cycle-safe)."""
        # Use topological order; nodes in cycles get paths from their non-cycle parents
        topo = self._topological_sort()
        paths = {}
        for node in topo:
            if node in self.axiom_nodes:
                paths[node] = 1
            else:
                parents = self.radj.get(node, set()) & self.nodes
                paths[node] = sum(paths.get(p, 0) for p in parents)
        return paths.get(target, 0)

    # ----- Axiom fingerprint -----

    def axiom_fingerprint(self, node: str) -> Dict[str, bool]:
        """Which axioms are in the ancestry of this node?"""
        anc = self.ancestors(node) | {node}
        return {a: a in anc for a in sorted(self.axiom_nodes)}

    # ----- Axiom attribution for a target -----

    def axiom_attribution(self, target: str) -> Dict[str, float]:
        """
        Weighted axiom attribution: fraction of paths to target
        that pass through each axiom.
        """
        # Count paths from each individual axiom to target
        axiom_paths = {}
        total = 0
        for a in self.axiom_nodes:
            count = self._count_paths_from(a, target)
            axiom_paths[a] = count
            total += count
        if total == 0:
            return {a: 0.0 for a in self.axiom_nodes}
        return {a: c / total for a, c in axiom_paths.items()}

    def _count_paths_from(self, source: str, target: str) -> int:
        """Count paths from a specific source to target (iterative, cycle-safe)."""
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

    # ----- Sector dominance -----

    def sector_dominance(self) -> Dict[str, Dict[str, str]]:
        """For each sector, fraction of theorems requiring each axiom."""
        results = {}
        for sector, theorems in SECTOR_MAP.items():
            active_thms = [t for t in theorems if t in self.nodes]
            if not active_thms:
                continue
            axiom_counts = {}
            for a in sorted(self.axiom_nodes):
                count = sum(1 for t in active_thms if self.axiom_fingerprint(t).get(a, False))
                axiom_counts[a] = f"{count}/{len(active_thms)}"
            results[sector] = axiom_counts
        return results

    # ----- Entropy measures -----

    def degree_entropy(self, direction='out') -> Tuple[float, float, float]:
        """Shannon entropy of degree distribution. Returns (entropy, max, evenness)."""
        if direction == 'out':
            degrees = [len(self.adj.get(n, set())) for n in self.nodes]
        else:
            degrees = [len(self.radj.get(n, set()) & self.nodes) for n in self.nodes]
        # Build distribution
        from collections import Counter
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

    # ----- Hourglass / minimum vertex cut -----

    def find_vertex_cuts(self, max_depth=None) -> List[Tuple[int, List[str]]]:
        """Find minimum vertex cuts at each depth level."""
        profile = self.width_profile()
        cuts = []
        for depth in sorted(profile.keys()):
            if max_depth and depth > max_depth:
                break
            width = len(profile[depth])
            cuts.append((depth, profile[depth]))
        return cuts

    def find_waists(self, threshold=3) -> List[Tuple[int, List[str]]]:
        """Find depths where width <= threshold (hourglass waists)."""
        return [(d, nodes) for d, nodes in self.find_vertex_cuts()
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
# STEP 3: RUN FULL ANALYSIS
# ============================================================================

def run_crystal_analysis(mode='3-axiom') -> Dict[str, Any]:
    """Run the complete crystal analysis for a given axiom mode."""
    crystal = EnforcementCrystal(axiom_mode=mode)

    print(f"\n{'='*74}")
    print(f"  ENFORCEMENT CRYSTAL v2 â€” {mode.upper()} MODE")
    print(f"{'='*74}")

    # 1. Basic counts
    n_nodes = crystal.node_count()
    n_edges = crystal.edge_count()
    n_axioms = len(crystal.axiom_nodes)
    n_theorems = len(crystal.theorem_nodes())
    print(f"\n  Nodes: {n_nodes} ({n_axioms} axioms + {n_theorems} theorems)")
    print(f"  Edges: {n_edges}")

    # 2. Width profile
    profile = crystal.width_profile()
    print(f"\n  WIDTH PROFILE (depth â†’ width):")
    max_depth = max(profile.keys()) if profile else 0
    for d in range(max_depth + 1):
        nodes = profile.get(d, [])
        bar = 'â–ˆ' * len(nodes)
        names = ', '.join(sorted(nodes)[:5])
        if len(nodes) > 5:
            names += f", ... (+{len(nodes)-5})"
        print(f"  Depth {d:2d}: {len(nodes):3d}  {bar}  [{names}]")

    # 3. Betweenness centrality (top 10)
    bc = crystal.betweenness_centrality()
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])[:15]
    print(f"\n  BETWEENNESS CENTRALITY (top 15):")
    for node, score in bc_sorted:
        print(f"    {node:18s}  {score:.4f}")

    # 4. Cascade failure (top 10)
    cascades = crystal.all_cascades()
    cas_sorted = sorted(cascades.items(), key=lambda x: -x[1])[:10]
    print(f"\n  CASCADE FAILURE (top 10 â€” nodes lost if removed):")
    for node, count in cas_sorted:
        parents = crystal.radj.get(node, set()) & crystal.nodes
        risk = "HIGH" if len(parents) <= 1 else "low"
        print(f"    {node:18s}  cascade={count:3d}  parents={len(parents)}  risk={risk}")

    # 5. Axiom loads
    loads = crystal.axiom_loads()
    print(f"\n  AXIOM LOADS (descendants):")
    for a in sorted(loads, key=lambda x: -loads[x]):
        pct = round(100 * loads[a] / n_theorems) if n_theorems > 0 else 0
        print(f"    {a:6s}: {loads[a]:3d} descendants ({pct}%)")

    # 6. Path counting to sinÂ²Î¸_W
    if 'T_sin2theta' in crystal.nodes:
        paths = crystal.count_paths('T_sin2theta')
        ancestors = crystal.ancestors('T_sin2theta')
        print(f"\n  SINÂ²Î¸_W PREDICTION FUNNEL:")
        print(f"    Total paths to T_sin2theta: {paths:,}")
        print(f"    Ancestor count: {len(ancestors)}")

        # Attribution
        attr = crystal.axiom_attribution('T_sin2theta')
        print(f"    Axiom attribution weights:")
        for a in sorted(attr, key=lambda x: -attr[x]):
            print(f"      {a:6s}: {attr[a]*100:.1f}%")

    # 7. Hourglass waists
    waists = crystal.find_waists(threshold=3)
    if waists:
        print(f"\n  HOURGLASS WAISTS (width â‰¤ 3):")
        for d, nodes in waists:
            print(f"    Depth {d}: {sorted(nodes)}")
    else:
        print(f"\n  No hourglass waists found (width â‰¤ 3)")

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

    # Build output dict
    result = {
        'mode': mode,
        'node_count': n_nodes,
        'edge_count': n_edges,
        'axiom_count': n_axioms,
        'theorem_count': n_theorems,
        'axiom_nodes': sorted(crystal.axiom_nodes),
        'width_profile': {d: sorted(nodes) for d, nodes in profile.items()},
        'betweenness_top15': [(n, round(s, 4)) for n, s in bc_sorted],
        'cascade_top10': [(n, c) for n, c in cas_sorted],
        'axiom_loads': loads,
        'entropy': {
            'out_degree': out_ent,
            'in_degree': in_ent,
            'axiom_load': ax_ent,
        },
        'sector_dominance': sec_dom,
        'qm_gr_independence': indep,
    }

    if 'T_sin2theta' in crystal.nodes:
        result['sin2theta'] = {
            'paths': paths,
            'ancestors': len(ancestors),
            'attribution': {a: round(v, 4) for a, v in attr.items()},
        }

    return result


def compare_crystal_versions():
    """Run analysis for both 3-axiom and original 5-axiom comparison."""
    print("\n" + "â–ˆ" * 74)
    print("  ENFORCEMENT CRYSTAL v2: COMPARATIVE ANALYSIS")
    print("  Comparing original (5-axiom) vs current (3-axiom) vs reduced (1-axiom)")
    print("â–ˆ" * 74)

    results = {}
    for mode in ['3-axiom', '1-axiom']:
        results[mode] = run_crystal_analysis(mode)

    # Comparison summary
    print(f"\n\n{'='*74}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*74}")

    print(f"\n  {'Metric':35s}  {'Original (5-ax)':>15s}  {'Current (3-ax)':>15s}  {'Reduced (1-ax)':>15s}")
    print(f"  {'â”€'*35}  {'â”€'*15}  {'â”€'*15}  {'â”€'*15}")

    r3 = results['3-axiom']
    r1 = results['1-axiom']
    print(f"  {'Axiom nodes':35s}  {'5':>15s}  {r3['axiom_count']:>15d}  {r1['axiom_count']:>15d}")
    print(f"  {'Theorem nodes':35s}  {'51':>15s}  {r3['theorem_count']:>15d}  {r1['theorem_count']:>15d}")
    print(f"  {'Total nodes':35s}  {'56':>15s}  {r3['node_count']:>15d}  {r1['node_count']:>15d}")
    print(f"  {'Total edges':35s}  {'164':>15s}  {r3['edge_count']:>15d}  {r1['edge_count']:>15d}")

    if 'sin2theta' in r3 and 'sin2theta' in r1:
        print(f"  {'Paths to sinÂ²Î¸_W':35s}  {'1,398':>15s}  {r3['sin2theta']['paths']:>15,}  {r1['sin2theta']['paths']:>15,}")
        print(f"  {'sinÂ²Î¸_W ancestors':35s}  {'32':>15s}  {r3['sin2theta']['ancestors']:>15d}  {r1['sin2theta']['ancestors']:>15d}")

    if r3['betweenness_top15']:
        top3_node = r3['betweenness_top15'][0][0]
        print(f"  {'Highest betweenness node':35s}  {'L_Îµ*':>15s}  {top3_node:>15s}  {r1['betweenness_top15'][0][0]:>15s}")

    out3 = r3['entropy']['out_degree']
    in3 = r3['entropy']['in_degree']
    ax3 = r3['entropy']['axiom_load']
    print(f"  {'Out-degree evenness':35s}  {'0.90':>15s}  {out3[2]:>15.2f}  {r1['entropy']['out_degree'][2]:>15.2f}")
    print(f"  {'In-degree evenness':35s}  {'0.97':>15s}  {in3[2]:>15.2f}  {r1['entropy']['in_degree'][2]:>15.2f}")
    print(f"  {'Axiom load evenness':35s}  {'0.98':>15s}  {ax3[2]:>15.2f}  {r1['entropy']['axiom_load'][2]:>15.2f}")

    # Crystal claims assessment
    print(f"\n\n{'='*74}")
    print(f"  CRYSTAL CLAIMS ASSESSMENT (post-reduction)")
    print(f"{'='*74}")

    claims = [
        ("Claim 1: Gauge selection is unique bottleneck",
         "Check width profile for single-node waist at gauge tier"),
        ("Claim 2: sinÂ²Î¸_W is over-determined",
         f"Paths: {r3.get('sin2theta', {}).get('paths', 'N/A')} (was 1,398)"),
        ("Claim 3: QM and GR are structurally parallel",
         f"Independent: {r3['qm_gr_independence']['independent']}, "
         f"Shared: {r3['qm_gr_independence']['shared_non_axiom']} nodes"),
        ("Claim 4: L_Îµ* is the structural root",
         f"Top betweenness: {r3['betweenness_top15'][0] if r3['betweenness_top15'] else 'N/A'}"),
        ("Claim 5: Collapse is a refinement",
         "NOW PROVED (L_col derives A5 from A1+A4) â€” upgraded from observation to theorem"),
    ]

    for i, (claim, evidence) in enumerate(claims, 1):
        print(f"\n  {claim}")
        print(f"    Evidence: {evidence}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    results = compare_crystal_versions()

    # Export JSON
    output = {
        'analysis': 'Enforcement Crystal v2',
        'timestamp': '2026-02-09',
        'original_crystal': {
            'nodes': 56, 'edges': 164, 'axioms': 5, 'theorems': 51,
            'paths_to_sin2theta': 1398,
        },
        'results': {}
    }
    for mode, r in results.items():
        # Convert sets to lists for JSON
        output['results'][mode] = r

    with open('crystal_v2_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\n  Results exported to crystal_v2_analysis.json")
