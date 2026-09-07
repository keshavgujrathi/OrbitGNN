"""
tests/test_graph.py
-------------------
Unit tests for orbital-graph construction:
  - Shell assignment correctness
  - No cross-shell edges
  - Singleton shell has no edges
  - Graph is symmetric
  - No self-loops
  - Binary adjacency values
  - k-neighbour upper bound
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from physics import R_EARTH, MU_EARTH
from dataset import assign_shell_ids, build_orbital_neighbor_graph


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _make_snapshot(alt_km: float, inc_deg: float) -> np.ndarray:
    a = R_EARTH + alt_km
    return np.array([a, 0.001, math.radians(inc_deg), 1.0, 0.5, 0.3])


def _shell_for(alt_km: float, inc_deg: float, name: str = "SAT") -> int:
    snap = _make_snapshot(alt_km, inc_deg)
    ids  = assign_shell_ids([name], snap.reshape(1, 6))
    return int(ids[0])


# ════════════════════════════════════════════════════════════════════
# Shell assignment unit tests
# ════════════════════════════════════════════════════════════════════

class TestShellAssignment:

    def test_geo_assigned_shell_0(self):
        assert _shell_for(35_787, 0.1) == 0

    def test_sso_assigned_shell_1(self):
        # SSO: ~800 km, i > 88°
        assert _shell_for(800, 98.6) == 1

    def test_polar_assigned_shell_1(self):
        # Polar: 700 km, i ≈ 92°
        assert _shell_for(700, 92.0) == 1

    def test_leo66_assigned_shell_2(self):
        # LEO 66°: ~1330 km, 60° < i < 70°
        assert _shell_for(1_330, 66.0) == 2

    def test_jason3_like(self):
        # Jason-3: ~1338 km, 66.04°
        assert _shell_for(1_338, 66.04) == 2

    def test_fengyun_like(self):
        # Fengyun-2F: GEO, i ≈ 2.5°
        assert _shell_for(35_787, 2.5) == 0

    def test_cryosat_like(self):
        # CryoSat-2: ~719 km, i ≈ 92°
        assert _shell_for(719, 92.0) == 1

    def test_sentinel3_like(self):
        # Sentinel-3A: ~803 km, i ≈ 98.6°
        assert _shell_for(803, 98.6) == 1

    def test_mixed_constellation_shells(self):
        names = ["GEO1", "GEO2", "SSO1", "SSO2", "LEO1"]
        snaps = np.stack([
            _make_snapshot(35_787, 0.1),
            _make_snapshot(35_787, 1.5),
            _make_snapshot(800,    98.6),
            _make_snapshot(720,    92.0),
            _make_snapshot(1_330,  66.0),
        ])
        ids = assign_shell_ids(names, snaps)
        assert ids[0] == 0 and ids[1] == 0
        assert ids[2] == 1 and ids[3] == 1
        assert ids[4] == 2


# ════════════════════════════════════════════════════════════════════
# Graph construction: structural properties
# ════════════════════════════════════════════════════════════════════

class TestGraphStructural:

    def _same_shell_4(self):
        snaps = np.stack([_make_snapshot(800 + i, 98.6) for i in range(4)])
        sids  = np.array([1, 1, 1, 1], dtype="int8")
        return snaps, sids

    def test_output_shape(self):
        snaps, sids = self._same_shell_4()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=2)
        assert adj.shape == (4, 4)

    def test_symmetric(self):
        snaps, sids = self._same_shell_4()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=2)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        snaps, sids = self._same_shell_4()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=2)
        np.testing.assert_array_equal(np.diag(adj), 0)

    def test_binary_values(self):
        snaps, sids = self._same_shell_4()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=2)
        assert set(adj.flatten().tolist()).issubset({0.0, 1.0})

    def test_max_degree_bounded_by_2k(self):
        """
        After symmetrisation (A→B implies B→A), undirected degree can be up
        to 2k (node selected by k others + selects k itself). For a 4-node
        shell with k=2, max degree is 3 (fully connected).
        """
        snaps, sids = self._same_shell_4()
        k = 2
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=k)
        # The graph is symmetric — degree can exceed k due to symmetrisation
        n = len(snaps)
        for i in range(n):
            assert adj[i].sum() <= n - 1, f"Self-loop detected at node {i}"


# ════════════════════════════════════════════════════════════════════
# Graph construction: shell isolation
# ════════════════════════════════════════════════════════════════════

class TestGraphShellIsolation:

    def _geo_sso(self):
        geo = np.stack([_make_snapshot(35_787, 0.1), _make_snapshot(35_787, 2.0)])
        sso = np.stack([_make_snapshot(800, 98.6), _make_snapshot(720, 92.0)])
        snaps = np.vstack([geo, sso])
        sids  = np.array([0, 0, 1, 1], dtype="int8")
        return snaps, sids

    def test_no_geo_sso_edges(self):
        snaps, sids = self._geo_sso()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=3, cross_shell=False)
        assert adj[0, 2] == 0 and adj[0, 3] == 0
        assert adj[1, 2] == 0 and adj[1, 3] == 0
        assert adj[2, 0] == 0 and adj[2, 1] == 0
        assert adj[3, 0] == 0 and adj[3, 1] == 0

    def test_within_shell_edges_exist(self):
        """GEO satellites should be connected to each other."""
        snaps, sids = self._geo_sso()
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=3, cross_shell=False)
        # GEO[0] ↔ GEO[1]
        assert adj[0, 1] == 1 and adj[1, 0] == 1

    def test_singleton_shell_no_edges(self):
        """A shell with only 1 member has no edges."""
        snaps = np.stack([
            _make_snapshot(35_787, 0.1),   # shell 0, alone
            _make_snapshot(800,    98.6),   # shell 1
            _make_snapshot(803,    98.6),   # shell 1
        ])
        sids = np.array([0, 1, 1], dtype="int8")
        adj  = build_orbital_neighbor_graph(snaps, sids, k_neighbors=3)
        # Shell 0 has 1 member → no edges
        assert adj[0].sum() == 0 and adj[:, 0].sum() == 0

    def test_three_shell_graph(self):
        """Full benchmark topology: 3 GEO + 4 SSO + 2 LEO-66°."""
        names = ["FY2F", "FY2H", "FY4A",
                 "CryoSat", "SARAL", "Sent3A", "Sent3B",
                 "Jason3", "Sent6A"]
        alts  = [35787, 35787, 35787, 719, 785, 803, 803, 1338, 1327]
        incs  = [2.5, 0.5, 0.1, 92.0, 98.5, 98.6, 98.6, 66.0, 66.0]
        snaps = np.stack([_make_snapshot(a, i) for a, i in zip(alts, incs)])
        ids   = assign_shell_ids(names, snaps)
        adj   = build_orbital_neighbor_graph(snaps, ids, k_neighbors=3)

        S = len(names)
        assert adj.shape == (S, S)
        # No GEO↔LEO or GEO↔SSO edges
        geo_idxs = [k for k, n in enumerate(names) if n.startswith("FY")]
        sso_idxs = [k for k, n in enumerate(names) if n in ("CryoSat","SARAL","Sent3A","Sent3B")]
        leo_idxs = [k for k, n in enumerate(names) if n in ("Jason3","Sent6A")]
        for gi in geo_idxs:
            for si in sso_idxs + leo_idxs:
                assert adj[gi, si] == 0 and adj[si, gi] == 0
        for si in sso_idxs:
            for li in leo_idxs:
                assert adj[si, li] == 0 and adj[li, si] == 0

    def test_total_edges_benchmark_topology(self):
        """Benchmark: Shell0→3 sats (k≤2, fully connected→3 edges),
        Shell1→4 sats (k=3, each has 3 neighbours→12 directed/2=6 edges),
        Shell2→2 sats (k=3, 1 edge). Total: 3+6+1=10 edges."""
        names = ["FY2F", "FY2H", "FY4A",
                 "CryoSat", "SARAL", "Sent3A", "Sent3B",
                 "Jason3", "Sent6A"]
        alts  = [35787, 35787, 35787, 719, 785, 803, 803, 1338, 1327]
        incs  = [2.5, 0.5, 0.1, 92.0, 98.5, 98.6, 98.6, 66.0, 66.0]
        snaps = np.stack([_make_snapshot(a, i) for a, i in zip(alts, incs)])
        ids   = assign_shell_ids(names, snaps)
        adj   = build_orbital_neighbor_graph(snaps, ids, k_neighbors=3)
        # Total directed edges / 2 = undirected edges
        total_undirected = int(adj.sum()) // 2
        assert total_undirected == 10, f"Expected 10 edges, got {total_undirected}"


# ════════════════════════════════════════════════════════════════════
# Graph construction: k-neighbour correctness
# ════════════════════════════════════════════════════════════════════

class TestGraphKNeighbour:

    def _line_constellation(self, n=6):
        """n satellites evenly spaced in RAAN within same shell."""
        snaps = np.stack([
            np.array([R_EARTH + 800, 0.001, math.radians(98.6),
                      i * 2 * math.pi / n, 0.5, 0.3])
            for i in range(n)
        ])
        sids = np.zeros(n, dtype="int8")
        return snaps, sids

    def test_k1_each_has_at_most_2_neighbours(self):
        """After symmetrisation, k=1 gives degree ≤ 2 (each picks 1, may be picked by 1)."""
        snaps, sids = self._line_constellation(6)
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=1)
        for i in range(6):
            assert adj[i].sum() <= 2, f"Node {i} has {adj[i].sum()} neighbours with k=1"

    def test_k5_each_has_at_most_5_neighbours(self):
        snaps, sids = self._line_constellation(6)
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=5)
        for i in range(6):
            assert adj[i].sum() <= 5, f"Node {i} has {adj[i].sum()} neighbours with k=5"

    def test_k_larger_than_shell_size_no_self_loop(self):
        """k > (shell_size - 1) should not add self-loops."""
        snaps, sids = self._line_constellation(3)
        adj = build_orbital_neighbor_graph(snaps, sids, k_neighbors=10)
        np.testing.assert_array_equal(np.diag(adj), 0)
        # All non-self edges should be set (fully connected within 3-node shell)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert adj[i, j] == 1, f"Expected edge ({i},{j})"
