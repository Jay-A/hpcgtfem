from __future__ import annotations

import numpy as np


class DOFMap:
    """
    High-order triangular FEM DOF mapping (Python-native version).

    Assumes:
        Nodes   : (N,2)
        Edges   : [nodeA, nodeB, elem_left, elem_right]
        Conn    : (Ne,3)

    Fully 0-based indexing.
    """

    def __init__(self, p: int, Nodes: np.ndarray, Edges: np.ndarray, Conn: np.ndarray):
        self.p = p
        self.Nodes = Nodes
        self.Edges = Edges
        self.Conn = Conn

        self.num_nodes = Nodes.shape[0]
        self.num_edges = Edges.shape[0]
        self.num_elems = Conn.shape[0]

        self.loc_dim = (p + 1) * (p + 2) // 2

        self.map = np.zeros((self.num_elems, self.loc_dim), dtype=int)
        self.sgn = np.ones((self.num_elems, self.loc_dim), dtype=int)

        self._build()

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------

    def element_dofs(self, e: int):
        return self.map[e]

    def element_signs(self, e: int):
        return self.sgn[e]

    def ndofs(self):
        return int(self.map.max() + 1)

    # ------------------------------------------------------------
    # build
    # ------------------------------------------------------------

    def _build(self):
        self._build_vertices()
        self._build_interior()
        self._build_edges()

    # ------------------------------------------------------------
    # vertices
    # ------------------------------------------------------------

    def _build_vertices(self):
        self.map[:, 0:3] = self.Conn[:, 0:3]
        self.sgn[:, 0:3] = 1

    # ------------------------------------------------------------
    # interior DOFs (unchanged structure, 0-based safe)
    # ------------------------------------------------------------

    def _build_interior(self):

        p = self.p
        num_int = (p - 1) * (p - 2) // 2

        if num_int <= 0:
            return

        glob_start = self.num_nodes + (p - 1) * self.num_edges

        for e in range(self.num_elems):
            start = glob_start + e * num_int
            self.map[e, 3 + 3 * (p - 1):] = np.arange(start, start + num_int)
            self.sgn[e, 3 + 3 * (p - 1):] = 1

    # ------------------------------------------------------------
    # edges (FIXED: no MATLAB adjacency assumptions)
    # ------------------------------------------------------------

    def _build_edges(self):

        p = self.p
        if p <= 1:
            return

        # build fast lookup: node pair -> edge index
        edge_lookup = {}

        for ei, (a, b, _, _) in enumerate(self.Edges):
            key = (a, b) if a < b else (b, a)
            edge_lookup[key] = ei

        for e, tri in enumerate(self.Conn):

            v0, v1, v2 = tri

            # triangle edges
            local_edges = [
                (v0, v1),
                (v1, v2),
                (v2, v0),
            ]

            glob_edges = np.zeros(3, dtype=int)
            edge_sign = np.ones(3, dtype=int)

            for i, (a, b) in enumerate(local_edges):

                key = (a, b) if a < b else (b, a)

                ei = edge_lookup.get(key, None)
                if ei is None:
                    raise ValueError(f"Edge not found for ({a},{b})")

                glob_edges[i] = ei

                # orientation sign (pure geometric)
                edge_sign[i] = 1 if (a < b) else -1

            # assign edge DOFs
            for m in range(1, p):

                cols = slice(3 + (m - 1) * 3, 3 + m * 3)

                base = self.num_nodes + (m - 1) * self.num_edges

                self.map[e, cols] = base + glob_edges

                # simple consistent orientation
                self.sgn[e, cols] = edge_sign