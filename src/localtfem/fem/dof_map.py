from __future__ import annotations

import numpy as np


class DOFMap:
    """
    High-order triangular FEM DOF mapping.

    Edges are stored as

        Edges = {
            "nodes": (num_edges,2),
            "elements": list[list[int]]
        }

    where edge nodes are globally ordered (min node, max node).

    The sign array stores the orientation factor required for edge modes.
    Even edge modes are invariant under edge reversal, while odd edge
    modes change sign.
    """

    def __init__(
        self,
        p: int,
        Nodes: np.ndarray,
        Edges: dict,
        Conn: np.ndarray,
    ):
        self.p = p
        self.Nodes = Nodes
        self.Edges = Edges
        self.Conn = Conn

        self.num_nodes = Nodes.shape[0]
        self.num_edges = Edges["nodes"].shape[0]
        self.num_elems = Conn.shape[0]

        self.loc_dim = (p + 1) * (p + 2) // 2

        self.map = np.zeros((self.num_elems, self.loc_dim), dtype=int)
        self.sgn = np.ones((self.num_elems, self.loc_dim), dtype=int)

        self._build()

    # ============================================================
    # API
    # ============================================================

    def element_dofs(self, e: int):
        return self.map[e]

    def element_signs(self, e: int):
        return self.sgn[e]

    def ndofs(self):
        return int(self.map.max() + 1)

    # ============================================================
    # build
    # ============================================================

    def _build(self):
        self._build_vertices()
        self._build_interior()
        self._build_edges()

    # ============================================================
    # vertices
    # ============================================================

    def _build_vertices(self):

        self.map[:, :3] = self.Conn
        self.sgn[:, :3] = 1

    # ============================================================
    # interior modes
    # ============================================================

    def _build_interior(self):

        p = self.p

        num_int = (p - 1) * (p - 2) // 2

        if num_int == 0:
            return

        start_global = self.num_nodes + (p - 1) * self.num_edges

        first_local = 3 + 3 * (p - 1)

        for e in range(self.num_elems):

            first = start_global + e * num_int

            self.map[e, first_local:] = np.arange(first, first + num_int)
            self.sgn[e, first_local:] = 1

    # ============================================================
    # edge modes
    # ============================================================

    def _build_edges(self):

        p = self.p

        if p <= 1:
            return

        edge_nodes = self.Edges["nodes"]

        # canonical lookup
        edge_lookup = {
            (a, b): edge_id
            for edge_id, (a, b) in enumerate(edge_nodes)
        }

        for e, tri in enumerate(self.Conn):

            A, B, C = tri

            local_edges = [
                (A, B),
                (B, C),
                (C, A),
            ]

            global_edges = np.zeros(3, dtype=int)
            edge_orientation = np.ones(3, dtype=int)

            # ---------------------------------------------
            # identify global edge and orientation
            # ---------------------------------------------
            for k, (i, j) in enumerate(local_edges):

                key = (i, j) if i < j else (j, i)

                if key not in edge_lookup:
                    raise ValueError(f"Missing edge {key}")

                global_edges[k] = edge_lookup[key]

                # local orientation relative to canonical edge
                edge_orientation[k] = 1 if i < j else -1

            # ---------------------------------------------
            # assign edge DOFs
            # ---------------------------------------------
            for m in range(1, p):

                cols = slice(
                    3 + (m - 1) * 3,
                    3 + m * 3,
                )

                base = self.num_nodes + (m - 1) * self.num_edges

                self.map[e, cols] = base + global_edges

                # -----------------------------------------
                # parity rule:
                #
                # m=1  quadratic edge mode   -> even
                # m=2  cubic edge mode       -> odd
                # m=3  quartic edge mode     -> even
                # ...
                # -----------------------------------------

                parity = (m - 1) % 2

                self.sgn[e, cols] = edge_orientation ** parity