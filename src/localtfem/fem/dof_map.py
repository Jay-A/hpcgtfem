from __future__ import annotations

import numpy as np


class DOFMap:
    """
    High-order triangular FEM DOF mapping.

    Python equivalent of MATLAB Create_Map.m

    Produces:
        map[e, i]  -> global DOF index
        sgn[e, i]  -> orientation sign (+1 or -1)

    Ordering:
        1–3: vertex modes
        4–3+3(p-1): edge modes (A, B, C per edge level)
        interior modes last
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
    # public helpers
    # ------------------------------------------------------------

    def element_dofs(self, e: int):
        return self.map[e]

    def element_signs(self, e: int):
        return self.sgn[e]

    def ndofs(self):
        return int(self.map.max() + 1)

    # ------------------------------------------------------------
    # core construction
    # ------------------------------------------------------------

    def _build(self):
        self._build_vertices()
        self._build_interior()
        self._build_edges()

    # ------------------------------------------------------------
    # vertex DOFs (trivial mapping)
    # ------------------------------------------------------------

    def _build_vertices(self):
        # local vertices -> global nodes
        self.map[:, 0:3] = self.Conn[:, 0:3]
        self.sgn[:, 0:3] = 1

    # ------------------------------------------------------------
    # interior DOFs (block-wise numbering per element)
    # ------------------------------------------------------------

    def _build_interior(self):
        p = self.p
        num_int = (p - 1) * (p - 2) // 2

        if num_int <= 0:
            return

        glob_int_start = self.num_nodes + (p - 1) * self.num_edges

        for e in range(self.num_elems):
            start = glob_int_start + e * num_int
            self.map[e, 3 + 3 * (p - 1):] = np.arange(start, start + num_int)
            self.sgn[e, 3 + 3 * (p - 1):] = 1

    # ------------------------------------------------------------
    # edge DOFs (MOST IMPORTANT PART)
    # ------------------------------------------------------------

    def _build_edges(self):
        p = self.p

        if p <= 1:
            return

        # local vertex indices
        for e in range(self.num_elems):

            v = self.map[e, 0:3]

            # ----------------------------------------------------
            # find edges belonging to this element
            # Edges format assumed:
            #   Edges[:,0:2] = node endpoints
            #   Edges[:,2:4] = adjacent elements (signed)
            # ----------------------------------------------------

            elem_edges = np.where(
                (np.abs(self.Edges[:, 2]) == e) |
                (np.abs(self.Edges[:, 3]) == e)
            )[0]

            if len(elem_edges) != 3:
                raise ValueError(f"Element {e} does not have 3 edges found.")

            # ----------------------------------------------------
            # match local vertices to edges
            # ----------------------------------------------------

            glob_edges = np.zeros(3, dtype=int)
            edge_sign = np.ones(3, dtype=int)

            for i in range(3):
                vi = v[i]

                # find edge containing this vertex pairing
                match = None

                for ed in elem_edges:
                    if self.Edges[ed, 0] == vi or self.Edges[ed, 1] == vi:
                        match = ed
                        break

                if match is None:
                    raise ValueError("Edge matching failed.")

                glob_edges[i] = match

                # orientation sign (MATLAB equivalent)
                if self.Edges[match, 2] == e:
                    edge_sign[i] = 1
                else:
                    edge_sign[i] = -1

            # ----------------------------------------------------
            # assign edge DOFs
            # ----------------------------------------------------

            for m in range(1, p):

                base = 3 + (m - 1) * 3

                isodd = (m - 1) % 2

                cols = slice(base, base + 3)

                self.map[e, cols] = self.num_nodes + (m - 1) * self.num_edges + glob_edges

                # MATLAB: edge_sign.^isodd
                self.sgn[e, cols] = edge_sign ** isodd