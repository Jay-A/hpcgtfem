"""
structured_mesh.py

Structured triangular mesh on [-1,1]^2.

CLEAN TOPOLOGY FORMAT (matches updated Mesh):

edges:
    nodes: (M,2)
    elements: adjacency list (list of element indices)

No signed integers.
No boundary sentinels.
Boundary is inferred from adjacency size.
"""

from __future__ import annotations

import numpy as np
from .mesh import Mesh


def structured_mesh(level: int) -> Mesh:

    # ============================================================
    # GRID
    # ============================================================

    num_nodes_x = 2 ** (level + 1) + 1
    num_nodes_y = num_nodes_x

    xs = np.linspace(-1.0, 1.0, num_nodes_x)
    ys = np.linspace(-1.0, 1.0, num_nodes_y)

    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def node_id(ix, iy):
        return iy * num_nodes_x + ix

    # ============================================================
    # ELEMENTS
    # ============================================================

    elements = []
    areas = []

    h = 2.0 / (num_nodes_x - 1)

    for iy in range(1, num_nodes_y):
        for ix in range(num_nodes_x - 1):

            n11 = node_id(ix, iy)
            n12 = node_id(ix, iy - 1)
            n21 = node_id(ix + 1, iy - 1)
            n22 = node_id(ix + 1, iy)

            e1 = (n11, n12, n21)
            e2 = (n11, n21, n22)

            elements.append(e1)
            elements.append(e2)

            area = 0.5 * h * h
            areas.extend([area, area])

    elements = np.asarray(elements, dtype=int)
    areas = np.asarray(areas, dtype=float)

    num_elements = elements.shape[0]

    # ============================================================
    # EDGE CONSTRUCTION (PURE TOPOLOGY)
    # ============================================================

    edge_dict = {}
    edge_nodes = []
    edge_adj = []

    def add_edge(a, b, elem_id):

        key = (a, b) if a < b else (b, a)

        if key not in edge_dict:
            edge_dict[key] = len(edge_nodes)
            edge_nodes.append([key[0], key[1]])
            edge_adj.append([elem_id])
        else:
            idx = edge_dict[key]
            edge_adj[idx].append(elem_id)

    for elem_id, tri in enumerate(elements):

        local_edges = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0]),
        ]

        for a, b in local_edges:
            add_edge(a, b, elem_id)

    edges = {
        "nodes": np.asarray(edge_nodes, dtype=int),
        "elements": edge_adj,  # list of adjacency lists
    }

    # ============================================================
    # BOUNDARY (NOT STORED, DERIVED IN MESH)
    # ============================================================

    return Mesh(
        nodes=nodes,
        elements=elements,
        areas=areas,
        edges=edges,
        boundary=None,
    )