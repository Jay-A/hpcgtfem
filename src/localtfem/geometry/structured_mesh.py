"""
structured_mesh.py

Fully Python-native structured triangular mesh on [-1,1]^2.

Key properties:
- 0-based indexing everywhere
- edges store: [nodeA, nodeB, elem_left, elem_right]
- DOFMap-compatible WITHOUT MATLAB assumptions
"""

from __future__ import annotations

import numpy as np
from .mesh import Mesh


def structured_mesh(level: int) -> Mesh:

    # ------------------------------------------------------------
    # grid
    # ------------------------------------------------------------
    num_nodes_x = 2 ** (level + 1) + 1
    num_nodes_y = num_nodes_x

    xs = np.linspace(-1.0, 1.0, num_nodes_x)
    ys = np.linspace(-1.0, 1.0, num_nodes_y)

    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def node_id(ix, iy):
        return iy * num_nodes_x + ix

    # ------------------------------------------------------------
    # elements
    # ------------------------------------------------------------
    elements = []
    areas = []

    step = 2.0 / (num_nodes_x - 1)

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

            area = 0.5 * step * step
            areas.extend([area, area])

    elements = np.array(elements, dtype=int)

    # ------------------------------------------------------------
    # edges (0-based adjacency)
    # ------------------------------------------------------------
    edge_dict = {}
    edges = []

    def add_edge(a, b, elem):

        key = (a, b) if a < b else (b, a)

        if key not in edge_dict:
            edge_dict[key] = len(edges)
            edges.append([key[0], key[1], elem, -1])  # -1 = boundary placeholder
        else:
            idx = edge_dict[key]
            edges[idx][3] = elem  # second element

    # ------------------------------------------------------------
    # build edges
    # ------------------------------------------------------------
    for elem_id, tri in enumerate(elements):

        edges_local = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0]),
        ]

        for a, b in edges_local:
            add_edge(a, b, elem_id)

    edges = np.array(edges, dtype=int)

    # boundary edges: only one adjacent element
    boundary = np.where(edges[:, 3] == -1)[0]

    return Mesh(
        nodes=nodes,
        elements=elements,
        areas=np.array(areas, dtype=float),
        edges=edges,
        boundary=boundary,
    )