"""
structured_mesh.py

Structured triangular mesh generator on [-1, 1]^2.

Creates a 2x2 split per quad cell (same topology as MATLAB code),
including edges and boundary classification.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def structured_mesh(level: int) -> Mesh:
    """
    Generate structured triangular mesh.

    Parameters
    ----------
    level : int
        Refinement level (same meaning as MATLAB n).

    Returns
    -------
    Mesh
        Fully assembled mesh object.
    """

    # ------------------------------------------------------------
    # grid resolution
    # ------------------------------------------------------------
    step = 2 ** -(level)
    num_nodes_x = 2 ** (level + 1) + 1
    num_nodes_y = num_nodes_x

    xs = np.linspace(-1.0, 1.0, num_nodes_x)
    ys = np.linspace(-1.0, 1.0, num_nodes_y)

    # ------------------------------------------------------------
    # nodes
    # ------------------------------------------------------------
    nodes = np.array(
        [(x, y) for y in ys for x in xs],
        dtype=float
    )

    # ------------------------------------------------------------
    # elements (two triangles per cell)
    # ------------------------------------------------------------
    elements = []
    areas = []

    def node_id(ix, iy):
        return iy * num_nodes_x + ix

    for iy in range(1, num_nodes_y):
        for ix in range(num_nodes_x - 1):

            n11 = node_id(ix, iy)
            n12 = node_id(ix, iy - 1)
            n21 = node_id(ix + 1, iy - 1)
            n22 = node_id(ix + 1, iy)

            # two triangles (same as MATLAB ordering)
            e1 = (n11, n12, n21)
            e2 = (n11, n21, n22)

            elements.append(e1)
            elements.append(e2)

            # area (constant on structured grid)
            area = 0.5 * step * step
            areas.extend([area, area])

    elements = np.array(elements, dtype=int)
    areas = np.array(areas, dtype=float)

    num_elements = len(elements)

    # ------------------------------------------------------------
    # edges (build unique edge table)
    # ------------------------------------------------------------
    edge_dict = {}
    edges = []
    edge_to_elem = {}

    def add_edge(a, b, elem, sign):
        key = tuple(sorted((a, b)))

        if key not in edge_dict:
            edge_dict[key] = len(edges)
            edges.append([key[0], key[1], 0, 0])
            edge_to_elem[key] = [0, 0]

        idx = edge_dict[key]

        # assign left/right element (MATLAB-style signed storage)
        if edge_to_elem[key][0] == 0:
            edges[idx][2] = elem * sign
            edge_to_elem[key][0] = elem * sign
        else:
            edges[idx][3] = elem * sign
            edge_to_elem[key][1] = elem * sign

    elem_id = 0

    for iy in range(1, num_nodes_y):
        for ix in range(num_nodes_x - 1):

            n11 = node_id(ix, iy)
            n12 = node_id(ix, iy - 1)
            n21 = node_id(ix + 1, iy - 1)
            n22 = node_id(ix + 1, iy)

            e1 = (n11, n12, n21)
            e2 = (n11, n21, n22)

            for tri in (e1, e2):

                # edges of triangle
                edges_local = [
                    (tri[0], tri[1]),
                    (tri[1], tri[2]),
                    (tri[2], tri[0]),
                ]

                for a, b in edges_local:

                    # orientation sign (MATLAB-style)
                    sign = 1 if (a < b) else -1
                    add_edge(a, b, elem_id + 1, sign)

                elem_id += 1

    edges = np.array(edges, dtype=int)

    # ------------------------------------------------------------
    # boundary edges
    # ------------------------------------------------------------
    boundary = np.where(edges[:, 3] == 0)[0]

    # ------------------------------------------------------------
    # return mesh object
    # ------------------------------------------------------------
    return Mesh(
        nodes=nodes,
        elements=elements,
        areas=areas,
        edges=edges,
        boundary=boundary,
    )