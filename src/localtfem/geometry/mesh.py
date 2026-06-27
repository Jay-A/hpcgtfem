"""
mesh.py

Mesh data structure for triangular finite element meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Mesh:
    nodes: np.ndarray
    elements: np.ndarray
    areas: np.ndarray

    edges: Optional[np.ndarray] = None
    boundary: Optional[np.ndarray] = None
    """
    Triangular finite element mesh.

    Parameters
    ----------
    nodes
        (N,2) array of vertex coordinates.

    elements
        (E,3) array of vertex indices.

    areas
        (E,) element areas.

    edges
        (M,4) edge table

            [node1, node2, elem_left, elem_right]

        where elem_left/elem_right are signed exactly as in the
        original MATLAB implementation.

    boundary
        Indices of boundary edges.
    """

    nodes: np.ndarray
    elements: np.ndarray
    areas: np.ndarray

    edges: np.ndarray | None = None
    boundary: np.ndarray | None = None

    # ------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def num_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def num_edges(self) -> int:
        if self.edges is None:
            return 0
        return self.edges.shape[0]

    # ------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self.nodes.shape[1]

    # ------------------------------------------------------------

    def vertices(self, element: int) -> np.ndarray:
        """
        Coordinates of one element.

        Returns
        -------
        (3,2) ndarray
        """
        return self.nodes[self.elements[element]]

    # ------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_elements

    # ------------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"Mesh("
            f"nodes={self.num_nodes}, "
            f"elements={self.num_elements}, "
            f"edges={self.num_edges})"
        )