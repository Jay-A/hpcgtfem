"""
mesh.py

Clean topology-based triangular FEM mesh.

Edge model:
    edges connect nodes only.
    adjacency defines elements.
    no signed encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Mesh:
    """
    Triangular finite element mesh.

    Parameters
    ----------
    nodes : (N,2)
        Node coordinates.

    elements : (E,3)
        Triangle connectivity.

    areas : (E,)
        Element areas.

    edges : dict-like optional
        Expected structure (recommended):

            edges.nodes     -> (M,2)
            edges.elements  -> list of lists (adjacency)

        Example:
            edge k:
                nodes: (i,j)
                elements: [e0, e1] or [e0] for boundary
    """

    nodes: np.ndarray
    elements: np.ndarray
    areas: np.ndarray

    edges: Optional[dict] = None
    boundary: Optional[np.ndarray] = None  # computed, not stored raw

    # ============================================================
    # BASIC PROPERTIES
    # ============================================================

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
        return len(self.edges["nodes"])

    @property
    def dimension(self) -> int:
        return self.nodes.shape[1]

    def __len__(self) -> int:
        return self.num_elements

    def __repr__(self) -> str:
        return (
            f"Mesh("
            f"nodes={self.num_nodes}, "
            f"elements={self.num_elements}, "
            f"edges={self.num_edges})"
        )

    # ============================================================
    # GEOMETRY HELPERS
    # ============================================================

    def vertices(self, e: int) -> np.ndarray:
        return self.nodes[self.elements[e]]

    def centroid(self, e: int) -> np.ndarray:
        return self.vertices(e).mean(axis=0)

    def element_edges(self, e: int):
        tri = self.elements[e]
        return [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0]),
        ]

    # ============================================================
    # EDGE QUERY HELPERS (NEW CORE IDEA)
    # ============================================================

    def edge_elements(self, edge_id: int) -> List[int]:
        """
        Return adjacent elements of edge.
        """
        if self.edges is None:
            return []
        return self.edges["elements"][edge_id]

    def is_boundary_edge(self, edge_id: int) -> bool:
        return len(self.edge_elements(edge_id)) == 1

    def boundary_edges(self) -> np.ndarray:
        return np.array([
            i for i in range(self.num_edges)
            if self.is_boundary_edge(i)
        ], dtype=int)

    # ============================================================
    # TOPLOGY PLOTTING
    # ============================================================

    def plot_topology(
        self,
        *,
        node_ids: bool = True,
        element_ids: bool = True,
        local_labels: bool = False,
        local_edge_arrows: bool = False, 
        edge_arrows: bool = False,
        figsize=(6, 6),
    ):
        fig, ax = plt.subplots(figsize=figsize)

        self._draw_elements(ax)
        self._draw_nodes(ax)

        if node_ids:
            self._draw_node_ids(ax)

        if element_ids:
            self._draw_element_ids(ax)

        if local_labels:
            self._draw_local_labels(ax)

        if local_edge_arrows:
            self._draw_local_element_arrows(ax)            

        if edge_arrows:
            self._draw_edge_arrows(ax)

        ax.set_aspect("equal")
        ax.set_title("Mesh Topology")

        return fig, ax

    # ============================================================
    # DRAWING
    # ============================================================

    def _draw_elements(self, ax):
        for e in range(self.num_elements):
            tri = self.vertices(e)
            poly = np.vstack([tri, tri[0]])
            ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.8)

    def _draw_nodes(self, ax):
        ax.scatter(
            self.nodes[:, 0],
            self.nodes[:, 1],
            s=200,
            c="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

    def _draw_node_ids(self, ax):
        for i, (x, y) in enumerate(self.nodes):
            ax.text(x, y, str(i),
                    ha="center", va="center",
                    fontsize=8, color="blue")

    def _draw_element_ids(self, ax):
        for e in range(self.num_elements):
            c = self.centroid(e)
            ax.text(c[0], c[1], str(e),
                    ha="center", va="center",
                    fontsize=9, color="red")

    def _draw_local_labels(self, ax):
        labels = ["A", "B", "C"]

        for e in range(self.num_elements):
            tri = self.vertices(e)
            c = tri.mean(axis=0)

            for i, (x, y) in enumerate(tri):
                pos = 0.75 * np.array([x, y]) + 0.25 * c
                ax.text(pos[0], pos[1], labels[i],
                        fontsize=10, color="green",
                        ha="center", va="center")

    def _draw_local_element_arrows(self, ax):
        """
        Draw LOCAL element orientation:
            A → B → C → A
    
        This is purely visual and independent of global edge structure.
        """
    
        for e in range(self.num_elements):
    
            tri = self.vertices(e)
    
            A, B, C = tri
    
            # edges of reference ordering
            edges = [
                (A, B),
                (B, C),
                (C, A),
            ]
    
            for p0, p1 in edges:
    
                mid = 0.5 * (p0 + p1)
    
                t = p1 - p0
                t = t / (np.linalg.norm(t) + 1e-14)
    
                # inward offset (toward centroid)
                c = tri.mean(axis=0)
    
                n = np.array([-t[1], t[0]])
    
                if np.dot(n, c - mid) < 0:
                    n = -n
    
                offset = 0.05 * n
    
                start = mid - 0.10 * t + offset
                end   = mid + 0.10 * t + offset
    
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=1,
                        color="green",
                    ),
                )                

    def _draw_edge_arrows(self, ax):
        """
        GLOBAL edge connectivity visualization.
    
        Draws arrows directly on edges:
            node_i → node_j (sorted order)
    
        No offsets, no element bias.
        Pure topology visualization.
        """
    
        if self.edges is None:
            return
    
        edge_nodes = self.edges["nodes"]
    
        for n1, n2 in edge_nodes:
    
            p0 = self.nodes[n1]
            p1 = self.nodes[n2]
    
            # direction along edge
            t = p1 - p0
            L = np.linalg.norm(t) + 1e-14
            t = t / L
    
            # shorten arrow so it doesn't overlap nodes
            shrink = 0.15
    
            start = p0 + shrink * t
            end   = p1 - shrink * t
    
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1,
                    color="gray",
                ),
            )