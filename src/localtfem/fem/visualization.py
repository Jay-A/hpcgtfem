import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from localtfem.modified_dubiner.dubiner_sym import dubiner_basis_lambdified
from localtfem.quadrature.lglnodes import lglnodes
from localtfem.quadrature.lgrnodes import lgrnodes

from localtfem.geometry.reference_triangle import (
    square_to_triangle,
    reference_to_physical,
)


def plot_solution(mesh, dofmap, u, p, resolution=20, zlim=None, element=None):
    """
    Plot FEM solution with globally consistent color scaling.
    """

    basis = dubiner_basis_lambdified(p)

    # ------------------------------------------------------------
    # Collapsed quadrature grid
    # ------------------------------------------------------------
    xi, _ = lglnodes(resolution)
    eta, _, _ = lgrnodes(resolution)

    XI, ETA = np.meshgrid(xi, eta, indexing="xy")

    # ------------------------------------------------------------
    # Evaluate basis on collapsed coordinates
    # ------------------------------------------------------------
    Phi = np.stack([phi(XI, ETA) for phi in basis], axis=-1)

    # ------------------------------------------------------------
    # Map to reference triangle
    # ------------------------------------------------------------
    R, S = square_to_triangle(XI, ETA)

    # ------------------------------------------------------------
    # GLOBAL COLOR LIMITS (vertex-based as you decided)
    # ------------------------------------------------------------

    # collect vertex DOFs (robust because your basis is vertex-normalized)
    # assumes first 3 DOFs correspond to vertices in each element
    vertex_vals = []

    for e in range(mesh.num_elements):
        dofs = dofmap.element_dofs(e)
        vertex_vals.extend(u[dofs[:3]])

    vertex_vals = np.array(vertex_vals)

    if zlim is not None:
        vmin, vmax = zlim
    else:
        vmin = np.min(vertex_vals)
        vmax = np.max(vertex_vals)

        pad = 0.0#max(0.5, 0.01 * (vmax - vmin))
        vmin -= pad
        vmax += pad

    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 41)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    mappable = None

    if element is None:
        elements = range(mesh.num_elements)
    else:
        if not (0 <= element < mesh.num_elements):
            raise ValueError(
                f"Invalid element {element}; mesh has {mesh.num_elements} elements."
            )
        elements = [element]

    for e in elements:

        verts = mesh.vertices(e)
        dofs = dofmap.element_dofs(e)

        # physical coordinates
        X, Y = reference_to_physical(R, S, verts)

        # solution reconstruction
        U = np.sum(Phi * u[dofs][None, None, :], axis=2)

        mappable = ax.contourf(
            X,
            Y,
            U,
            levels=levels,
            cmap="viridis",
            norm=norm,
        )

        # element boundary
        poly = np.vstack([verts, verts[0]])
        ax.plot(poly[:, 0], poly[:, 1], "k-", lw=0.1)

        # debug mode
        if element is not None:

            ax.scatter(X.ravel(), Y.ravel(), s=8, c="k", alpha=0.4)

            labels = ["A", "B", "C"]
            for label, (x, y) in zip(labels, verts):
                ax.text(
                    x, y, label,
                    color="red",
                    fontsize=12,
                    ha="center",
                    va="center",
                )

    ax.set_aspect("equal")

    ax.set_title(
        "FEM Solution" if element is None
        else f"FEM Solution (Element {element})"
    )

    if mappable is not None:
        fig.colorbar(mappable, ax=ax)

    plt.tight_layout()
    plt.show()