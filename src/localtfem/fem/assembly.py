"""
assembly.py

Generic global assembly routines.

These routines assemble arbitrary local element matrices and vectors
using a DOFMap describing the local-to-global numbering and orientation
of basis functions.

The same routines may be used for

    - Mass matrices
    - Stiffness matrices
    - Convection matrices
    - Load vectors
    - Boundary contributions

provided the appropriate local operators are supplied.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ---------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------

def assemble_global_matrix(
    local_matrix,
    areas,
    dofmap,
    num_global_dofs=None,
):
    """
    Assemble a global sparse matrix from a reference-element matrix.

    Parameters
    ----------
    local_matrix : (n,n) ndarray
        Reference element matrix (mass, stiffness, etc.).

    areas : (num_elements,) array_like
        Physical triangle areas.

    dofmap : DOFMap
        Local-to-global mapping.

    num_global_dofs : int, optional
        Number of global degrees of freedom.
        If omitted, inferred from dofmap.

    Returns
    -------
    scipy.sparse.csr_matrix
    """

    local_matrix = np.asarray(local_matrix, dtype=float)

    loc_dim = local_matrix.shape[0]
    num_elems = dofmap.map.shape[0]

    if num_global_dofs is None:
        num_global_dofs = int(dofmap.map.max()) + 1

    A = lil_matrix((num_global_dofs, num_global_dofs), dtype=float)

    for e in range(num_elems):

        scale = areas[e] / 2.0

        for i in range(loc_dim):

            I = dofmap.map[e, i]
            si = dofmap.sgn[e, i]

            for j in range(loc_dim):

                J = dofmap.map[e, j]
                sj = dofmap.sgn[e, j]

                A[I, J] += scale * si * sj * local_matrix[i, j]

    return A.tocsr()


# ---------------------------------------------------------------------
# Vector assembly
# ---------------------------------------------------------------------

def assemble_global_vector(
    local_vectors,
    dofmap,
    num_global_dofs=None,
):
    """
    Assemble a global vector from element vectors.

    Parameters
    ----------
    local_vectors : ndarray
        Shape (num_elements, local_dimension)

    dofmap : DOFMap

    num_global_dofs : int, optional

    Returns
    -------
    ndarray
    """

    local_vectors = np.asarray(local_vectors, dtype=float)

    num_elems, loc_dim = local_vectors.shape

    if num_global_dofs is None:
        num_global_dofs = int(dofmap.map.max()) + 1

    b = np.zeros(num_global_dofs)

    for e in range(num_elems):

        for i in range(loc_dim):

            I = dofmap.map[e, i]
            s = dofmap.sgn[e, i]

            b[I] += s * local_vectors[e, i]

    return b


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def assemble_global_system(
    local_matrix,
    local_vectors,
    areas,
    dofmap,
    num_global_dofs=None,
):
    """
    Assemble a linear system

        A x = b

    from a reference local matrix and element load vectors.

    Returns
    -------
    A : csr_matrix
    b : ndarray
    """

    A = assemble_global_matrix(
        local_matrix,
        areas,
        dofmap,
        num_global_dofs,
    )

    b = assemble_global_vector(
        local_vectors,
        dofmap,
        num_global_dofs,
    )

    return A, b