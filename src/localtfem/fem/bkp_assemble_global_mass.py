import numpy as np
from scipy.sparse import lil_matrix

def assemble_global_mass(
    p,
    L_mass,
    num_global_dofs,
    num_elems,
    Nodes,
    Elements,
    Areas,
    dofmap
):
    """
    Python equivalent of Create_Global_Mass.m
    """

    L_mass = np.asarray(L_mass, dtype=float)

    GMass = lil_matrix((num_global_dofs, num_global_dofs), dtype=float)

    loc_dim = (p + 1) * (p + 2) // 2

    for e in range(num_elems):

        area = Areas[e] / 2.0

        for i in range(loc_dim):
            I = dofmap.map[e, i]
            si = dofmap.sgn[e, i]

            for j in range(loc_dim):
                J = dofmap.map[e, j]
                sj = dofmap.sgn[e, j]

                GMass[I, J] += area * si * sj * L_mass[i, j]

    return GMass.tocsr()