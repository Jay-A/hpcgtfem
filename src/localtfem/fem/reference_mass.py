import numpy as np


def reference_mass(basis_eval, quad, verts=None):
    """
    Local mass matrix on reference element.

    Parameters
    ----------
    basis_eval : function
        (xi, eta) -> phi[ i ] array of shape (nq, nq, n_modes)

    quad : (R, S, W, XI, ETA)
        from reftriquad

    verts : ignored here (kept for API symmetry)
    """

    R, S, W, XI, ETA = quad

    Phi = basis_eval(XI, ETA)  # (nq, nq, n_modes)
    n_modes = Phi.shape[-1]

    M = np.zeros((n_modes, n_modes))

    # tensor quadrature
    for i in range(n_modes):
        for j in range(n_modes):
            M[i, j] = np.sum(Phi[..., i] * Phi[..., j] * W)

    return M