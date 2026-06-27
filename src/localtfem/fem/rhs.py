import numpy as np


def eval_rhs(func, verts, area, basis_eval, quad_x, quad_w):
    """
    Python equivalent of Eval_RHS.m

    Parameters
    ----------
    func : callable (x,y)
    verts : (3,2) triangle vertices
    area  : float
    basis_eval : function (xi, eta) -> (n_modes,)
    quad_x : LGL nodes
    quad_w : LGL weights
    """

    # ------------------------------------------------------------
    # affine map (reference triangle)
    # ------------------------------------------------------------

    def Xmap(r, s):
        return 0.5 * (verts[2,0] - verts[1,0]) * r \
             + 0.5 * (verts[0,0] - verts[1,0]) * s \
             + 0.5 * (verts[0,0] + verts[2,0])

    def Ymap(r, s):
        return 0.5 * (verts[2,1] - verts[1,1]) * r \
             + 0.5 * (verts[0,1] - verts[1,1]) * s \
             + 0.5 * (verts[0,1] + verts[2,1])

    # ------------------------------------------------------------
    # tensor-product quadrature
    # ------------------------------------------------------------

    XI, ETA = np.meshgrid(quad_x, quad_x, indexing="xy")
    W = np.outer(quad_w, quad_w)

    # collapse map
    R = 0.5 * (XI + 1) * (1 - ETA) - 1
    S = ETA

    # physical coordinates
    X = Xmap(R, S)
    Y = Ymap(R, S)

    # forcing
    F = func(X, Y)

    # ------------------------------------------------------------
    # basis evaluation
    # ------------------------------------------------------------

    Phi = basis_eval(XI, ETA)  # shape (nq, nq, n_modes)

    # ------------------------------------------------------------
    # Jacobian factor (collapsed triangle)
    # ------------------------------------------------------------

    jac = (area / 4.0) * (1 - ETA)

    # ------------------------------------------------------------
    # RHS assembly
    # ------------------------------------------------------------

    n_modes = Phi.shape[-1]
    rhs = np.zeros(n_modes)

    for i in range(n_modes):
        rhs[i] = np.sum(F * Phi[..., i] * W * jac)

    return rhs