import numpy as np


def reference_stiffness(basis_grad, quad, jac_det):
    """
    Local stiffness matrix:

        ∫ ∇φ_i · ∇φ_j dxdy
    """

    R, S, W, XI, ETA = quad

    dphi_x, dphi_y = basis_grad(XI, ETA)  
    # each: (nq, nq, n_modes)

    n_modes = dphi_x.shape[-1]

    K = np.zeros((n_modes, n_modes))

    for i in range(n_modes):
        for j in range(n_modes):

            integrand = (
                dphi_x[..., i] * dphi_x[..., j] +
                dphi_y[..., i] * dphi_y[..., j]
            )

            K[i, j] = np.sum(integrand * W * jac_det)

    return K