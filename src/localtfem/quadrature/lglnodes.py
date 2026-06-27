import numpy as np


def lglnodes(N):
    """
    Legendre-Gauss-Lobatto nodes and weights on [-1, 1].

    This is a direct translation of the MATLAB implementation by
    Greg von Winckel (spectral methods).
    """

    N1 = N + 1

    # Initial guess: Chebyshev-Gauss-Lobatto nodes
    x = np.cos(np.pi * np.arange(N1) / N).astype(float)

    # Vandermonde matrix
    P = np.zeros((N1, N1), dtype=float)

    xold = 2.0

    eps = np.finfo(float).eps

    # Newton iteration
    while np.max(np.abs(x - xold)) > eps:

        xold = x.copy()

        P[:, 0] = 1.0
        P[:, 1] = x

        for k in range(2, N + 1):
            P[:, k] = (
                (2 * k - 1) * x * P[:, k - 1]
                - (k - 1) * P[:, k - 2]
            ) / k

        x = xold - (x * P[:, N] - P[:, N - 1]) / (N1 * P[:, N])

    # Weights
    w = 2.0 / (N * N1 * (P[:, N] ** 2))

    # Sort (MATLAB compatibility)
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    return x, w