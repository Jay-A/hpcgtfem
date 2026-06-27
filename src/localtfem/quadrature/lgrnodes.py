import numpy as np


def lgrnodes(N):
    """
    Legendre-Gauss-Radau nodes and weights on [-1, 1].

    Matches MATLAB implementation by Greg von Winckel.

    Nodes are zeros of:
        P_N(x) + P_{N+1}(x)
    with endpoint x = -1 included.
    """

    N1 = N + 1

    # Chebyshev-Gauss-Radau initial guess
    x = -np.cos(2 * np.pi * np.arange(N1) / (2 * N + 1)).astype(float)

    P = np.zeros((N1, N1 + 1), dtype=float)

    xold = np.full_like(x, 2.0)

    free = np.arange(1, N1)  # MATLAB: 2:N1

    eps = np.finfo(float).eps

    while np.max(np.abs(x - xold)) > eps:

        xold = x.copy()

        # P_0 at all points (including boundary)
        P[0, :] = (-1) ** np.arange(N1 + 1)

        # interior nodes only
        P[free, 0] = 1.0
        P[free, 1] = x[free]

        for k in range(2, N1 + 1):
            P[free, k] = (
                (2 * k - 1) * x[free] * P[free, k - 1]
                - (k - 1) * P[free, k - 2]
            ) / k

        # Newton update (Radau correction)
        x[free] = xold[free] - (
            (1 - xold[free]) / N1
        ) * (
            (P[free, N1] + P[free, N1 - 1])
            / (P[free, N1] - P[free, N1 - 1])
        )

    # Final Vandermonde (trim last column like MATLAB)
    P = P[:, :N1]

    # Weights
    w = np.zeros(N1, dtype=float)

    w[0] = 2.0 / (N1 ** 2)

    w[1:] = (
        (1 - x[1:])
        / (N1 * (P[1:, N1 - 1] ** 2))
    )

    return x, w