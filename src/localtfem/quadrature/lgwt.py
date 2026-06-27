import numpy as np


def lgwt(N, a, b):
    """
    Legendre-Gauss nodes and weights on [a, b].

    Direct translation of Greg von Winckel's MATLAB implementation.
    """

    N = N - 1
    N1 = N + 1
    N2 = N + 2

    xu = np.linspace(-1, 1, N1).reshape(-1, 1)

    # Initial guess
    y = (
        np.cos((2 * np.arange(N1) + 1) * np.pi / (2 * N + 2))
        + (0.27 / N1) * np.sin(np.pi * xu.flatten() * N / N2)
    )

    L = np.zeros((N1, N2), dtype=float)
    Lp = np.zeros((N1, N2), dtype=float)

    y0 = np.full_like(y, 2.0)

    eps = np.finfo(float).eps

    # Newton iteration
    while np.max(np.abs(y - y0)) > eps:

        L[:, 0] = 1.0
        Lp[:, 0] = 0.0

        L[:, 1] = y
        Lp[:, 1] = 1.0

        for k in range(2, N1 + 1):
            L[:, k] = (
                (2 * k - 1) * y * L[:, k - 1]
                - (k - 1) * L[:, k - 2]
            ) / k

        # derivative via identity
        Lp = N2 * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y**2)

        y0 = y.copy()
        y = y0 - L[:, N2 - 1] / Lp

    # Map to [a, b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Weights
    w = (
        (b - a)
        / ((1 - y**2) * (Lp**2))
        * (N2 / N1) ** 2
    )

    return x, w