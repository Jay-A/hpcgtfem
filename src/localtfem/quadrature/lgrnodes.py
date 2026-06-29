import numpy as np


def lgrnodes(N):
    """
    Legendre-Gauss-Radau nodes, weights, and Vandermonde matrix.

    Literal translation of Greg von Winckel's MATLAB lgrnodes.m.

    Parameters
    ----------
    N : int
        Polynomial degree.

    Returns
    -------
    x : (N+1,) ndarray
        LGR nodes (includes x=-1).

    w : (N+1,) ndarray
        Quadrature weights.

    P : (N+1,N+1) ndarray
        Legendre Vandermonde matrix.
    """

    N1 = N + 1

    # ------------------------------------------------------------
    # Chebyshev-Gauss-Radau initial guess
    # ------------------------------------------------------------

    x = -np.cos(2.0 * np.pi * np.arange(N1) / (2 * N + 1))

    # ------------------------------------------------------------
    # Legendre Vandermonde
    # ------------------------------------------------------------

    P = np.zeros((N1, N1 + 1))

    xold = np.full_like(x, 2.0)

    free = np.arange(1, N1)

    tol = np.finfo(float).eps

    while np.max(np.abs(x - xold)) > tol:

        xold[:] = x

        # MATLAB:
        # P(1,:) = (-1).^(0:N1)

        P[0, :] = (-1.0) ** np.arange(N1 + 1)

        # MATLAB:
        # P(free,1)=1
        # P(free,2)=x

        P[free, 0] = 1.0
        P[free, 1] = x[free]

        # MATLAB recursion

        for k in range(2, N1 + 1):

            P[free, k] = (
                ((2 * k - 1) * x[free] * P[free, k - 1])
                - (k - 1) * P[free, k - 2]
            ) / k

        # MATLAB Newton step

        PN = P[free, N1 - 1]
        PN1 = P[free, N1]

        x[free] = (
            xold[free]
            - ((1.0 - xold[free]) / N1)
            * (PN + PN1)
            / (PN - PN1)
        )

    # ------------------------------------------------------------
    # Trim Vandermonde
    # MATLAB: P=P(1:N1,1:N1)
    # ------------------------------------------------------------

    P = P[:, :N1]

    # ------------------------------------------------------------
    # Weights
    # MATLAB:
    #
    # w(1)=2/N1^2;
    # w(free)=(1-x)./(N1*P(:,N1)).^2;
    # ------------------------------------------------------------

    w = np.zeros(N1)

    w[0] = 2.0 / (N1 * N1)

    w[free] = (
        (1.0 - x[free])
        / (N1 * PN) ** 2
    )

    return x, w, P