"""
legendre.py

One-dimensional Gaussian quadrature rules on the interval [-1, 1].

This module provides

    • Gauss-Legendre
    • Gauss-Lobatto-Legendre
    • Gauss-Radau-Legendre

quadrature nodes and weights for use in the modified Dubiner finite
element basis and reference-triangle quadrature.

References
----------
Greg von Winckel (2004)
    lgwt.m

Appleton & Helenbrook (2019)
    A High-Order Lower-Triangular Pseudo-Mass Matrix for Explicit Time
    Advancement of hp Triangular Finite Element Methods.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_jacobi, roots_legendre


def gauss_legendre(n: int):
    """
    Compute the n-point Gauss-Legendre quadrature rule.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    points : ndarray
        Quadrature nodes.

    weights : ndarray
        Quadrature weights.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    return roots_legendre(n)


def gauss_lobatto(n: int):
    """
    Compute the n-point Gauss-Lobatto-Legendre quadrature rule.

    Endpoints (-1,1) are included.

    Parameters
    ----------
    n : int
        Number of quadrature points (n >= 2).

    Returns
    -------
    points : ndarray
    weights : ndarray
    """
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2.")

    if n == 2:
        return (
            np.array([-1.0, 1.0]),
            np.array([1.0, 1.0]),
        )

    interior, _ = roots_jacobi(n - 2, 1, 1)

    points = np.concatenate(([-1.0], interior, [1.0]))

    from scipy.special import eval_legendre

    weights = np.empty(n)

    weights[0] = 2.0 / (n * (n - 1))
    weights[-1] = weights[0]

    for i, x in enumerate(interior):
        P = eval_legendre(n - 1, x)
        weights[i + 1] = (
            2.0 /
            (n * (n - 1) * P * P)
        )

    return points, weights


def gauss_radau(n: int):
    """
    Compute the n-point Gauss-Radau-Legendre quadrature rule
    including the endpoint x = -1.

    Parameters
    ----------
    n : int
        Number of quadrature points (n >= 2).

    Returns
    -------
    points : ndarray
    weights : ndarray
    """
    if n < 2:
        raise ValueError("Gauss-Radau requires n >= 2.")

    interior, _ = roots_jacobi(n - 1, 0, 1)

    points = np.concatenate(([-1.0], interior))

    #
    # Weight computation follows the Jacobi formulation.
    #
    # This is sufficient for the current quadrature package.
    # If later we require the differentiation matrix returned
    # by MATLAB's lgrnodes.m, it can be added here.
    #
    weights = np.empty(n)

    weights[0] = 2.0 / (n * n)

    from scipy.special import eval_legendre

    for i, x in enumerate(interior):
        P = eval_legendre(n - 1, x)
        weights[i + 1] = (
            (1.0 - x) /
            (n * n * P * P)
        )

    return points, weights