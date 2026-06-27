"""
reference_triangle.py

Geometry utilities for the reference triangular element.

Defines the collapsed-coordinate mapping used throughout the
modified Dubiner finite element implementation.

Coordinates
-----------
Collapsed square:
    (xi, eta) ∈ [-1,1] × [-1,1]

Reference triangle:
    (r, s)

Mapping (Helenbrook / Appleton):

    r = 0.5*(1 + xi)*(1 - eta) - 1
    s = eta

The Jacobian determinant is

    J = (1 - eta)/2

which appears in all quadrature and exact integrations.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------
# Forward map
# ------------------------------------------------------------

def square_to_triangle(xi, eta):
    """
    Map collapsed square coordinates (xi, eta)
    to reference triangle coordinates (r, s).
    """
    r = 0.5 * (1.0 + xi) * (1.0 - eta) - 1.0
    s = eta
    return r, s


# ------------------------------------------------------------
# Inverse map
# ------------------------------------------------------------

def triangle_to_square(r, s):
    """
    Map reference triangle coordinates (r, s)
    back to collapsed square coordinates (xi, eta).
    """
    xi = 2.0 * (r + 1.0) / (1.0 - s) - 1.0
    eta = s
    return xi, eta


# ------------------------------------------------------------
# Jacobian
# ------------------------------------------------------------

def jacobian(xi, eta):
    """
    Determinant of the collapsed-coordinate mapping.

        dr ds = J dxi deta

    J = (1 - eta)/2
    """
    return 0.5 * (1.0 - eta)


# ------------------------------------------------------------
# Inverse Jacobian
# ------------------------------------------------------------

def inverse_jacobian(xi, eta):
    """
    Inverse Jacobian determinant.
    """
    return 2.0 / (1.0 - eta)


# ------------------------------------------------------------
# Reference triangle vertices
# ------------------------------------------------------------

def vertices():
    """
    Vertices of the reference triangle
    in (r, s) coordinates.
    """
    return np.array([
        [-1.0,  1.0],   # vertex A
        [-1.0, -1.0],   # vertex B
        [ 1.0, -1.0],   # vertex C
    ])


# ------------------------------------------------------------
# Point inclusion
# ------------------------------------------------------------

def contains(r, s, tol=1e-12):
    """
    Return True if (r,s) lies inside the reference triangle.
    """
    return (
        (s >= -1.0 - tol)
        and (r >= -1.0 - tol)
        and (r <= -s + tol)
    )