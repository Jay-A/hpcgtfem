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
    # s[s == 1.0] = 1.0 - 1e-16
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

def reference_to_physical(r, s, verts):
    """
    Affine map from the reference triangle

        (-1,  1)
        (-1, -1)
        ( 1, -1)

    to a physical triangle.

    Parameters
    ----------
    r, s : ndarray
        Reference triangle coordinates.

    verts : (3,2) ndarray
        Triangle vertices ordered as

            verts[0] = (-1,  1)
            verts[1] = (-1, -1)
            verts[2] = ( 1, -1)

    Returns
    -------
    x, y : ndarray
        Physical coordinates.
    """

    x = (
        0.5 * (verts[2, 0] - verts[1, 0]) * r
        + 0.5 * (verts[0, 0] - verts[1, 0]) * s
        + 0.5 * (verts[0, 0] + verts[2, 0])
    )

    y = (
        0.5 * (verts[2, 1] - verts[1, 1]) * r
        + 0.5 * (verts[0, 1] - verts[1, 1]) * s
        + 0.5 * (verts[0, 1] + verts[2, 1])
    )

    return x, y

def triangle_area(verts):
    """
    Area of a triangle with vertices verts.
    """

    x1, y1 = verts[0]
    x2, y2 = verts[1]
    x3, y3 = verts[2]

    return 0.5 * abs(
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )    