"""
dubiner_sym.py

Symbolic mirror of DubinerBasis.

This file is intended to match the numerical implementation in
dubiner.py exactly, including

    * basis ordering
    * Jacobi polynomial parameters
    * edge families
    * interior families
    * weighting factors

so that symbolic evaluation agrees with DubinerBasis.evaluate().
"""

from __future__ import annotations

from sympy import Rational, expand, simplify, symbols, Matrix
from sympy.functions.special.polynomials import jacobi


import sympy as sp

r, s = sp.symbols("r s")

xi, eta = symbols("xi eta")


# ---------------------------------------------------------------------
# Jacobi polynomial helper
# ---------------------------------------------------------------------

def jacobi_P(n, alpha, beta, x):
    """Symbolic Jacobi polynomial."""
    return jacobi(n, alpha, beta, x)


# ---------------------------------------------------------------------
# Vertex modes
# ---------------------------------------------------------------------

def vertex_modes():
    return [
        (1 + eta) / 2,
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
    ]


# ---------------------------------------------------------------------
# Edge modes
# ---------------------------------------------------------------------

def edge_mode_A(m):

    P = jacobi_P(m - 1, 2, 2, xi)

    return simplify(expand(
        ((1 + xi) / 2)
        * ((1 - xi) / 2)
        * P
        * ((1 - eta) / 2) ** (m + 1)
    ))


def edge_mode_B(m):

    P = jacobi_P(m - 1, 2, 2, eta)

    return simplify(expand(
        ((1 + xi) / 2)
        * ((1 - eta) / 2)
        * ((1 + eta) / 2)
        * P
    ))


def edge_mode_C(m):

    P = jacobi_P(m - 1, 2, 2, eta)

    return simplify(expand(
        (-1) ** (m - 1)
        * ((1 - xi) / 2)
        * ((1 - eta) / 2)
        * ((1 + eta) / 2)
        * P
    ))


# ---------------------------------------------------------------------
# Interior modes
# ---------------------------------------------------------------------

def interior_mode(m, n):

    Pxi = jacobi_P(m, 2, 2, xi)
    Peta = jacobi_P(n, 2 * m + 5, 2, eta)

    return simplify(expand(
        ((1 - xi) / 2)
        * ((1 + xi) / 2)
        * Pxi
        * ((1 - eta) / 2) ** (m + 2)
        * ((1 + eta) / 2)
        * Peta
    ))


# ---------------------------------------------------------------------
# Full basis
# ---------------------------------------------------------------------

def dubiner_basis(p):

    basis = []

    # Vertices
    basis.extend(vertex_modes())

    # Edges
    for m in range(1, p):
        basis.append(edge_mode_A(m))
        basis.append(edge_mode_B(m))
        basis.append(edge_mode_C(m))

    # Interior
    for m in range(p - 2):
        for n in range(p - 2 - m):
            basis.append(interior_mode(m, n))

    return basis


# ---------------------------------------------------------------------
# Jacobian determinant (triangle collapse mapping)
# ---------------------------------------------------------------------

def jacobian():
    """
    Determinant of triangle collapse map:

        r = 0.5*(xi+1)*(1-eta) - 1
        s = eta
    """
    return Rational(1, 2) * (1 - eta)


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------

def eval_basis(basis, xi_val, eta_val):

    return [
        simplify(b.subs({xi: xi_val, eta: eta_val}))
        for b in basis
    ]


# ---------------------------------------------------------------------
# SYMBOLIC REFERENCE MASS MATRIX
# ---------------------------------------------------------------------

def symbolic_mass_matrix(p, integrate=True):
    """
    Symbolic reference mass matrix:

        M_ij = ∫∫ φ_i(ξ,η) φ_j(ξ,η) (1/2)(1-η) dξ dη

    Parameters
    ----------
    p : polynomial order
    integrate : bool
        False -> return symbolic integrand matrix
        True  -> perform exact symbolic integration (slow)
    """

    import sympy as sp

    basis = dubiner_basis(p)

    weight = Rational(1, 2) * (1 - eta)

    n = len(basis)

    M = Matrix(n, n, lambda i, j: 0)

    for i in range(n):
        for j in range(n):

            integrand = simplify(basis[i] * basis[j] * weight)

            if not integrate:
                M[i, j] = integrand
            else:
                M[i, j] = simplify(
                    sp.integrate(
                        sp.integrate(integrand, (xi, -1, 1)),
                        (eta, -1, 1)
                    )
                )

    return M

def symbolic_stiffness_matrix(p, integrate=True):
    """
    Symbolic reference stiffness matrix:

        K_ij = ∫ (∇φ_i · ∇φ_j) J(η) dξ dη
    """

    import sympy as sp

    basis = dubiner_basis(p)
    J = Rational(1, 2) * (1 - eta)

    n = len(basis)
    K = Matrix(n, n, lambda i, j: 0)

    for i in range(n):
        for j in range(n):

            dphii_xi = sp.diff(basis[i], xi)
            dphii_eta = sp.diff(basis[i], eta)

            dphij_xi = sp.diff(basis[j], xi)
            dphij_eta = sp.diff(basis[j], eta)

            integrand = simplify(
                (dphii_xi * dphij_xi +
                 dphii_eta * dphij_eta) * J
            )

            if not integrate:
                K[i, j] = integrand
            else:
                K[i, j] = simplify(
                    sp.integrate(
                        sp.integrate(integrand, (xi, -1, 1)),
                        (eta, -1, 1)
                    )
                )

    return K    

# ------------------------------------------------------------
# mapping from collapsed triangle
# ------------------------------------------------------------
def xi_map(r, s):
    return -1 + 2*(1 + r)/(1 - s)

def eta_map(r, s):
    return s


# ------------------------------------------------------------
# derivatives of mapping
# ------------------------------------------------------------
def dxi_dr(s):
    return 2/(1 - s)

def dxi_ds(r, s):
    return 2*(1 + r)/(1 - s)**2

def deta_dr():
    return 0

def deta_ds():
    return 1


# ------------------------------------------------------------
# reference gradient builder
# ------------------------------------------------------------
def reference_gradient(phi_xi, phi_eta):
    """
    Converts (xi,eta)-gradients into (r,s)-gradients.
    """

    def grad_r(r_val, s_val):
        return (
            phi_xi.subs({xi: xi_map(r_val, s_val), eta: eta_map(r_val, s_val)})
            * dxi_dr(s_val)
        )

    def grad_s(r_val, s_val):
        return (
            phi_xi.subs({xi: xi_map(r_val, s_val), eta: eta_map(r_val, s_val)})
            * dxi_ds(r_val, s_val)
            + phi_eta.subs({xi: xi_map(r_val, s_val), eta: eta_map(r_val, s_val)})
        )

    return grad_r, grad_s    