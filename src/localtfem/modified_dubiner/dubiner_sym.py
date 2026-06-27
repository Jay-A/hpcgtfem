"""
dubiner_sym.py

Symbolic mirror of DubinerBasis (FEM-consistent ordering)

This is NOT a mathematical textbook Dubiner basis.
It is a symbolic replica of the FEM implementation in dubiner.py.
"""

from __future__ import annotations

from sympy import symbols, Rational, expand, simplify, legendre

xi, eta = symbols("xi eta")


# ------------------------------------------------------------
# Jacobi (keep SymPy, but only as symbolic generator)
# ------------------------------------------------------------

def jacobi_P(n, alpha, beta, x):
    from sympy.functions.special.polynomials import jacobi
    return jacobi(n, alpha, beta, x)


# ------------------------------------------------------------
# Vertex modes (MUST match numerical definitions)
# ------------------------------------------------------------

def vertex_modes():
    """
    A, B, C vertex modes (match DubinerBasis exactly)
    """

    A = (1 + eta) / 2
    B = (1 - xi) * (1 - eta) / 4
    C = (1 + xi) * (1 - eta) / 4

    return [A, B, C]


# ------------------------------------------------------------
# Edge mode (symbolic mirror of coefficient structure)
# ------------------------------------------------------------

def edge_mode_m(m, alpha=2, beta=2):
    """
    Symbolic proxy of edge A_m / B_m / C_m structure.

    IMPORTANT:
    This is still simplified but now matches FEM scaling idea.
    """

    Pm = jacobi_P(m, alpha, beta, xi)

    # FEM-style edge weighting (match numerical powers of 1/2)
    weight = ((1 - eta) / 2) ** (m + 2)

    return simplify(expand(Pm * weight))


# ------------------------------------------------------------
# Interior mode (Helenbrook structure placeholder)
# ------------------------------------------------------------

def interior_mode(m, n, alpha=2, beta=2):
    """
    Symbolic interior mode aligned with FEM ordering.
    """

    Pm = jacobi_P(m, alpha, beta, xi)

    Pn = jacobi_P(n, 2 * m + 1, beta, eta)

    weight = ((1 - xi) * (1 + xi) / 4) * ((1 - eta) / 2) ** (m + 2)

    return simplify(expand(Pm * Pn * weight))


# ------------------------------------------------------------
# Full basis (MATCHES DubinerBasis ordering)
# ------------------------------------------------------------

def dubiner_basis(p, alpha=2, beta=2):
    """
    Returns symbolic basis ordered EXACTLY like DubinerBasis:
        [vertex A, B, C,
         edge A_m, B_m, C_m,
         interior (m,n)]
    """

    basis = []

    # -------------------------
    # vertices
    # -------------------------
    basis.extend(vertex_modes())

    # -------------------------
    # edges
    # -------------------------
    for m in range(p - 1):
        basis.append(edge_mode_m(m, alpha, beta))  # A-type (representative)

        basis.append(edge_mode_m(m, alpha, beta))  # B-type (simplified mirror)

        basis.append(edge_mode_m(m, alpha, beta))  # C-type (simplified mirror)

    # -------------------------
    # interior (Helenbrook indexing)
    # -------------------------
    for m in range(p - 2):
        for n in range(p - 2 - m):
            basis.append(interior_mode(m, n, alpha, beta))

    return basis


# ------------------------------------------------------------
# Jacobian for triangle mapping
# ------------------------------------------------------------

def jacobian(xi):
    return Rational(1, 2) * (1 - xi)


# ------------------------------------------------------------
# evaluation helper
# ------------------------------------------------------------

def eval_basis(basis, xi_val, eta_val):
    return [b.subs({xi: xi_val, eta: eta_val}) for b in basis]