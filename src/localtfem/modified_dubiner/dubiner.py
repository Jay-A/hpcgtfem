from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, List

from .jacobi import jacobi_coeffs_float


# ============================================================
# polynomial utilities
# ============================================================

def poly_eval(coeffs, x):
    coeffs = np.asarray(coeffs)[::-1]
    x = np.asarray(x, dtype=float)

    out = np.zeros_like(x, dtype=float)
    for c in coeffs:
        out = out * x + c
    return out


def poly_derivative(coeffs):
    coeffs = np.asarray(coeffs)
    n = len(coeffs)
    if n <= 1:
        return np.array([0.0])
    return np.array([i * coeffs[i] for i in range(1, n)])


def jacobi_pair(coeffs):
    """Return (P, dP) from coefficient vector"""
    return (
        lambda x: poly_eval(coeffs, x),
        lambda x: poly_eval(poly_derivative(coeffs), x)
    )


# ============================================================
# mode container (MATLAB-STYLE: explicit derivatives)
# ============================================================

@dataclass
class DubinerMode:
    phi: Callable
    dxi: Callable
    deta: Callable

    def evaluate(self, xi, eta):
        return self.phi(xi, eta)

    def gradient(self, xi, eta):
        return self.dxi(xi, eta), self.deta(xi, eta)


# ============================================================
# Dubiner basis
# ============================================================

class DubinerBasis:

    def __init__(self, p: int):
        self.p = p
        self.modes: List[DubinerMode] = []
        self._build()

    # --------------------------------------------------------
    # evaluation
    # --------------------------------------------------------

    def evaluate(self, xi, eta):
        xi = np.asarray(xi, dtype=float)
        eta = np.asarray(eta, dtype=float)

        n = len(self.modes)
        out = np.zeros(xi.shape + (n,), dtype=float)

        for i, m in enumerate(self.modes):
            out[..., i] = m.evaluate(xi, eta)

        return out

    def evaluate_grad(self, xi, eta):
        xi = np.asarray(xi, dtype=float)
        eta = np.asarray(eta, dtype=float)

        n = len(self.modes)
        gx = np.zeros(xi.shape + (n,))
        gy = np.zeros(xi.shape + (n,))

        for i, m in enumerate(self.modes):
            dx, dy = m.gradient(xi, eta)
            gx[..., i] = dx
            gy[..., i] = dy

        return gx, gy

    # ========================================================
    # vertex modes (EXACT MATCH TO MATLAB)
    # ========================================================

    def _build_vertex_modes(self):

        # φ1 = 1/2*(1+η)
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.5 * (1 + eta),
            dxi=lambda xi, eta: 0.0,
            deta=lambda xi, eta: 0.5
        ))

        # φ2 = 1/4*(1-ξ)*(1-η)
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.25 * (1 - xi) * (1 - eta),
            dxi=lambda xi, eta: -0.25 * (1 - eta),
            deta=lambda xi, eta: -0.25 * (1 - xi)
        ))

        # φ3 = 1/4*(1+ξ)*(1-η)
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.25 * (1 + xi) * (1 - eta),
            dxi=lambda xi, eta: 0.25 * (1 - eta),
            deta=lambda xi, eta: -0.25 * (1 + xi)
        ))

    # ========================================================
    # edge modes (A_m, B_m, C_m)
    # ========================================================

    def _build_edge_modes(self):

        for m in range(1, self.p):

            P = jacobi_coeffs_float(m - 1, 2, 2)
            Pp = poly_derivative(P)

            P_eval = lambda x: poly_eval(P, x)
            dP_eval = lambda x: poly_eval(Pp, x)

            # ----------------------------
            # A_m
            # ----------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_eval:
                    ((1 + xi)/2) * ((1 - xi)/2)
                    * P(xi)
                    * ((1 - eta)/2)**(m + 1),

                dxi=lambda xi, eta, m=m, P=P_eval, dP=dP_eval:
                    0.5 * ((1 - xi)/2) * P(xi) * ((1 - eta)/2)**(m+1)
                    - 0.5 * ((1 + xi)/2) * P(xi) * ((1 - eta)/2)**(m+1)
                    + ((1 + xi)/2) * ((1 - xi)/2) * dP(xi) * ((1 - eta)/2)**(m+1),

                deta=lambda xi, eta, m=m, P=P_eval:
                    -0.5 * (m+1) * ((1 - eta)/2)**m
                    * ((1 + xi)/2) * ((1 - xi)/2) * P(xi)
            ))

            # ----------------------------
            # B_m
            # ----------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_eval:
                    ((1 + xi)/2)
                    * ((1 - eta)/2) * ((1 + eta)/2)
                    * P(eta),

                dxi=lambda xi, eta, m=m, P=P_eval:
                    0.5 * ((1 - eta)/2) * ((1 + eta)/2) * P(eta),

                deta=lambda xi, eta, m=m, P=P_eval, dP=dP_eval:
                    ((1 + xi)/2) * (
                        -0.5 * eta * P(eta)
                        + ((1 - eta**2)/4) * (m+3) * dP(eta)
                    )
            ))

            # ----------------------------
            # C_m
            # ----------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_eval:
                    (-1)**(m-1)
                    * ((1 - xi)/2)
                    * ((1 - eta)/2) * ((1 + eta)/2)
                    * P(eta),

                dxi=lambda xi, eta, m=m, P=P_eval:
                    (-1)**(m-1) * (-0.5)
                    * ((1 - eta)/2) * ((1 + eta)/2) * P(eta),

                deta=lambda xi, eta, m=m, P=P_eval, dP=dP_eval:
                    (-1)**(m-1) * ((1 - xi)/2) * (
                        -0.5 * eta * P(eta)
                        + ((1 - eta**2)/4) * (m+3) * dP(eta)
                    )
            ))

    # ========================================================
    # interior modes
    # ========================================================

    def _build_interior_modes(self):

        for m in range(self.p - 2):
            for n in range(self.p - 2 - m):

                Pxi = jacobi_coeffs_float(m, 2, 2)
                Peta = jacobi_coeffs_float(n, 2*m + 5, 2)

                dPxi = poly_derivative(Pxi)
                dPeta = poly_derivative(Peta)

                Pxi_f = lambda x: poly_eval(Pxi, x)
                dPxi_f = lambda x: poly_eval(dPxi, x)

                Peta_f = lambda x: poly_eval(Peta, x)
                dPeta_f = lambda x: poly_eval(dPeta, x)

                self.modes.append(DubinerMode(

                    phi=lambda xi, eta, m=m:
                        ((1 - xi)/2) * ((1 + xi)/2)
                        * Pxi_f(xi)
                        * ((1 - eta)/2)**(m+2)
                        * ((1 + eta)/2)
                        * Peta_f(eta),

                    dxi=lambda xi, eta, m=m:
                        (
                            (-0.5*(1 + xi) + 0.5*(1 - xi)) * Pxi_f(xi)
                            + ((1 - xi)/2)*((1 + xi)/2)*dPxi_f(xi)
                        )
                        * ((1 - eta)/2)**(m+2)
                        * ((1 + eta)/2)
                        * Peta_f(eta),

                    deta=lambda xi, eta, m=m:
                        ((1 - xi)/2)*((1 + xi)/2)*Pxi_f(xi)
                        * (
                            -0.5*(m+2)*((1 - eta)/2)**(m+1)
                            * ((1 + eta)/2)*Peta_f(eta)
                            + 0.5*((1 - eta)/2)**(m+2)*Peta_f(eta)
                            + ((1 - eta)/2)**(m+2)*dPeta_f(eta)
                        )
                ))

    # ========================================================
    def _build(self):
        self._build_vertex_modes()
        self._build_edge_modes()
        self._build_interior_modes()