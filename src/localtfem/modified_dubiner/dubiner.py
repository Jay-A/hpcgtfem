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


# ============================================================
# mode container
# ============================================================

@dataclass
class DubinerMode:
    phi: Callable
    dxi: Callable
    deta: Callable


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
            out[..., i] = m.phi(xi, eta)

        return out

    def evaluate_grad(self, xi, eta):
        xi = np.asarray(xi, dtype=float)
        eta = np.asarray(eta, dtype=float)

        n = len(self.modes)
        gx = np.zeros(xi.shape + (n,), dtype=float)
        gy = np.zeros(xi.shape + (n,), dtype=float)

        for i, m in enumerate(self.modes):
            gx[..., i] = m.dxi(xi, eta)
            gy[..., i] = m.deta(xi, eta)

        return gx, gy

    # ========================================================
    # vertex modes
    # ========================================================

    def _build_vertex_modes(self):

        # φ1
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.5 * (1 + eta),
            dxi=lambda xi, eta: 0.0,
            deta=lambda xi, eta: 0.5
        ))

        # φ2
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.25 * (1 - xi) * (1 - eta),
            dxi=lambda xi, eta: -0.25 * (1 - eta),
            deta=lambda xi, eta: -0.25 * (1 - xi)
        ))

        # φ3
        self.modes.append(DubinerMode(
            phi=lambda xi, eta: 0.25 * (1 + xi) * (1 - eta),
            dxi=lambda xi, eta: 0.25 * (1 - eta),
            deta=lambda xi, eta: -0.25 * (1 + xi)
        ))

    # ========================================================
    # edge modes (stable formulation)
    # ========================================================

    def _build_edge_modes(self):

        for m in range(1, self.p):

            P = jacobi_coeffs_float(m - 1, 2, 2)
            dP = poly_derivative(P)

            P_f = lambda x, P=P: poly_eval(P, x)
            dP_f = lambda x, dP=dP: poly_eval(dP, x)

            # -------------------------
            # A-type (xi-direction)
            # -------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_f:
                    0.25 * (1 - xi**2) * P(xi) * ((1 - eta)/2)**(m + 1),

                dxi=lambda xi, eta, m=m, P=P_f, dP=dP_f:
                    (
                        -0.5 * xi * P(xi)
                        + 0.25 * (1 - xi**2) * dP(xi)
                    ) * ((1 - eta)/2)**(m + 1),

                deta=lambda xi, eta, m=m, P=P_f:
                    -0.5 * (m + 1)
                    * 0.25 * (1 - xi**2)
                    * P(xi)
                    * ((1 - eta)/2)**m
            ))

            # -------------------------
            # B-type (eta-symmetric)
            # -------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_f:
                    0.25 * (1 + xi) * (1 - eta**2) * P(eta),

                dxi=lambda xi, eta, m=m, P=P_f:
                    0.25 * (1 - eta**2) * P(eta),

                deta=lambda xi, eta, m=m, P=P_f, dP=dP_f:
                    0.25 * (1 + xi) * (
                        -2 * eta * P(eta) + (1 - eta**2) * dP(eta)
                    )
            ))

            # -------------------------
            # C-type (antisymmetric)
            # -------------------------
            self.modes.append(DubinerMode(

                phi=lambda xi, eta, m=m, P=P_f:
                    (-1)**m * 0.25 * (1 - xi) * (1 - eta**2) * P(eta),

                dxi=lambda xi, eta, m=m, P=P_f:
                    (-1)**m * (-0.25) * (1 - eta**2) * P(eta),

                deta=lambda xi, eta, m=m, P=P_f, dP=dP_f:
                    (-1)**m * 0.25 * (1 - xi) * (
                        -2 * eta * P(eta) + (1 - eta**2) * dP(eta)
                    )
            ))

    # ========================================================
    # interior modes (stable tensor form)
    # ========================================================

    def _build_interior_modes(self):

        for m in range(self.p - 2):
            for n in range(self.p - 2 - m):

                Pxi = jacobi_coeffs_float(m, 2, 2)
                Peta = jacobi_coeffs_float(n, 2*m + 5, 2)

                dPxi = poly_derivative(Pxi)
                dPeta = poly_derivative(Peta)

                Pxi_f = lambda x, Pxi=Pxi: poly_eval(Pxi, x)
                dPxi_f = lambda x, dPxi=dPxi: poly_eval(dPxi, x)

                Peta_f = lambda x, Peta=Peta: poly_eval(Peta, x)
                dPeta_f = lambda x, dPeta=dPeta: poly_eval(dPeta, x)

                self.modes.append(DubinerMode(

                    phi=lambda xi, eta, m=m, n=n:
                        0.25
                        * (1 - xi**2)
                        * (1 - eta**2)
                        * Pxi_f(xi)
                        * Peta_f(eta),

                    dxi=lambda xi, eta, m=m, n=n:
                        0.25 * (
                            -2 * xi * (1 - eta**2) * Pxi_f(xi) * Peta_f(eta)
                            + (1 - xi**2) * (1 - eta**2) * dPxi_f(xi) * Peta_f(eta)
                        ),

                    deta=lambda xi, eta, m=m, n=n:
                        0.25 * (
                            (1 - xi**2)
                            * (-2 * eta * Peta_f(eta) + (1 - eta**2) * dPeta_f(eta))
                            * Pxi_f(xi)
                        )
                ))

    # ========================================================
    def _build(self):
        self._build_vertex_modes()
        self._build_edge_modes()
        self._build_interior_modes()