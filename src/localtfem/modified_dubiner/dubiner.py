from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from .jacobi import jacobi_coeffs_float


# ------------------------------------------------------------
# polynomial utilities
# ------------------------------------------------------------

def poly_eval(coeffs, x):
    coeffs = np.asarray(coeffs)[::-1]
    x = np.asarray(x, dtype=float)

    out = np.zeros_like(x, dtype=float)
    for c in coeffs:
        out = out * x + c
    return out


def poly_mul(p, q):
    return np.convolve(p, q)


def lin(a, b):
    """(a + b x)"""
    return np.array([a, b])


def lin_pow(a, b, k):
    """(a + b x)^k"""
    out = np.array([1.0])
    base = np.array([a, b])
    for _ in range(k):
        out = poly_mul(out, base)
    return out


# ------------------------------------------------------------
# mode container (NOW FUNCTIONAL, NOT ARTIFICIAL SEPARATION)
# ------------------------------------------------------------

@dataclass
class DubinerMode:
    func: callable
    scale: float = 1.0

    def evaluate(self, xi, eta):
        return self.scale * self.func(xi, eta)


# ------------------------------------------------------------
# Dubiner basis
# ------------------------------------------------------------

class DubinerBasis:

    def __init__(self, p: int):
        self.p = p
        self.modes: List[DubinerMode] = []
        self._build()

    # --------------------------------------------------------
    def evaluate(self, xi, eta):
        xi = np.asarray(xi, dtype=float)
        eta = np.asarray(eta, dtype=float)

        n = len(self.modes)
        out = np.zeros(xi.shape + (n,), dtype=float)

        for i, m in enumerate(self.modes):
            out[..., i] = m.evaluate(xi, eta)

        return out

    # --------------------------------------------------------
    # vertex modes (MATLAB EXACT)
    # --------------------------------------------------------

    def _build_vertex_modes(self):

        self.modes.append(DubinerMode(
            func=lambda xi, eta: 0.5 * (1 + eta)
        ))

        self.modes.append(DubinerMode(
            func=lambda xi, eta: 0.25 * (1 - xi) * (1 - eta)
        ))

        self.modes.append(DubinerMode(
            func=lambda xi, eta: 0.25 * (1 + xi) * (1 - eta)
        ))

    # --------------------------------------------------------
    # edge modes (MATLAB structure preserved)
    # --------------------------------------------------------

    def _build_edge_modes(self):

        for m in range(1, self.p):

            P = jacobi_coeffs_float(m - 1, 2, 2)

            # A_m
            self.modes.append(DubinerMode(
                func=lambda xi, eta, P=P, m=m:
                   ( (1+xi)/2) 
                    * ( (1-xi)/2)  
                    * poly_eval(P, xi)   # if this is P_{m-1}^{2,2}(xi) 
                    * ((1 - eta)/2)**(m + 1) 
            ))

            # B_m
            self.modes.append(DubinerMode(
                func=lambda xi, eta, P=P, m=m:
                    ( (1+xi)/2) 
                    * ( (1-eta)/2)  
                    * ( (1+eta)/2)  
                    * poly_eval(P, eta)   # if this is P_{m-1}^{2,2}(eta)                
            ))

            # C_m
            self.modes.append(DubinerMode(
                func=lambda xi, eta, P=P, m=m:
                    (-1)**(m-1)
                    * ( (1-xi)/2) 
                    * ( (1-eta)/2)  
                    * ( (1+eta)/2)  
                    * poly_eval(P, eta)   # if this is P_{m-1}^{2,2}(xi) 
            ))

    # --------------------------------------------------------
    # interior modes (fully coupled like MATLAB)
    # --------------------------------------------------------

    def _build_interior_modes(self):

        for m in range(self.p - 2):
            for n in range(self.p - 2 - m):

                Pxi = jacobi_coeffs_float(m, 2, 2)
                Peta = jacobi_coeffs_float(n, 2*m + 5, 2)

                self.modes.append(DubinerMode(
                    func=lambda xi, eta, Pxi=Pxi, Peta=Peta, m=m:
                        poly_eval(Pxi, xi)
                        * poly_eval(Peta, eta)
                        * 0.25 * (1 - xi**2)
                        * (1 + eta) * (1 - eta)**(m + 2)
                ))

    # --------------------------------------------------------
    def _build(self):
        self._build_vertex_modes()
        self._build_edge_modes()
        self._build_interior_modes()