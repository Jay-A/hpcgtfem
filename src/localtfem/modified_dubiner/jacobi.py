#!/usr/bin/env python3

import argparse
import numpy as np
import sympy as sp

def jacobi_coeffs_float(n, alpha, beta):
    """
    Jacobi polynomial coefficients in monomial basis (ascending order)
    using float arithmetic.
    """

    if n == 0:
        return np.array([1.0], dtype=float)

    if n == 1:
        c0 = 0.5 * (2*(alpha + 1) - (alpha + beta + 2))
        c1 = 0.5 * (alpha + beta + 2)
        return np.array([c0, c1], dtype=float)

    Cfs = np.zeros((n + 1, n + 1), dtype=float)

    Cfs[0, 0] = 1.0

    Cfs[1, 0] = 0.5 * (2*(alpha + 1) - (alpha + beta + 2))
    Cfs[1, 1] = 0.5 * (alpha + beta + 2)

    def c(k):
        return k + alpha + beta

    for counter in range(2, n + 1):
        k = counter

        temp_poly = np.zeros(n + 1, dtype=float)
        temp_poly[-2] = c(2*k - 1) * c(2*k - 2) * c(2*k)
        temp_poly[-1] = c(2*k - 1) * (alpha**2 - beta**2)

        temp = np.convolve(temp_poly, Cfs[counter - 1, :])
        left = temp[:n+1]

        correction = (
            2 * (k - 1 + alpha) *
            (k - 1 + beta) *
            c(2*k) *
            Cfs[counter - 2, :]
        )

        denom = 2 * k * c(k) * c(2*k - 2)

        Cfs[counter, :] = (left - correction) / denom

    return Cfs[n, :]

def poly_conv(a, b):
    """Pure Python convolution for SymPy lists."""
    out = [sp.Integer(0)] * (len(a) + len(b) - 1)

    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj

    return out


def jacobi_coeffs_sym(n, alpha, beta):
    """
    Exact Jacobi coefficients using SymPy rationals.
    """

    alpha = sp.Rational(alpha)
    beta = sp.Rational(beta)

    if n == 0:
        return [sp.Integer(1)]

    if n == 1:
        c0 = sp.Rational(1, 2) * (2*(alpha + 1) - (alpha + beta + 2))
        c1 = sp.Rational(1, 2) * (alpha + beta + 2)
        return [c0, c1]

    Cfs = [[sp.Integer(0)] * (n + 1) for _ in range(n + 1)]

    Cfs[0][0] = sp.Integer(1)

    Cfs[1][0] = sp.Rational(1, 2) * (2*(alpha + 1) - (alpha + beta + 2))
    Cfs[1][1] = sp.Rational(1, 2) * (alpha + beta + 2)

    def c(k):
        return k + alpha + beta

    for counter in range(2, n + 1):
        k = counter

        temp_poly = [sp.Integer(0)] * (n + 1)
        temp_poly[-2] = c(2*k - 1) * c(2*k - 2) * c(2*k)
        temp_poly[-1] = c(2*k - 1) * (alpha**2 - beta**2)

        temp = poly_conv(temp_poly, Cfs[counter - 1])
        left = temp[:n+1]

        correction = [
            2*(k - 1 + alpha)*(k - 1 + beta)*c(2*k)*Cfs[counter - 2][i]
            for i in range(n+1)
        ]

        denom = 2*k*c(k)*c(2*k - 2)

        Cfs[counter] = [
            (left[i] - correction[i]) / denom
            for i in range(n+1)
        ]

    return Cfs[n]

def main():
    parser = argparse.ArgumentParser(
        description="Jacobi polynomial coefficients (Dubiner FEM)"
    )

    parser.add_argument("n", type=int, help="Polynomial degree")
    parser.add_argument("alpha", type=float, help="Jacobi alpha")
    parser.add_argument("beta", type=float, help="Jacobi beta")

    parser.add_argument(
        "-s", "--sym",
        action="store_true",
        help="Use SymPy exact rational arithmetic"
    )

    args = parser.parse_args()

    if args.sym:
        coeffs = jacobi_coeffs_sym(args.n, args.alpha, args.beta)
    else:
        coeffs = jacobi_coeffs_float(args.n, args.alpha, args.beta)

    print(coeffs)


if __name__ == "__main__":
    main()    