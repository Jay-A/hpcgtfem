import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from localtfem.modified_dubiner.dubiner_sym import (
    dubiner_basis,
    dubiner_basis_lambdified,
    symbolic_mass_matrix,
)
from localtfem.geometry.reference_triangle import (
    square_to_triangle,
    triangle_to_square,
    jacobian,
    reference_to_physical
)

from localtfem.fem.rhs import eval_rhs


class ReferenceProjectionTest:
    """
    FEM L2 projection test in Dubiner reference element.

    Computes:
        M c_rec = rhs
    where:
        rhs_i = ∫ f φ_i (1-η)/2 dξ dη
    """

    # ------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------
    def __init__(self, p=1, c=None):
        self.p = p

        self.xi, self.eta = sp.symbols("xi eta")

        self.basis_sym = dubiner_basis(p)
        self.basis_np = [
            sp.lambdify((self.xi, self.eta), phi, "numpy")
            for phi in self.basis_sym
        ]

        self.n_modes = len(self.basis_np)

        self.c = self._make_coefficients(c)

        # symbolic mass matrix (IMPORTANT: same space)
        self.M = np.array(
            symbolic_mass_matrix(p, integrate=True),
            dtype=float
        )

    # ------------------------------------------------------------
    # coefficient builder
    # ------------------------------------------------------------
    def _make_coefficients(self, c):
        n = self.n_modes

        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape != (n,):
                raise ValueError(f"Expected shape {(n,)}, got {c.shape}")
            return c

        out = np.zeros(n)
        if n > 1:
            out[1] = 1.0
        return out

    # ------------------------------------------------------------
    # evaluate u_h
    # ------------------------------------------------------------
    def evaluate_function(self, XI, ETA):
        u = np.zeros_like(XI, dtype=float)

        for ci, phi in zip(self.c, self.basis_np):
            u += ci * phi(XI, ETA)

        return u

    # ------------------------------------------------------------
    # basis eval
    # ------------------------------------------------------------
    def basis_eval(self, XI, ETA):
        return np.stack(
            [phi(XI, ETA) for phi in self.basis_np],
            axis=-1
        )

    # ------------------------------------------------------------
    # RHS (REFERENCE SPACE ONLY)
    # ------------------------------------------------------------
    def compute_rhs(self, f, quad_x, quad_w):

        XI, ETA = np.meshgrid(quad_x, quad_x, indexing="xy")
        W = np.outer(quad_w, quad_w)

        Phi = self.basis_eval(XI, ETA)
        F = f(XI, ETA)

        jac = 0.5 * (1.0 - ETA)

        rhs = np.zeros(self.n_modes)

        for i in range(self.n_modes):
            rhs[i] = np.sum(F * Phi[..., i] * W * jac)

        return rhs

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    def plot_field(self, XI, ETA, U_exact):
        R, S = square_to_triangle(XI, ETA)

        plt.figure(figsize=(6, 6))

        cf = plt.contourf(R, S, U_exact, levels=30, cmap="viridis")
        cs = plt.contour(R, S, U_exact, colors="k", linewidths=0.5)

        plt.clabel(cs, fontsize=8)

        tri = np.array([
            [-1, -1],
            [-1,  1],
            [ 1, -1],
            [-1, -1],
        ])

        plt.plot(tri[:, 0], tri[:, 1], "k-", linewidth=1)

        plt.gca().set_aspect("equal")
        plt.colorbar(cf, shrink=0.7)
        plt.title("u_h on reference triangle (r,s)")
        plt.show()

    def plot_error(self, XI, ETA, U_exact, U_rec):
        plt.close('all')
        R, S = square_to_triangle(XI, ETA)
    
        err = np.abs(U_exact - U_rec)
    
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
        # --------------------------------------------------------
        # solution
        # --------------------------------------------------------
        cf1 = axs[0].contourf(R, S, U_exact, levels=30, cmap="viridis")
        cs = axs[0].contour(R, S, U_exact, colors="k", linewidths=0.5)
        tri = np.array([
            [-1, -1],
            [-1,  1],
            [ 1, -1],
            [-1, -1],
        ])

        axs[0].plot(tri[:, 0], tri[:, 1], "k-", linewidth=1)
        axs[0].set_title("u_rec (projected)")
        axs[0].set_aspect("equal")
        plt.clabel(cs, fontsize=8)
        plt.colorbar(cf1, ax=axs[0], shrink=0.8)
    
        # --------------------------------------------------------
        # error
        # --------------------------------------------------------
        cf2 = axs[1].contourf(R, S, err, levels=30, cmap="magma")
        axs[1].set_title("|u_exact - u_rec|")
        axs[1].set_aspect("equal")
        plt.colorbar(cf2, ax=axs[1], shrink=0.8)
    
        plt.tight_layout()
        plt.show()        

    # ------------------------------------------------------------
    # main run (THIS IS THE ACTUAL PROJECTION TEST)
    # ------------------------------------------------------------
    def run(self, n=200):

        xi_vals = np.linspace(-1, 1, n)
        eta_vals = np.linspace(-1, 1, n)
        XI, ETA = np.meshgrid(xi_vals, eta_vals)

        # --------------------------------------------------------
        # manufactured function
        # --------------------------------------------------------
        
        def f(r, s):
            
            return self.evaluate_function( triangle_to_square(r, s)[0],
                                           triangle_to_square(r, s)[1] )

        # --------------------------------------------------------
        # GL quadrature
        # --------------------------------------------------------
        
        quad_x, quad_w = np.polynomial.legendre.leggauss(8)
        
        XIq, ETAq = np.meshgrid(quad_x, quad_x, indexing="xy")
        
        # collapsed-to-triangle mapping
        R, S = square_to_triangle(XIq, ETAq)
        
        # Jacobian for collapse mapping
        J = jacobian(XIq, ETAq)
        
        # quadrature weights on reference triangle
        W = np.outer(quad_w, quad_w) * J
        
        # reference triangle vertices (identity element)
        verts = np.array([
            [-1.0,  1.0],
            [-1.0, -1.0],
            [ 1.0, -1.0],
        ])
        
        # --------------------------------------------------------
        # RHS old (still valid baseline)
        # --------------------------------------------------------
        
        rhs_old = self.compute_rhs(
            lambda xi, eta: self.evaluate_function(xi, eta),
            quad_x,
            quad_w,
        )
        
        # --------------------------------------------------------
        # RHS new (correct coordinate separation)
        # --------------------------------------------------------

        rhs_new = eval_rhs(
            func=f,
            verts=verts,
            p=self.p
        )

        # --------------------------------------------------------
        # solve projections
        # --------------------------------------------------------
        c_exact = self.c

        print( f"rhs_old: {rhs_old} " )
        print( f"rhs_new: {rhs_new} " )
        
        c_old = np.linalg.solve(self.M, rhs_old)
        c_new = np.linalg.solve(self.M, rhs_new)
        
        # --------------------------------------------------------
        # errors
        # --------------------------------------------------------
        err_old = np.linalg.norm(c_old - c_exact)
        err_new = np.linalg.norm(c_new - c_exact)
        
        print("||c_old - c_exact|| :", err_old)
        print("||c_new - c_exact|| :", err_new)
        
        print("RHS difference:", np.linalg.norm(rhs_old - rhs_new))            
        