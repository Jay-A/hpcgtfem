import numpy as np

from .lglnodes import lglnodes
from .lgrnodes import lgrnodes


def RefTriQuad(p):
    """
    Quadrature on collapsed reference triangle.

    Matches MATLAB RefTri_Quad exactly:

        (xi, eta) ∈ [-1,1]^2
        r = 0.5*(xi+1)*(1-eta) - 1
        s = eta

    Uses:
        LGL in xi direction
        LGR in eta direction
    """

    # ------------------------------------------------------------
    # 1D quadrature rules
    # ------------------------------------------------------------

    lgl_pts, lgl_wts = lglnodes(3 * p)
    lgr_pts, lgr_wts = lgrnodes(3 * p)[:2]

    # MATLAB does a flip (important for matching ordering)
    lgr_pts = np.flip(lgr_pts)
    lgr_wts = np.flip(lgr_wts)

    # ------------------------------------------------------------
    # tensor-product grid in (xi, eta)
    # ------------------------------------------------------------

    XI, ETA = np.meshgrid(lgl_pts, lgr_pts, indexing="xy")
    WX, WY = np.meshgrid(lgl_wts, lgr_wts, indexing="xy")

    # ------------------------------------------------------------
    # collapsed triangle map (Rmap, Smap)
    # ------------------------------------------------------------

    R = 0.5 * (XI + 1.0) * (1.0 - ETA) - 1.0
    S = ETA

    # ------------------------------------------------------------
    # Jacobian of collapse map
    # ------------------------------------------------------------

    J = 0.5 * (1.0 - ETA)

    # ------------------------------------------------------------
    # combined quadrature weights
    # ------------------------------------------------------------

    W = WX * WY * J

    return R, S, W, XI, ETA