import numpy as np
from localtfem.quadrature.lglnodes import lglnodes
from localtfem.quadrature.lgrnodes import lgrnodes
from localtfem.geometry.reference_triangle import (
    square_to_triangle,
    jacobian,
    reference_to_physical,
)
from localtfem.modified_dubiner.dubiner_sym import (
    dubiner_basis_lambdified
)

def triangle_area(verts):
    A = np.array(verts[0])
    B = np.array(verts[1])
    C = np.array(verts[2])

    return 0.5 * abs(
        (B[0] - A[0]) * (C[1] - A[1]) -
        (B[1] - A[1]) * (C[0] - A[0])
    )

def eval_rhs(func,      # (x,y) -> R
             verts,     # np.array( [ [x_A, y_A], [x_B, y_B], [x_C, y_C] ] )
             p):         # polynomial basis order

    # let's define area and get phis
    
    area = triangle_area(verts)
    
    basis = dubiner_basis_lambdified(p)
    n_basis = len(basis)

    [lgl_nodes,lgl_weights] = lglnodes(15);
    [lgr_nodes,lgr_weights,_] = lgrnodes(15);

    XI, ETA = np.meshgrid(lgl_nodes, lgr_nodes, indexing="xy")
    W = np.outer(lgr_weights, lgl_weights)   # order here is important for correct broadcasting semantics

    Phi = np.stack([phi(XI, ETA) for phi in basis], axis=-1)

    #   Xmap = @(r,s) (1/2)*(Verts(3,1)-Verts(2,1)).*r ...
    #           + (1/2)*(Verts(1,1)-Verts(2,1)).*s + (1/2)*(Verts(1,1)+Verts(3,1));
    #   Ymap = @(r,s) (1/2)*(Verts(3,2)-Verts(2,2)).*r ...
    #           + (1/2)*(Verts(1,2)-Verts(2,2)).*s + (1/2)*(Verts(1,2)+Verts(3,2));
    
    A, B, C = verts
    S = ETA
    R = 0.5 * (XI + 1.0) * (1.0 - ETA) - 1.0
    
    X = ( 0.5 * (C[0] - B[0]) * R
        + 0.5 * (A[0] - B[0]) * S
        + 0.5 * (A[0] + C[0]) )
    
    Y = (  0.5 * (C[1] - B[1]) * R
        + 0.5 * (A[1] - B[1]) * S
        + 0.5 * (A[1] + C[1]) )
    
    F0 = func(X, Y)

    J = jacobian(XI, ETA)   # ( 1 - eta )/2.0
    # (area / 2.0) is the determinant of the affine map from 
    #   the reference triangle (r,s) to the physical triangle (x,y)

    rhs = (area / 2.0) * np.sum( F0[..., None] * Phi * W[..., None] * J[..., None], axis=(0, 1) )

    return rhs