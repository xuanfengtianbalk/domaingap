import numpy as np
from solver_opt_pnp_hesch2 import solver_opt_pnp_hesch2

from utils import dls_g, dls_H


def dlsu(X: np.ndarray, x: np.ndarray, Xs: np.ndarray, Xe: np.ndarray, l: np.ndarray,
         Sigmas: np.ndarray, SigmasLines: np.ndarray, Sigmas2D: np.ndarray,
         SigmasLines2D: np.ndarray, R_est = None,
         t_est = None, do_refine = True):
    """
    Python implementation of MATLAB's dlsu function for PnP problem with lines

    Parameters:
    X: 3D points (3 x np)
    x: 2D projections (2 x np)
    Xs: Line start points (3 x nl)
    Xe: Line end points (3 x nl)
    l: Line representations (3 x nl)
    Sigmas: Covariances for points (3 x 3 x np)
    SigmasLines: Covariances for lines (6 x 6 x nl)
    Sigmas2D: Additional covariances for points (2 x 2 x np)
    SigmasLines2D: Additional covariances for lines (nl x 1)
    R_est: Initial rotation estimate (3 x 3) or None
    t_est: Initial translation estimate (3 x 1) or None
    do_refine: Whether to perform refinement

    Returns:
    Rs: Rotation matrices (3 x 3 x nsols)
    ts: Translation vectors (3 x nsols)
    costs: Final costs for each solution
    """
    np_ = X.shape[1]  # Number of points
    nl = Xs.shape[1]  # Number of lines

    # Initialize variables
    T = np.zeros((3, 3))
    Tp = np.zeros((2, 3, np_))
    A = np.zeros((3, 9))
    S2s = np.zeros((2, 2, np_))
    Api = np.zeros((2, 9, np_))

    # Handle initial estimates
    if R_est is None or t_est is None:
        depths_est = np.ones(np_)
        depths_start_est = np.ones(nl)
        depths_end_est = np.ones(nl)
        R_est = np.eye(3)
        Xc = None
        Xsc = None
        Xec = None
    else:
        # Transform points and lines with initial estimate
        Xc = R_est @ X + t_est.reshape(3, 1)
        depths_est = Xc[2, :]
        Xsc = R_est @ Xs + t_est.reshape(3, 1)
        Xec = R_est @ Xe + t_est.reshape(3, 1)
        depths_start_est = Xsc[2, :]
        depths_end_est = Xec[2, :]

    # Process points
    for i in range(np_):
        # Compute Sigma2
        x1, x2 = x[:, i]
        if R_est is not None and t_est is not None:
            Sigma3 = R_est @ Sigmas[:, :, i] @ R_est.T
            Sigma2 = np.array([
                [Sigma3[2, 2] * x1 ** 2 - 2 * Sigma3[0, 2] * x1 + Sigma3[0, 0],
                 Sigma3[2, 2] * x1 * x2 - Sigma3[0, 2] * x2 - Sigma3[1, 2] * x1 + Sigma3[0, 1]],
                [Sigma3[2, 2] * x1 * x2 - Sigma3[0, 2] * x2 - Sigma3[1, 2] * x1 + Sigma3[0, 1],
                 Sigma3[2, 2] * x2 ** 2 - 2 * Sigma3[1, 2] * x2 + Sigma3[1, 1]]
            ])
            SigmaAdd = depths_est[i] ** 2 * Sigmas2D[:, :, i]
        else:
            Sigma2 = np.zeros((2, 2))
            SigmaAdd = np.zeros((2, 2))

        SigmaUnc = Sigma2 + SigmaAdd
        S2s[:, :, i] = np.linalg.inv(SigmaUnc + 1e-6 * np.eye(2))

        # Compute T and Tp
        Ti = np.array([[-1, 0, x1], [0, -1, x2]])
        T += Ti.T @ S2s[:, :, i] @ Ti
        Tp[:, :, i] = Ti

        # Compute A and Api
        Xp = X[:, i]
        Ai = np.vstack([
            np.hstack([-Xp, np.zeros(3), x1 * Xp]),
            np.hstack([np.zeros(3), -Xp, x2 * Xp])
        ])
        Api[:, :, i] = Ai
        A += Ti.T @ S2s[:, :, i] @ Ai

    # # Process lines
    # S2ls = np.zeros((2, 2, nl))
    # Tl = np.zeros((2, 3, nl))
    # Ali = np.zeros((2, 9, nl))

    # for i in range(nl):
    #     # Compute Sigma2 for lines
    #     li = l[:, i]
    #     if R_est is not None and t_est is not None:
    #         Sigma2 = np.array([
    #             [li.T @ R_est @ SigmasLines[:3, :3, i] @ R_est.T @ li,
    #              li.T @ R_est @ SigmasLines[:3, 3:6, i] @ R_est.T @ li],
    #             [li.T @ R_est @ SigmasLines[3:6, :3, i] @ R_est.T @ li,
    #              li.T @ R_est @ SigmasLines[3:6, 3:6, i] @ R_est.T @ li]
    #         ])
    #         SigmaAdd = np.diag([depths_start_est[i] ** 2, depths_end_est[i] ** 2]) * SigmasLines2D[i]
    #     else:
    #         Sigma2 = np.zeros((2, 2))
    #         SigmaAdd = np.zeros((2, 2))
    #
    #     Sigma2 += SigmaAdd
    #     S2ls[:, :, i] = np.linalg.inv(Sigma2 + 1e-9 * np.eye(2))
    #
    #     # Compute Tl and Ali
    #     Ti = np.vstack([li, li]).T
    #     Tl[:, :, i] = Ti
    #     T += Ti.T @ S2ls[:, :, i] @ Ti
    #
    #     # Compute Ali
    #     Xsi = Xs[:, i]
    #     Asi = li @ np.vstack([
    #         np.hstack([Xsi, np.zeros(3), np.zeros(3)]),
    #         np.hstack([np.zeros(3), Xsi, np.zeros(3)]),
    #         np.hstack([np.zeros(3), np.zeros(3), Xsi])
    #     ])
    #
    #     Xei = Xe[:, i]
    #     Aei = li @ np.vstack([
    #         np.hstack([Xei, np.zeros(3), np.zeros(3)]),
    #         np.hstack([np.zeros(3), Xei, np.zeros(3)]),
    #         np.hstack([np.zeros(3), np.zeros(3), Xei])
    #     ])
    #
    #     Ai = np.vstack([Asi, Aei])
    #     Ali[:, :, i] = Ai
    #     A += Ti.T @ S2ls[:, :, i] @ Ai

    # Compute A2
    A1 = A.copy()
    A2 = np.zeros((9, 9))

    # Points contribution to A2
    for i in range(np_):
        Ai = Api[:, :, i] - Tp[:, :, i] @ np.linalg.solve(T, A1)
        A2 += Ai.T @ S2s[:, :, i] @ Ai

    # # Lines contribution to A2
    # for i in range(nl):
    #     Ai = Ali[:, :, i] - Tl[:, :, i] @ np.linalg.solve(T, A1)
    #     A2 += Ai.T @ S2ls[:, :, i] @ Ai

    # QR matrix (MATLAB reshape equivalent with order='F')
    QR = np.array([
        [1, 0, 0, 0, 1, 0, 0, -1, 0, -1],
        [0, 0, 0, -2, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 0, 0, 1, 0, -1],
        [0, -2, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, -2, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
        [1, 0, 0, 0, -1, 0, 0, -1, 0, 1]
    ])

    A3 = QR.T @ A2 @ QR

    # Solve polynomial system (placeholder - need actual implementation)
    sols = solver_opt_pnp_hesch2(A3)

    # Process solutions
    Rs = []
    ts = []
    costs = []

    for i in range(sols.shape[1]):
        s_cand = sols[:, i]
        if np.all(np.isreal(s_cand)):
            s = s_cand.real
            Rc = rot_from_sol(s)
            r = Rc.T.ravel()  # MATLAB (:)
            tc = -np.linalg.solve(T, A1 @ r)

            # Check if points are in front of camera
            Xc = Rc @ X + tc.reshape(3, 1)
            if np.all(Xc[2, :] > 0):
                # Refinement
                sp = -s
                sinit = sp.copy()

                if do_refine:
                    costval = [0, 0, 0]
                    sps = np.zeros((2, 3))

                    Rit = rot_from_sol(sp)
                    r = Rit.T.ravel()
                    costval[0] = r @ A2 @ r

                    for it in range(2):
                        g = dls_g(sp[0], sp[1], sp[2], A2)
                        H = dls_H(sp[0], sp[1], sp[2], A2)
                        sp = sp - np.linalg.solve(H, g)
                        Rit = rot_from_sol(sp)
                        r = Rit.T.ravel()
                        costval[it + 1] = r @ A2 @ r
                        sps[it, :] = sp

                    if costval[2] > costval[0]:
                        sp = sinit

                sfin = -sp
                Rc = rot_from_sol(sfin)
                r = Rc.T.ravel()
                tc = -np.linalg.solve(T, A1 @ r)

                # Compute final cost
                Xce = Rc @ X + tc.reshape(3, 1)
                xce = Xce[:2, :] / Xce[2, :]
                fin_cost = np.max(np.sqrt(np.sum((xce - x) ** 2, axis=0)))

                Rs.append(Rc)
                ts.append(tc)
                costs.append(fin_cost)

    # Convert results to arrays
    if Rs:
        Rs = np.stack(Rs, axis=2)
        ts = np.stack(ts, axis=1)
        costs = np.array(costs)
    else:
        Rs = np.zeros((3, 3, 0))
        ts = np.zeros((3, 0))
        costs = np.array([])

    return Rs, ts, costs


# def rot_from_sol(sp):
#     norm_sp = norm(sp)
#     return (1.0 / (1 + norm_sp ** 2)) * ((1 - norm_sp ** 2) * np.eye(3) + 2 * cpmat(sp) + 2 * np.outer(sp, sp))
#
#
# def cpmat(sp):
#     return np.array([[0, -sp[2], sp[1]],
#                      [sp[2], 0, -sp[0]],
#                      [-sp[1], sp[0], 0]])

def rot_from_sol(sp: np.ndarray) -> np.ndarray:
    """Create rotation matrix from solution vector"""
    s = sp.reshape(3, 1)
    I = np.eye(3)
    denom = 1 + s.T @ s
    cp = cpmat(sp)
    return (1/denom) * ((1 - s.T @ s) * I + 2*cp + 2*s @ s.T)

def cpmat(v: np.ndarray) -> np.ndarray:
    """Cross product matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])



