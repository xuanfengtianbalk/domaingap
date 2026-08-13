import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from dlsu import dlsu

def dlsu_full(X, x, Sigmas, R_est_3, t_est_3, full_cov_mode=1):
    """
    Uncertainty-aware PnPL solver for points and lines

    Parameters:
    X: 3 x n_pt array of 3D points
    x: 2 x n_pt array of 2D point projections
    Sigmas: 3 x 3 x n_pt array of 3D point covariances
    Sigmas2D: 2 x 2 x n_pt array of 2D point covariances
    R_est_3: 3x3 rotation matrix of a pose hypothesis (or empty array)
    t_est_3: 3x1 translation vector of a pose hypothesis (or average scene depth)
    Xs: 3 x n_l array of 3D line start points
    Xe: 3 x n_l array of 3D line end points
    l: 3 x n_l array of line equations in the image plane
    Sigmas3DLines: 6 x 6 x n_l array of 3D line covariances
    Sigmas2DLines: n_l array of 2D line covariances
    full_cov_mode: whether to use full covariance mode

    Returns:
    R_est: List of 3x3 rotation matrices
    t_est: List of 3x1 translation vectors
    """

    # Handle default covariance cases
    if Sigmas.ndim < 3:
        npt = X.shape[1]
        Sigmas = np.tile(np.eye(3), (1, 1, npt))
        Sigmas2D = np.tile(np.eye(2), (1, 1, npt))
        t_est_3 = 1.0

    # Centralize 3D points
    meanX = np.mean(X, axis=1, keepdims=True)
    X = X - meanX
    if t_est_3.size > 1 and R_est_3.size > 0:
        t_est_3 = t_est_3 - R_est_3 @ meanX

    # Precompute isotropic covariances if needed
    if full_cov_mode == 0:
        sigmas2 = compute_traces(Sigmas)
        sigmas2d_2 = compute_traces(Sigmas2D)


    R_est, t_est, costs_est = dlsu(
        X, x, R_est_3, t_est_3
    )

    # Filter solutions by cost
    valid_mask = costs_est < 0.1
    R_est = R_est[:, :, valid_mask]
    t_est = t_est[:, valid_mask]

    # Handle no-solution case with random rotations
    if R_est.size == 0:
        R_est_r = []
        t_est_r = []
        costs_r = []

        for _ in range(3):
            randax = 3 * np.random.randn(3)
            randax[np.random.randint(3)] = 0
            Rr = Rotation.from_rotvec(randax).as_matrix()

            if R_est_3.size > 0:
                R_est_new = R_est_3 @ Rr.T
            else:
                R_est_new = np.array([])

            if n_lines == 0:
                Xs_rot = np.zeros((3, 0))
                Xe_rot = np.zeros((3, 0))
            else:
                Xs_rot = Rr @ Xs
                Xe_rot = Rr @ Xe

            # Prepare rotated covariances
            if full_cov_mode == 1:
                n_pt = Sigmas.shape[2]
                SigmasR = np.zeros_like(Sigmas)
                for j in range(n_pt):
                    SigmasR[:, :, j] = Rr @ Sigmas[:, :, j] @ Rr.T

                n_ln = Sigmas3DLines.shape[2]
                Sigmas3DLinesR = np.zeros_like(Sigmas3DLines)
                for j in range(n_ln):
                    Sigmas3DLinesR[:3, :3, j] = Rr @ Sigmas3DLines[:3, :3, j] @ Rr.T
                    Sigmas3DLinesR[3:, 3:, j] = Rr @ Sigmas3DLines[3:, 3:, j] @ Rr.T
            else:
                SigmasR = Sigmas
                Sigmas3DLinesR = Sigmas3DLines

            # Solve with rotated features
            try:
                if full_cov_mode == 0:
                    R_tmp, t_tmp, costs_tmp = dlsu_fast(
                        Rr @ X, x, Xs_rot, Xe_rot, l, sigmas2,
                        sigmas_lines_2, sigmas2d_2, Sigmas2DLines,
                        R_est_new, t_est_3
                    )
                else:
                    R_tmp, t_tmp, costs_tmp = dlsu(
                        Rr @ X, x, Xs_rot, Xe_rot, l, SigmasR,
                        Sigmas3DLinesR, Sigmas2D, Sigmas2DLines,
                        R_est_new, t_est_3
                    )
            except:
                continue

            # Store solutions
            for k in range(R_tmp.shape[2]):
                R_corrected = R_tmp[:, :, k] @ Rr
                R_est_r.append(R_corrected)
                t_est_r.append(t_tmp[:, k])
                costs_r.append(costs_tmp[k])

        # Select best solution
        if costs_r:
            best_idx = np.argmin(costs_r)
            R_est = np.expand_dims(R_est_r[best_idx], axis=2)
            t_est = np.expand_dims(t_est_r[best_idx], axis=1)

    # Apply centralization correction
    for si in range(R_est.shape[2]):
        t_est[:, si] = t_est[:, si] - R_est[:, :, si] @ meanX.flatten()

    return R_est, t_est


def compute_traces(covs):
    """Compute traces of covariance matrices"""
    if covs.ndim == 2:
        return np.trace(covs)
    elif covs.ndim == 3:
        return np.array([np.trace(covs[:, :, i]) for i in range(covs.shape[2])])
    return 0