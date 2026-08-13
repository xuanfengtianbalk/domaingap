import numpy as np
import cv2
from scipy.linalg import sqrtm


def compute_projected_covariance(pts_3d, covariances_3d, K, R, t, delta1=1e-3):
    """
    计算3D点投影到2D图像平面后的协方差矩阵

    参数:
        pts_3d: 3xN 的3D点坐标矩阵 (世界坐标系)
        covariances_3d: 3x3xN 的3D点协方差矩阵数组
        K: 3x3 相机内参矩阵
        R: 3x3 旋转矩阵 (世界坐标系到相机坐标系)
        t: 3x1 平移向量
        delta1: 小扰动值，用于数值稳定性 (默认1e-3)

    返回:
        pts_2d: 2xN 的投影2D点坐标
        covariances_2d: 2x2xN 的2D协方差矩阵数组
    """
    # 确保输入形状正确
    pts_3d = np.asarray(pts_3d)
    if pts_3d.ndim == 1:
        pts_3d = pts_3d[:, np.newaxis]

    n = pts_3d.shape[1]  # 点的数量

    # 转换到相机坐标系
    pts_cam = R @ pts_3d + t

    # 提取深度值 (Z坐标)
    depths = pts_cam[2, :]

    # 投影到归一化平面
    x_norm = pts_cam[0, :] / depths
    y_norm = pts_cam[1, :] / depths

    # 应用内参矩阵得到像素坐标
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    pts_2d = np.vstack([u, v])

    # 初始化2D协方差数组
    covariances_2d = np.zeros((2, 2, n))

    # 对每个点计算投影后的协方差
    for i in range(n):
        # 提取当前点的坐标和深度
        Xc = pts_cam[0, i]
        Yc = pts_cam[1, i]
        Zc = depths[i]

        # 提取当前点的3D协方差
        Sigma3d = covariances_3d[:, :, i]

        # 旋转协方差到相机坐标系
        Sigma_cam = R @ Sigma3d @ R.T

        # 计算投影雅可比矩阵 (关于相机坐标系)
        J_proj = np.array([
            [K[0, 0] / Zc, 0, -K[0, 0] * Xc / (Zc * Zc)],
            [0, K[1, 1] / Zc, -K[1, 1] * Yc / (Zc * Zc)]
        ])

        # 计算投影后的协方差
        Sigma2d = J_proj @ Sigma_cam @ J_proj.T

        # 添加小扰动保证正定性
        Sigma2d += np.eye(2) * delta1

        # 存储结果
        covariances_2d[:, :, i] = Sigma2d

    return pts_2d, covariances_2d

def rodrigues(rvec):
    """罗德里格斯公式：将旋转向量转换为旋转矩阵"""
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    r = rvec / theta
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R


def rotation_matrix_to_euler_zyx(R):
    """
    将旋转矩阵转换为 Z-Y-X 欧拉角 (yaw, pitch, roll)

    参数:
        R : 3x3 旋转矩阵

    返回:
        euler : 欧拉角向量 [yaw, pitch, roll] (弧度)
    """
    # 确保矩阵是旋转矩阵
    assert R.shape == (3, 3), "输入必须是 3x3 矩阵"

    # 提取矩阵元素
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    # 计算 pitch (y 轴旋转)
    pitch = np.arcsin(-r31)

    # 避免万向节锁情况 (pitch = ±π/2)
    if np.abs(np.abs(r31) - 1) < 1e-6:
        # 万向节锁情况
        yaw = 0
        roll = np.arctan2(-r12, r11)
    else:
        # 正常情况
        cy = np.cos(pitch)
        yaw = np.arctan2(r21 / cy, r11 / cy)
        roll = np.arctan2(r32 / cy, r33 / cy)

    return np.array([yaw, pitch, roll])


def rodrigues_to_euler(v):
    """
    将罗德里格参数转换为 Z-Y-X 欧拉角

    参数:
        v : 罗德里格参数向量 [v1, v2, v3]

    返回:
        euler : 欧拉角向量 [yaw, pitch, roll] (弧度)
    """
    R = rodrigues(v)
    return rotation_matrix_to_euler_zyx(R)


def rodrigues_to_q(v):
    """
    将罗德里格参数转换为单位四元数

    参数:
        v : 罗德里格参数向量 [v1, v2, v3]

    返回:
        q : 单位四元数 [w, x, y, z]
    """
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # 计算四元数分量
    w = np.cos(theta / 2)
    sin_half_theta = np.sin(theta / 2)
    n = v / theta  # 旋转轴
    x, y, z = sin_half_theta * n

    return np.array([w, x, y, z])

def project_3d_to_uv(X_w, rvec, t, K, Sigma_Xw):
    """
    计算3D点投影到UV坐标的方差传递
    输入:
        X_w : 3D点 (世界坐标系) [3,]
        R   : 旋转矩阵 [3,3]
        t   : 平移向量 [3,]
        K   : 相机内参 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        Sigma_Xw : 3D点协方差 [3,3]
    输出:
        u, v : 投影坐标
        Sigma_uv : UV协方差 [2,2]
    """
    # 转换到相机坐标系
    R = rodrigues(rvec)
    X_c = R @ X_w + t
    Xc, Yc, Zc = X_c

    # 提取内参
    fx, fy = K[0, 0], K[1, 1]

    # 计算雅可比矩阵 J2
    inv_Zc = 1.0 / Zc
    J2 = np.array([
        [inv_Zc, 0, -Xc * inv_Zc ** 2],
        [0, inv_Zc, -Yc * inv_Zc ** 2]
    ])

    # 计算雅可比矩阵 J1
    J1 = np.array([[fx, 0], [0, fy]])

    # 完整雅可比
    J_u = J1 @ J2 @ R

    # 计算UV协方差
    Sigma_uv = J_u @ np.diag(Sigma_Xw) @ J_u.T

    # 计算投影坐标
    x_n = Xc / Zc
    y_n = Yc / Zc
    u = fx * x_n + K[0, 2]
    v = fy * y_n + K[1, 2]

    return [u, v], Sigma_uv

def project_points(pts_3d, rvec, tvec, K):
    """将3D点投影到2D图像平面"""
    R = rodrigues(rvec)
    pts_cam = (R @ pts_3d.T).T + tvec  # 转换为相机坐标系
    pts_cam = pts_cam / pts_cam[:, 2:]  # 归一化
    pts_2d = (K @ pts_cam.T).T  # 应用内参矩阵
    return pts_2d[:, :2]


def compute_reprojection_error(params, pts_3d, pts_2d, K, covariances_3d=None):
    """
    计算重投影误差（残差）
    如果提供3D点协方差矩阵，则使用马氏距离

    参数:
        params: 相机位姿参数 [rvec, tvec]
        pts_3d: 3D点坐标 (N×3)
        pts_2d: 2D观测点坐标 (N×2)
        K: 相机内参矩阵 (3×3)
        covariances_3d: 3D点协方差矩阵 (3×3×N)

    返回:
        残差向量 (当使用马氏距离时，返回每个点的马氏距离)
    """
    rvec = params[:3]
    tvec = params[3:6]
    # 如果没有提供协方差矩阵，直接返回扁平化的残差
    if covariances_3d is None:
        pts_proj = project_points(pts_3d, rvec, tvec, K)

        residuals = pts_proj - pts_2d
        return residuals.flatten()

    Sigma2d_list = []
    pts_proj_list = []
    mahalanobis_dist_list = []
    residuals_list = []
    for i in range(len(pts_3d)):
        pts_proj, Sigma2d = project_3d_to_uv(pts_3d[i], rvec, tvec, K, covariances_3d[i])
        # points_proj, jacobian_proj = cv2.projectPoints(pts_3d, rvec, tvec, K,distCoeffs=np.zeros((4, 1), dtype=np.float32))
        pts_proj_list.append(pts_proj)
        Sigma2d_list.append(Sigma2d)
        residuals = pts_proj - pts_2d[i]
        residuals_list.append(residuals)
        cov_inv = np.linalg.inv(Sigma2d)
        mahalanobis_dist_list.append(np.sqrt((residuals.T @ cov_inv @ residuals)))

    mahalanobis_residuals = np.stack(mahalanobis_dist_list)

    return mahalanobis_residuals


def numerical_jacobian(params, pts_3d, pts_2d, K, covariances=None, epsilon=1e-6):
    """数值方法计算雅可比矩阵（考虑协方差）"""
    n_params = len(params)  # 参数数量 (6)
    if covariances is None:
        n_residuals = len(pts_2d) * 2
    else:
        n_residuals = len(pts_2d)  # 残差数量 (每个点2个坐标)
    J = np.zeros((n_residuals, n_params))

    # 计算当前残差
    base_residuals = compute_reprojection_error(params, pts_3d, pts_2d, K, covariances)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += epsilon
        residuals_plus = compute_reprojection_error(params_plus, pts_3d, pts_2d, K, covariances)

        J[:, i] = (residuals_plus - base_residuals) / epsilon

    return J


def solve_pnp_lm(pts_3d, pts_2d, K, covariances=None, initial_rvec=None, initial_tvec=None, max_iter=10, tol=1e-8):
    """
    Levenberg-Marquardt 算法求解 PnP 问题（支持马氏距离）
    """
    # 初始化参数
    if initial_rvec is None:
        initial_rvec = np.zeros(3)
    if initial_tvec is None:
        initial_tvec = np.zeros(3)

    params = np.concatenate([initial_rvec, initial_tvec])

    # LM 参数
    lambda_ = 5e-2
    lambda_factor = 10
    last_error = float('inf')

    for iter in range(max_iter):
        # 计算当前误差和雅可比
        residuals = compute_reprojection_error(params, pts_3d, pts_2d, K, covariances)
        error = np.sum(residuals ** 2)/2

        # 检查收敛
        if abs(last_error - error) < tol:
            break

        J = numerical_jacobian(params, pts_3d, pts_2d, K, covariances)
        # J_ = []
        # n=200
        # for i in range(0, len(J), n):
        #     group = J[i:i + n]
        #     group_mean = group.mean(axis=0)
        #     J_.append(group_mean)
        #
        # J_ = np.array(J_)
        # 定义窗口大小 l (例如 l=20)
        l = 16
        # 计算每个方向上的桶数量 (图像范围 0~223)
        bins_x = int(np.ceil(224 / l))
        bins_y = int(np.ceil(224 / l))

        # 创建空桶: 三维列表 [bin_x][bin_y][点索引]
        buckets = [[[] for _ in range(bins_y)] for _ in range(bins_x)]

        # 将每个点分配到对应的桶中
        for i, (x, y) in enumerate(pts_2d):
            bin_x = min(int(x // l), bins_x - 1)
            bin_y = min(int(y // l), bins_y - 1)
            buckets[bin_x][bin_y].append(i)

        # 对每个桶内的点计算平均雅可比
        J_ = []
        for i in range(bins_x):
            for j in range(bins_y):
                if bucket_indices := buckets[i][j]:  # 非空桶
                    J_bucket = abs(J[bucket_indices])  # 获取桶内所有点的雅可比
                    J_.append(J_bucket.min(axis=0))  # 计算平均值
        # 将列表转换为NumPy数组
        J_ = np.array(J_)
        # 计算梯度
        grad = J.T @ residuals

        # 计算近似的 Hessian 矩阵
        H = J.T @ J
        # H = J_.T @ J_
        # LM 更新：添加阻尼因子
        diag_H = np.diag(np.diag(H))
        H_lm = H + lambda_ * diag_H

        H_=J_.T @ J_
        cov_p = np.linalg.inv(np.diag(np.diag(H_)))
        # 求解线性系统
        try:
            delta = np.linalg.solve(H_lm, -grad)
        except np.linalg.LinAlgError:
            # 如果求解失败，增加阻尼因子
            lambda_ *= lambda_factor
            continue

        # 尝试更新参数
        new_params = params + delta
        new_residuals = compute_reprojection_error(new_params, pts_3d, pts_2d, K, covariances)
        new_error = np.sum(new_residuals ** 2)/2

        # 检查误差是否减小
        if new_error < error:
            # 接受更新，减小阻尼因子
            params = new_params
            last_error = error
            # lambda_ /= lambda_factor
            print("接受更新")
        else:
            lambda_ /= lambda_factor
            print(lambda_, new_error , error)
            # break
    # 提取最终结果
    rvec = params[:3]
    tvec = params[3:6]
    # print(cov_p)

    # std_trans = np.sqrt(np.diag(cov_p[3:, 3:]).sum())
    # print(std_trans)
    std=np.sqrt(np.diag(cov_p))
    # rodrigues_to_euler(0)
    # J_deg=jacobian_finite_difference(rvec, type='q', epsilon=1e-6)
    # q_err=np.array([1,std[0],std[1],[2]])

    # print(2 * np.arcsin(np.linalg.norm(std[:3])) * 180 / 3.14159)
    # print(2*np.sqrt(np.diag(cov_p[:3, :3]).sum()) * 180 / 3.14159)
    # print(2 * np.sqrt(np.diag(cov_p[3:, 3:]).sum()))

    # print(np.sqrt((4*np.diag(J_deg @ cov_p[:3, :3] @ J_deg.T)[1:]).sum()) * 180 / 3.14159)
    #
    # var[:3] = np.sqrt(4*np.diag(J_deg @ cov_p[:3, :3] @ J_deg.T))[1:] * 180 / 3.14159

    return rvec, tvec, std


def jacobian_finite_difference(v, type='q', epsilon=1e-6):
    """
    使用有限差分法计算雅可比矩阵 deuler/dv

    参数:
        v : 罗德里格参数向量 [v1, v2, v3]
        epsilon : 有限差分步长 (可选)

    返回:
        J : 3x3 雅可比矩阵, J[i, j] = deuler_i / dv_j
    """
    # 基础欧拉角
    if type=='euler':
        euler_base = rodrigues_to_euler(v)
        J = np.zeros((3, 3))

        # 对每个罗德里格参数分量进行扰动
        for j in range(3):
            # 创建扰动向量
            v_perturbed = v.copy()
            v_perturbed[j] += epsilon

            # 计算扰动后的欧拉角
            euler_perturbed = rodrigues_to_euler(v_perturbed)

            # 计算有限差分导数
            J[:, j] = (euler_perturbed - euler_base) / epsilon
    elif type=='q':
        q_base = rodrigues_to_q(v)
        J = np.zeros((4, 3))

        # 对每个罗德里格参数分量进行扰动
        for j in range(3):
            # 创建扰动向量
            v_perturbed = v.copy()
            v_perturbed[j] += epsilon

            # 计算扰动后的欧拉角
            q_perturbed = rodrigues_to_q(v_perturbed)

            # 计算有限差分导数
            J[:, j] = (q_perturbed - q_base) / epsilon

    return J

def generate_3d_covariances(n_points, noise_level=1.0, correlation=0.0, seed=None):
    """
    生成3D点协方差矩阵列表

    参数:
        n_points: 点数
        noise_level: 噪声水平 (控制方差大小)
        correlation: 各维度之间的平均相关性 (-1到1)
        seed: 随机种子 (可选)

    返回:
        covariances: 3x3xN的协方差矩阵数组
    """
    if seed is not None:
        np.random.seed(seed)

    covariances = np.zeros((3, 3, n_points))

    for i in range(n_points):
        # 创建随机正定矩阵
        A = np.random.randn(3, 3)

        # 创建基本协方差矩阵
        cov_matrix = A.T @ A + np.eye(3) * 0.1  # 添加对角线元素确保正定性

        # 调整方差大小
        # 随机生成各维度的方差缩放因子
        var_scales = noise_level * (0.5 + np.random.rand(3))
        D = np.diag(np.sqrt(var_scales))
        cov_matrix = D @ cov_matrix @ D

        # 调整相关性
        if correlation != 0:
            # 创建目标相关结构
            target_corr = np.array([
                [1.0, correlation, correlation],
                [correlation, 1.0, correlation],
                [correlation, correlation, 1.0]
            ])

            # 将协方差矩阵转换为相关矩阵
            std_devs = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

            # 向目标相关矩阵移动
            alpha = 0.7  # 混合因子
            mixed_corr = alpha * target_corr + (1 - alpha) * corr_matrix

            # 转换回协方差矩阵
            cov_matrix = mixed_corr * np.outer(std_devs, std_devs)

        # 确保对称性
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        # 添加小扰动以确保正定性
        cov_matrix += np.eye(3) * 1e-6

        # 存储结果
        covariances[:, :, i] = cov_matrix

    return covariances

def generate_covariances(n_points, noise_level=1.0, correlation=0.0):
    """
    生成协方差矩阵列表
    参数:
        n_points: 点数
        noise_level: 噪声水平 (控制方差大小)
        correlation: x和y之间的相关性 (-1到1)
    """
    covariances = []
    for _ in range(n_points):
        # 创建协方差矩阵
        var_x = noise_level * (0.5 + np.random.random())  # 不同点的不同方差
        var_y = noise_level * (0.5 + np.random.random())
        cov_xy = correlation * np.sqrt(var_x * var_y)

        cov_matrix = np.array([[var_x, cov_xy],
                               [cov_xy, var_y]])
        covariances.append(cov_matrix)

    return covariances


def compute_pose_error(true_rvec, true_tvec, est_rvec, est_tvec):
    """
    计算姿态估计误差
    返回:
        rotation_error: 旋转角度误差（度）
        translation_error: 平移向量误差（米）
    """
    # 计算旋转误差
    true_R = rodrigues(true_rvec)
    est_R = rodrigues(est_rvec)

    # 计算相对旋转矩阵
    R_err = true_R.T @ est_R

    # 将旋转矩阵转换为旋转向量
    rvec_err, _ = cv2.Rodrigues(R_err)
    rotation_error = np.linalg.norm(rvec_err) * 180 / np.pi

    # 计算平移误差
    translation_error = np.linalg.norm(true_tvec - est_tvec)

    return rotation_error, translation_error


def generate_3d_grid_points(n_points=100):
    # 计算网格尺寸 (找到最接近立方体的整数尺寸)
    grid_size = int(round(n_points ** (1 / 3)))
    if grid_size ** 3 < n_points:
        grid_size += 1

    # 创建线性空间
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(1, 2, grid_size)

    # 创建网格
    xx, yy, zz = np.meshgrid(x, y, z)

    # 组合成点坐标
    pts_3d = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # 如果生成的点超过所需数量，随机选择
    if len(pts_3d) > n_points:
        indices = np.random.choice(len(pts_3d), n_points, replace=False)
        pts_3d = pts_3d[indices]

    return pts_3d[:n_points]  # 确保只返回所需数量的点

# 测试用例
if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    #
    improve_r=0
    improve_t=0
    for seed_idx in range(500):
        np.random.seed(seed_idx)
        # 创建虚拟相机内参
        K = np.array([[800, 0, 320],
                      [0, 800, 240],
                      [0, 0, 1]], dtype=float)

        # 创建虚拟3D点 (单位：米)
        n_points = 1000
        # 生成100个3D点
        pts_3d = generate_3d_grid_points(n_points)
        # pts_3d = np.array([
        #     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        #     [0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2],
        #     [0.5, 0.5, 1.5], [-0.5, 0.5, 1.5], [0.5, -0.5, 1.5], [-0.5, -0.5, 1.5]
        # ], dtype=np.float32)[:n_points]

        # 创建虚拟位姿
        true_rvec = np.array([0.3, -0.2, 0.1])
        true_tvec = np.array([0.1, -0.1, 0.5])

        # 投影点
        pts_2d = project_points(pts_3d, true_rvec, true_tvec, K)

        # 生成协方差矩阵（模拟不同点的不确定性）
        covariances = generate_3d_covariances(n_points, noise_level=0.01, correlation=0.0)

        # 添加与协方差相关的噪声
        noisy_pts_3d = []
        for i, point in enumerate(pts_3d):
            noise = np.random.multivariate_normal(mean=[0, 0, 0], cov=covariances[:,:,i])
            noisy_pts_3d.append(point + noise)
        noisy_pts_3d = np.array(noisy_pts_3d)

        # 使用OpenCV的solvePnP作为基准
        distCoeffs = np.zeros((4, 1), dtype=np.float32)
        _, rvec_cv, tvec_cv, inliers = cv2.solvePnPRansac(noisy_pts_3d, pts_2d, K, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec_cv = rvec_cv.flatten()
        tvec_cv = tvec_cv.flatten()

        # 使用我们的LM实现（带马氏距离）
        rvec_maha, tvec_maha = solve_pnp_lm(noisy_pts_3d, pts_2d, K, covariances, initial_rvec=rvec_cv, initial_tvec=tvec_cv)

        # 使用我们的LM实现（不带马氏距离）
        rvec_std, tvec_std = solve_pnp_lm(noisy_pts_3d, pts_2d, K)

        # 计算姿态误差
        cv_rot_err, cv_trans_err = compute_pose_error(true_rvec, true_tvec, rvec_cv, tvec_cv)
        maha_rot_err, maha_trans_err = compute_pose_error(true_rvec, true_tvec, rvec_maha, tvec_maha)
        std_rot_err, std_trans_err = compute_pose_error(true_rvec, true_tvec, rvec_std, tvec_std)

        # 打印结果
        print("=" * 60)
        print("真实位姿:")
        print(f"旋转向量: {true_rvec}")
        print(f"平移向量: {true_tvec}")
        print("=" * 60)

        print("\nOpenCV solvePnP 结果:")
        print(f"旋转向量: {rvec_cv}")
        print(f"平移向量: {tvec_cv}")
        print(f"旋转误差: {cv_rot_err:.4f} 度")
        print(f"平移误差: {cv_trans_err:.6f} 米")
        print("=" * 60)

        print("\n手写LM算法 (带马氏距离) 结果:")
        print(f"旋转向量: {rvec_maha}")
        print(f"平移向量: {tvec_maha}")
        print(f"旋转误差: {maha_rot_err:.4f} 度")
        print(f"平移误差: {maha_trans_err:.6f} 米")
        print("=" * 60)

        print("\n手写LM算法 (不带马氏距离) 结果:")
        print(f"旋转向量: {rvec_std}")
        print(f"平移向量: {tvec_std}")
        print(f"旋转误差: {std_rot_err:.4f} 度")
        print(f"平移误差: {std_trans_err:.6f} 米")
        print("=" * 60)

        # 比较结果
        rot_improvement = (std_rot_err - maha_rot_err) / std_rot_err * 100
        trans_improvement = (std_trans_err - maha_trans_err) / std_trans_err * 100
        improve_r+=rot_improvement
        improve_t+=trans_improvement
        print("\n马氏距离改进效果:")
        print(f"旋转精度提升: {rot_improvement:.2f}%")
        print(f"平移精度提升: {trans_improvement:.2f}%")
        print("=" * 60)

    print(f"all旋转精度提升: {improve_r/500:.2f}%")
    print(f"all平移精度提升: {improve_t/500:.2f}%")