import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# 设置随机种子以保证可重复性
np.random.seed(42)


# ====================== 1. 生成模拟数据 ====================== #
def generate_synthetic_data(num_points=20, noise_std=0.5, add_lines=False):
    """生成合成相机位姿和3D-2D对应点"""
    # 创建3D点云 (在单位立方体内)
    X = np.random.rand(3, num_points) * 2 - 1
    X[2, :] += 5  # 在z方向偏移

    # 创建真实相机位姿
    true_rot = Rotation.from_euler('xyz', [15, -25, 10], degrees=True).as_matrix()
    true_trans = np.array([0.5, -0.3, 1.2])

    # 投影3D点到相机坐标系
    X_cam = true_rot @ X + true_trans.reshape(3, 1)

    # 转换为归一化图像坐标 (假设单位焦距)
    x = X_cam[:2, :] / X_cam[2, :]

    # 添加高斯噪声
    x_noisy = x + np.random.normal(0, noise_std, x.shape)

    # 创建协方差矩阵
    Sigmas = np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, num_points)) * 0.1
    Sigmas2D = np.tile(np.eye(2)[:, :, np.newaxis], (1, 1, num_points)) * noise_std ** 2

    # 生成线段数据（可选）
    Xs, Xe, l = None, None, None
    Sigmas3DLines, Sigmas2DLines = None, None

    if add_lines:
        # 选择两个点作为线段端点
        idx1, idx2 = np.random.choice(num_points, 2, replace=False)
        Xs = X[:, idx1].reshape(3, 1)
        Xe = X[:, idx2].reshape(3, 1)

        # 计算图像中的线方程
        p1 = x_noisy[:, idx1]
        p2 = x_noisy[:, idx2]
        line_dir = np.array([p2[1] - p1[1], p1[0] - p2[0], p2[0] * p1[1] - p1[0] * p2[1]])
        line_dir /= np.linalg.norm(line_dir[:2])
        l = line_dir.reshape(3, 1)

        # 线段协方差（简化）
        Sigmas3DLines = np.tile(np.eye(6)[:, :, np.newaxis], (1, 1, 1)) * 0.1
        Sigmas2DLines = np.array([noise_std ** 2])

    return {
        "X": X,
        "x": x_noisy,
        "true_R": true_rot,
        "true_t": true_trans,
        "Sigmas": Sigmas,
        "Sigmas2D": Sigmas2D,
        "Xs": Xs if add_lines else np.zeros((3, 0)),
        "Xe": Xe if add_lines else np.zeros((3, 0)),
        "l": l if add_lines else np.zeros((3, 0)),
        "Sigmas3DLines": Sigmas3DLines if add_lines else np.zeros((6, 6, 0)),
        "Sigmas2DLines": Sigmas2DLines if add_lines else np.zeros(0)
    }


# # ====================== 2. 位姿求解框架 ====================== #
# def dlsu_full(X, x, Sigmas, Sigmas2D, R_est_3, t_est_3,
#               Xs, Xe, l, Sigmas3DLines, Sigmas2DLines,
#               full_cov_mode=0):
#     """简化版的DLS(L)U求解器（实际应用中应替换为完整实现）"""
#     # 在实际应用中，这里应调用完整的DLS(L)U实现
#     # 为示例目的，我们使用EPnP算法作为替代
#     from sklearn.decomposition import PCA
#
#     # 中心化3D点
#     meanX = np.mean(X, axis=1, keepdims=True)
#     X_centered = X - meanX
#
#     # 使用PCA估计初始旋转
#     pca = PCA(n_components=3)
#     pca.fit(X_centered.T)
#     R_pca = pca.components_.T
#
#     # 计算初始平移
#     t_init = -R_pca @ meanX
#
#     # 创建多个候选解
#     num_solutions = 3
#     Rs = np.zeros((3, 3, num_solutions))
#     ts = np.zeros((3, num_solutions))
#     costs = np.zeros(num_solutions)
#
#     for i in range(num_solutions):
#         # 添加随机扰动
#         angle = np.random.uniform(-5, 5, 3)
#         R_perturb = Rotation.from_euler('xyz', angle, degrees=True).as_matrix()
#         Rs[:, :, i] = R_perturb @ R_pca
#
#         # 计算平移
#         ts[:, i] = t_init.flatten() + np.random.uniform(-0.1, 0.1, 3)
#
#         # 计算重投影误差作为"成本"
#         X_proj = Rs[:, :, i] @ X + ts[:, i].reshape(3, 1)
#         x_proj = X_proj[:2, :] / X_proj[2, :]
#         costs[i] = np.mean(np.linalg.norm(x_proj - x, axis=0))
#
#     return Rs, ts


# ====================== 3. 位姿协方差估计 ====================== #
def estimate_pose_covariance(R, t, sigma2=1.0):
    """
    估计位姿的协方差矩阵（简化版）
    在实际应用中，应使用基于海森矩阵的方法
    """
    # 在实际应用中，这里应使用逆海森矩阵方法
    # 为示例目的，我们基于重投影误差估计协方差

    # 生成扰动位姿
    num_perturb = 100
    perturbed_Rs = []
    perturbed_ts = []

    for _ in range(num_perturb):
        # 添加旋转扰动
        angle_std = 0.5  # 度
        angle_perturb = np.random.normal(0, angle_std, 3)
        R_perturb = Rotation.from_euler('xyz', angle_perturb, degrees=True).as_matrix()
        R_perturbed = R_perturb @ R

        # 添加平移扰动
        trans_std = 0.01
        t_perturbed = t + np.random.normal(0, trans_std, 3)

        perturbed_Rs.append(R_perturbed)
        perturbed_ts.append(t_perturbed)

    # 计算位姿参数
    pose_params = []
    for R_pert, t_pert in zip(perturbed_Rs, perturbed_ts):
        # 使用旋转向量作为姿态参数
        rot_vec = Rotation.from_matrix(R_pert).as_rotvec()
        pose_params.append(np.concatenate([rot_vec, t_pert]))

    pose_params = np.array(pose_params)

    # 计算协方差
    cov_matrix = np.cov(pose_params.T)

    return cov_matrix

from init import dlsu_full
def dlsu_full_with_covariance(X, x, Sigmas):
    """
    带协方差估计的DLS(L)U求解器
    """
    # 求解位姿
    Rs, ts = dlsu_full(X, x, Sigmas)

    covariances = []
    valid_indices = []

    # 估计每个解的协方差
    for i in range(Rs.shape[2]):
        R = Rs[:, :, i]
        t = ts[:, i]

        # 检查物理可行性（所有点在相机前方）
        X_cam = R @ X + t.reshape(3, 1)
        if np.all(X_cam[2, :] > 0):
            cov_matrix = estimate_pose_covariance(R, t, sigma2)
            covariances.append(cov_matrix)
            valid_indices.append(i)

    # 过滤有效解
    Rs_valid = Rs[:, :, valid_indices]
    ts_valid = ts[:, valid_indices]

    return Rs_valid, ts_valid, covariances


# ====================== 4. 可视化函数 ====================== #
def plot_camera_pose(ax, R, t, color='r', label='Estimated', scale=0.5):
    """在3D图中绘制相机位姿"""
    # 相机坐标系轴
    axes = np.eye(3) * scale
    rot_axes = R @ axes

    # 绘制相机位置
    ax.scatter(t[0], t[1], t[2], c=color, s=50, label=label)

    # 绘制相机坐标系轴
    for i in range(3):
        ax.quiver(t[0], t[1], t[2],
                  rot_axes[0, i], rot_axes[1, i], rot_axes[2, i],
                  color=color, linewidth=2)

    # 绘制相机视锥体（简化）
    points = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]]) * scale
    rot_points = R @ points.T + t.reshape(3, 1)
    rot_points = rot_points.T

    # 连接点形成视锥体
    for i in range(4):
        ax.plot([rot_points[4, 0], rot_points[i, 0]],
                [rot_points[4, 1], rot_points[i, 1]],
                [rot_points[4, 2], rot_points[i, 2]], color=color, alpha=0.5)

    ax.plot([rot_points[0, 0], rot_points[1, 0], rot_points[2, 0], rot_points[3, 0], rot_points[0, 0]],
            [rot_points[0, 1], rot_points[1, 1], rot_points[2, 1], rot_points[3, 1], rot_points[0, 1]],
            [rot_points[0, 2], rot_points[1, 2], rot_points[2, 2], rot_points[3, 2], rot_points[0, 2]],
            color=color, alpha=0.5)


def plot_uncertainty_ellipsoid(ax, mean, cov, color='r', n_std=2.0, alpha=0.2):
    """绘制位置不确定性椭球体"""
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov[3:, 3:])

    # 椭球半径
    radii = n_std * np.sqrt(eigvals)

    # 生成椭球点
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # 旋转椭球
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = eigvecs @ [x[i, j], y[i, j], z[i, j]] + mean

    # 绘制椭球
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_rotation_uncertainty(ax, mean, cov, color='r', n_std=1.0):
    """绘制旋转不确定性（圆锥体）"""
    # 提取旋转部分协方差
    rot_cov = cov[:3, :3]

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(rot_cov)

    # 最大不确定性方向
    max_dir = eigvecs[:, np.argmax(eigvals)]
    max_std = np.sqrt(np.max(eigvals))

    # 转换为角度（度）
    angle_deg = n_std * max_std * 180 / np.pi

    # 绘制圆锥
    # 在实际应用中，应绘制三维圆锥，这里简化显示
    ax.text(mean[0], mean[1], mean[2], f"{angle_deg:.1f}°", color=color)


# ====================== 5. 主程序 ====================== #
# 生成合成数据
data = generate_synthetic_data(num_points=20, noise_std=0.5, add_lines=False)
true_R, true_t = data["true_R"], data["true_t"]

# 求解位姿及协方差
Rs, ts, covariances = dlsu_full_with_covariance(
    data["X"], data["x"],
    data["Sigmas"]
)

# 打印结果
print(f"找到 {len(covariances)} 个有效解")
for i, (R, t, cov) in enumerate(zip(np.rollaxis(Rs, 2), ts.T, covariances)):
    print(f"\n解 {i + 1}:")
    print(f"旋转矩阵:\n{R}")
    print(f"平移向量: {t}")

    # 计算与真实位姿的误差
    rot_error = Rotation.from_matrix(R.T @ true_R).magnitude() * 180 / np.pi
    trans_error = np.linalg.norm(t - true_t)
    print(f"旋转误差: {rot_error:.2f}°, 平移误差: {trans_error:.4f}m")

    # 分析协方差
    rot_cov = cov[:3, :3]
    trans_cov = cov[3:, 3:]

    rot_std = np.sqrt(np.diag(rot_cov)) * 180 / np.pi
    trans_std = np.sqrt(np.diag(trans_cov))

    print("旋转标准差 (度):", np.round(rot_std, 2))
    print("平移标准差 (米):", np.round(trans_std, 4))

# ====================== 6. 可视化 ====================== #
fig = plt.figure(figsize=(15, 10))

# 3D场景图
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("3D场景与相机位姿")

# 绘制3D点
ax1.scatter(data["X"][0], data["X"][1], data["X"][2], c='b', label='3D点')

# 绘制真实相机位姿
plot_camera_pose(ax1, true_R, true_t, color='g', label='真实位姿')

# 绘制估计相机位姿及不确定性
for i, (R, t, cov) in enumerate(zip(np.rollaxis(Rs, 2), ts.T, covariances)):
    color = 'r' if i == 0 else 'orange'
    label = '估计位姿' if i == 0 else f'候选解 {i}'
    plot_camera_pose(ax1, R, t, color=color, label=label)

    # 绘制位置不确定性椭球
    plot_uncertainty_ellipsoid(ax1, t, cov, color=color)

    # 绘制旋转不确定性
    plot_rotation_uncertainty(ax1, t, cov, color=color)

# 设置坐标轴
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
ax1.view_init(elev=20, azim=-45)

# 重投影误差图
ax2 = fig.add_subplot(122)
ax2.set_title("重投影误差")

for i, (R, t) in enumerate(zip(np.rollaxis(Rs, 2), ts.T)):
    # 计算重投影
    X_cam = R @ data["X"] + t.reshape(3, 1)
    x_proj = X_cam[:2, :] / X_cam[2, :]

    # 计算误差向量
    errors = x_proj - data["x"]

    # 绘制误差向量
    for j in range(errors.shape[1]):
        dx, dy = errors[:, j]
        ax2.quiver(data["x"][0, j], data["x"][1, j], dx, dy,
                   angles='xy', scale_units='xy', scale=1,
                   color='r' if i == 0 else 'orange', alpha=0.5)

# 绘制2D点
ax2.scatter(data["x"][0], data["x"][1], c='b', s=30, label='观测点')

# 添加误差椭圆（简化）
for j in range(data["x"].shape[1]):
    cov = data["Sigmas2D"][:, :, j]
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(eigvals)

    ellipse = plt.matplotlib.patches.Ellipse(
        xy=data["x"][:, j],
        width=width,
        height=height,
        angle=angle,
        alpha=0.2,
        color='blue'
    )
    ax2.add_patch(ellipse)

ax2.set_xlabel('图像X')
ax2.set_ylabel('图像Y')
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()