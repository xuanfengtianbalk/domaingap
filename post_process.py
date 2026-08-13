import torch
import numpy as np
from numpy import sin, cos, sqrt
import cv2
from cv2 import solvePnP, solvePnPRansac
from ucer_3d_pnp.LM import solve_pnp_lm


def mask_to_display(mask_tensor):  # (W,H) -> (H,W)
    return mask_tensor.cpu().numpy().T

def to_pnp_coors(tensor):  # (C,W,H) -> (H,W,C)
    return tensor.permute(2, 1, 0).cpu().detach().numpy()

def to_pnp_coors_np(arr):  # (C,W,H) -> (W,H,C)
    return np.transpose(arr, (1, 2, 0))

def coors_gs_to_pnp(array):  # (H,W,3) -> (W,H,3)
    return np.transpose(array, (1, 0, 2))
from math_.q_ import quatProduct
import math
def rotation_matrix_to_quaternion(r1, r2, r3):
    """
    r1, r2, r3: 每一行是旋转矩阵的行向量 (3,)
    返回: 四元数 (w, x, y, z)
    """

    theta = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
    if theta < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    qw = cos(theta * 0.5)
    qx = r1 * sin(theta * 0.5) / theta
    qy = r2 * sin(theta * 0.5) / theta
    qz = r3 * sin(theta * 0.5) / theta
    quat_wxyz = [qw, qx, qy, qz]
    return quat_wxyz
def pose_error(pred_t, pred_q, gt_t, gt_q):
    """
    输入:
        pred_t: (B, 3) 预测平移向量
        pred_q: (B, 4) 预测四元数 [qw, qx, qy, qz]
        gt_t:   (B, 3) GT平移向量
        gt_q:   (B, 4) GT四元数 [qw, qx, qy, qz]
    输出:
        t_err: (B,) 平移误差 (单位: 米，或跟你的单位一致)
        q_err: (B,) 旋转误差 (单位: 度)
    """
    # 保证四元数是单位四元数
    pred_q = torch.nn.functional.normalize(pred_q, dim=-1)
    gt_q = torch.nn.functional.normalize(gt_q, dim=-1)

    # 平移误差 (L2距离)
    t_err = torch.norm(pred_t - gt_t, dim=-1)  # (B,)

    # 旋转误差 (角度)
    # cos(theta/2) = |dot(q1, q2)|
    cos_theta_half = torch.abs(torch.sum(pred_q * gt_q, dim=-1))  # (B,)
    cos_theta_half = torch.clamp(cos_theta_half, -1.0, 1.0)  # 数值稳定
    theta = 2 * torch.acos(cos_theta_half)  # 弧度制
    q_err = torch.rad2deg(theta)  # 转为角度制

    return t_err, q_err

def compute_pose_error(qvecs, tvecs, qgt, rgt, is_true):
    # 以下为原始代码片段（仅将列表追加操作改为返回值）
    if is_true:
        # 四元数计算误差角
        qvecs = torch.tensor(qvecs).reshape(-1)
        tvecs = torch.tensor(tvecs).reshape(-1)
        qvecs = qvecs / torch.norm(qvecs)  # 归1化
        qgt_norm = qgt / torch.norm(qgt)   # 归1化
        # 计算误差角度
        qgt_ = qgt_norm * torch.tensor([1.0, -1.0, -1.0, -1.0])
        q_ = quatProduct(qgt_.type_as(qvecs), qvecs)
        q_ = q_ / torch.norm(q_)  # 归1化
        err_ori = 2 * torch.arccos(abs(q_[0])) * 180 / math.pi
        if err_ori < 0.169:
            err_ori = torch.tensor(0)
        # 计算误差距离
        err_r_abs = torch.norm(tvecs - rgt)

        err_r_rel = torch.norm(tvecs - rgt) / torch.norm(rgt)
        # print(err_r_rel, err_ori)
        if err_r_rel<2.173e-3:
            err_r_rel=torch.tensor(0)
        ###   记录计算平均值   ###
        err_ori_rad = 2 * torch.arccos(abs(q_[0]))
        err_ori_deg = err_ori_rad * 180 / math.pi
        # err_r = torch.norm(tvecs - rgt) / torch.norm(rgt)
        err_pose = err_ori_rad + err_r_rel
        # err_pose_list.append(err_pose)   # 移到外部
        # err_rot_list.append(err_ori_deg)
        # err_trans_list.append(err_r)
        inc_fail_miss = False   # 有效估计不增加失败/丢失计数
    else:
        qest = torch.tensor([1, 0, 0, 0])
        tvecs = torch.tensor([0, 0, 5])
        # 计算误差角度
        qgt_ = qgt * torch.tensor([1.0, -1.0, -1.0, -1.0])
        q_ = quatProduct(qgt_.type_as(qest), qest)
        # err_ori = 2 * torch.arccos(abs(q_[0])) * 180 / math.pi
        # 计算误差距离
        tvecs = torch.tensor(tvecs).reshape(-1)
        err_r_abs = torch.norm(tvecs - rgt)
        # print('false_epnp', target["filename"])
        # miss_idx += 1   # 移到外部
        ###   记录计算平均值   ###
        err_ori_rad = 2 * torch.arccos(abs(q_[0]))
        err_ori_deg = err_ori_rad * 180 / math.pi
        err_r_rel = torch.norm(tvecs - rgt) / torch.norm(rgt)
        err_pose = err_ori_rad + err_r_rel
        # err_pose_list.append(err_pose)
        # err_rot_list.append(err_ori_deg)
        # err_trans_list.append(err_r)
        # fail_idx += 1
        inc_fail_miss = True   # 无效估计需要增加失败/丢失计数

    # 条件判断（原代码最后部分）
    good_pose = (err_ori_deg <= 5) and (err_r_rel < 0.05)
    # deg_cm_5_5_idx += 1   # 移到外部

    # 返回所有需要外部累加的参数
    return err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose

def pose_calculats_from_coors(cam_K, coors, gtbbox):
    object_points = []
    image_points = []
    resize_H, resize_W, _ = coors.shape

    x_min, y_min = gtbbox[0], gtbbox[1]
    crop_orig_W = gtbbox[2] - gtbbox[0]
    crop_orig_H = gtbbox[3] - gtbbox[1]
    scale_x = resize_W / crop_orig_W
    scale_y = resize_H / crop_orig_H

    for u in range(resize_W):
        for v in range(resize_H):
            if not np.isnan(coors[v, u, 0]):
                x, y, z = coors[v, u, :3]
                object_points.append([x, y, z])
                u_crop = u / scale_x
                v_crop = v / scale_y
                u_orig = u_crop + x_min
                v_orig = v_crop + y_min
                image_points.append([u_orig, v_orig])


                # image_points.append([u, v])
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    distCoeffs = np.zeros((4, 1), dtype=np.float32)
    is_true, (r1, r2, r3), tvecs, inliers = solvePnPRansac(object_points, \
                                                     image_points, \
                                                     cam_K, \
                                                     distCoeffs,
                                                     iterationsCount=100,
                                                     confidence=0.99,
                                                     reprojectionError=8,
                                                     flags=cv2.SOLVEPNP_ITERATIVE
                                                     )
    # _, (r1, r2, r3), tvecs = solvePnP(object_points, \
    #                                                  image_points, \
    #                                                  cam_K, \
    #                                                  distCoeffs,
    #                                   flags=cv2.SOLVEPNP_ITERATIVE
    #                                                  )
    qvecs = rotation_matrix_to_quaternion(r1, r2, r3)
    # tx, ty, tz, qw, qx, qy, qz = pose
    # tx, ty, tz, qw, qx, qy, qz = invert_pose(tx, ty, tz, qw, qx, qy, qz)
    # print('pnp_t', tvecs)
    # print('label_t', tx, ty, tz)
    #
    # print('pnp_q', qvecs)
    # print('label_q', qw, qx, qy, qz)
    if is_true:
        return is_true, qvecs, tvecs
    else:
        return is_true, None, None

def pose_calculate_with_unc(cam_K, coors, unc, gtbbox=None):
    """Uncertainty-weighted PnP: RANSAC init + LM refinement with per-pixel 3D unc.

    coors: (H, W, 3) world coordinates per pixel
    unc:   (H, W, 3) per-axis uncertainty (std) per pixel
    gtbbox: optional (x_min, y_min, x_max, y_max) — map image points back
            to original resolution like pose_calculats_from_coors
    Returns (qvecs, tvecs, std)
    """
    object_points = []
    image_points = []
    unc_points = []
    H, W, _ = coors.shape
    if gtbbox is not None:
        x_min, y_min = gtbbox[0], gtbbox[1]
        crop_orig_W = gtbbox[2] - gtbbox[0]
        crop_orig_H = gtbbox[3] - gtbbox[1]
        scale_x = W / crop_orig_W
        scale_y = H / crop_orig_H
    for u in range(W):
        for v in range(H):
            # 提取有效像素点，忽略无效点（如NaN）
            if not np.isnan(coors[v, u, 0]):  # 如果3D坐标有效
                # 提取3D坐标
                x, y, z = coors[v, u, :3]
                dx, dy, dz = unc[v, u, :3]
                object_points.append([x, y, z])
                unc_points.append([dx, dy, dz])
                if gtbbox is not None:
                    image_points.append([u / scale_x + x_min, v / scale_y + y_min])
                else:
                    image_points.append([u, v])
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    unc_points = np.array(unc_points, dtype=np.float32)
    unc_points = np.maximum(unc_points, 1e-6)  # floor for matrix invertibility
    distCoeffs = np.zeros((4, 1), dtype=np.float32)
    _, (r1, r2, r3), tvecs, inliers = solvePnPRansac(object_points, \
                                                     image_points, \
                                                     cam_K, \
                                                     distCoeffs,
                                                     iterationsCount=100,
                                                     confidence=0.99,
                                                     reprojectionError=8,
                                                     flags=cv2.SOLVEPNP_ITERATIVE
                                                     )
    std = np.zeros(6)
    (r1, r2, r3), tvecs, std = solve_pnp_lm(object_points[inliers[:, 0]], \
                                            image_points[inliers[:, 0]], \
                                            cam_K, \
                                            unc_points[inliers[:, 0]], initial_rvec=np.array([r1[0], r2[0], r3[0]]), initial_tvec=tvecs[:, 0],
                                            max_iter=1)
    qvecs = rotation_matrix_to_quaternion(r1, r2, r3)
    return qvecs, tvecs, std


def pose_calculats_from_kps(cam_K, p_body, p_frame):

    for err_index in range(6):
        is_true, (r1, r2, r3), tvecs, inliers = solvePnPRansac(p_body[:, :-1].astype("float32"),
                                                               p_frame[:, :-1].to("cpu").detach().numpy().astype(
                                                                   "float32"),
                                                               np.array(cam_K),
                                                               None,
                                                               useExtrinsicGuess=False, iterationsCount=100,
                                                               reprojectionError=8 * (err_index + 1),
                                                               confidence=0.95,
                                                               flags=cv2.SOLVEPNP_EPNP)

    qvecs = rotation_matrix_to_quaternion(r1, r2, r3)
    # tx, ty, tz, qw, qx, qy, qz = pose
    # tx, ty, tz, qw, qx, qy, qz = invert_pose(tx, ty, tz, qw, qx, qy, qz)
    # print('pnp_t', tvecs)
    # print('label_t', tx, ty, tz)
    #
    # print('pnp_q', qvecs)
    # print('label_q', qw, qx, qy, qz)
    if is_true:
        return is_true, qvecs, tvecs
    else:
        return is_true, None, None
