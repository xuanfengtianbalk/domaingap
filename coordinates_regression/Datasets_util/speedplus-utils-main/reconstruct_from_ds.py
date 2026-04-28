import json
import numpy as np
import os
from pathlib import Path


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


# 更简洁的版本 - 假设输入是OpenCV，输出是Blender
def convert_opencv_to_blender(q, r):
    """从OpenCV坐标系转换到Blender坐标系"""

    # 构建物体到相机的变换 (OpenCV)
    R_obj_to_cam = quat2dcm(q).T
    t_obj_to_cam = np.array(r)

    T_obj_to_cam = np.eye(4)
    T_obj_to_cam[:3, :3] = R_obj_to_cam
    T_obj_to_cam[:3, 3] = t_obj_to_cam

    # 坐标系转换矩阵
    cv_to_bl = np.diag([1, -1, -1, 1])  # 翻转Y和Z

    # 正确的变换顺序：先转换坐标系，再求逆
    T_obj_to_cam_blender = cv_to_bl @ T_obj_to_cam

    # 返回相机到物体的变换（Blender中通常使用这个）
    T_cam_to_obj = np.linalg.inv(T_obj_to_cam_blender)

    return T_cam_to_obj.tolist()

def generate_simple_nerf_json(base_path, output_path="transforms.json"):
    """简化版本的JSON生成器，包含mask支持"""

    # 读取数据
    with open(os.path.join(base_path, 'camera.json'), 'r') as f:
        cam = json.load(f)
    with open(os.path.join(base_path, 'synthetic', 'train_tmp.json'), 'r') as f:
        train = json.load(f)

    # 构建输出
    output = {
        "camera_model": "OPENCV",
        "fl_x": cam['fx'] / cam['ppx'],
        "fl_y": cam['fy'] / cam['ppy'],
        "cx": cam['Nu'] / 2,
        "cy": cam['Nv'] / 2,
        "w": cam['Nu'],
        "h": cam['Nv'],
        "k1": cam['distCoeffs'][0],
        "k2": cam['distCoeffs'][1],
        "k3": cam['distCoeffs'][4] if len(cam['distCoeffs']) > 4 else 0.0,
        "k4": 0.0,
        "p1": cam['distCoeffs'][2],
        "p2": cam['distCoeffs'][3],
        "frames": []
    }

    for item in train:
        filename = item['filename']
        frame = {
            "file_path": f"images/{filename}",
            "transform_matrix": convert_opencv_to_blender(
                item['q_vbs2tango_true'],
                item['r_Vo2To_vbs_true']
            )
        }

        # 添加mask路径（假设命名约定）
        mask_name = f"{Path(filename).stem}_mask.png"
        frame["mask_path"] = f"masks/{mask_name}"

        output["frames"].append(frame)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"生成完成: {output_path}")
    print(f"包含 {len(output['frames'])} 帧，每帧都有mask路径")


# 使用
if __name__ == "__main__":
    base_path = "/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/"  # 修改为您的实际路径
    generate_simple_nerf_json(base_path,output_path="/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/synthetic/transforms.json")

# # 使用示例
# if __name__ == "__main__":
#     base_path = "/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/"  # 修改为您的实际路径
#     generate_nerf_json_advanced(base_path)