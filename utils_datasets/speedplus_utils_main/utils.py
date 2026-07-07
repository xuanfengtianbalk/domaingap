import numpy as np
import json
import os
import cv2
import warnings
warnings.filterwarnings('ignore', message='Got processor for keypoints, but no transform to process it.')
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

# import tools.quaternion as Quat

# deep learning framework imports
try:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.preprocessing import image as keras_image

    has_tf = True
except ModuleNotFoundError:
    has_tf = False

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    has_pytorch = True
except ImportError:
    has_pytorch = False

from math import sin, acos
import math



class Camera:
    """" Utility class for accessing camera parameters. """

    speed_root = '/opt/dl_workspace/datasets/speedplus/speedplus/'
    _alt_root = 'datasets/speedplus/speedplus/'
    if not os.path.exists(os.path.join(speed_root, 'camera.json')):
        if os.path.exists(os.path.join(_alt_root, 'camera.json')):
            speed_root = _alt_root

    with open(os.path.join(speed_root, 'camera.json'), 'r') as f:
        camera_params = json.load(f)

    fx = camera_params['fx']  # focal length[m]
    fy = camera_params['fy']  # focal length[m]
    nu = camera_params['Nu']  # number of horizontal[pixels]
    nv = camera_params['Nv']  # number of vertical[pixels]
    ppx = camera_params['ppx']  # horizontal pixel pitch[m / pixel]
    ppy = camera_params['ppy']  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = camera_params['cameraMatrix']
    K = np.array(k)  # cameraMatrix
    dcoef = camera_params['distCoeffs']


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango_true'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['validation'].append(image['filename'])

    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q / np.linalg.norm(q)

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


def project(q, r):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = Camera.dcoef

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0

    # projection to image plane
    x = Camera.K[0, 0] * x1 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y1 + Camera.K[1, 2]

    return x, y


def myproject(p_axes, q, r):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    # p_axes = np.array([[0.4,0.4,0.32,1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = Camera.dcoef

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0

    # projection to image plane
    x = Camera.K[0, 0] * x1 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y1 + Camera.K[1, 2]

    return x, y


def myproject2(p_axes, q, r):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    # p_axes = np.array([[0.4,0.4,0.32,1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    x0, y0 = (points_camera_frame[0], points_camera_frame[1])

    # apply distortion
    dist = Camera.dcoef

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
    x1 = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
    y1 = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0

    # projection to image plane
    x = Camera.K[0, 0] * x1 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y1 + Camera.K[1, 2]

    return x, y, p_cam[2]


def quat_to_rot_matrix(q):
    """
    将四元数 (w, x, y, z) 转换为旋转矩阵 (3x3)
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def cam_to_object_pose(pose_cam_obj):
    """
    将物体在相机坐标系下的 pose 转换为相机在物体坐标系下的 pose。

    参数:
        pose_cam_obj : list or array of shape (7,)
                       [tx, ty, tz, qw, qx, qy, qz]
                       其中 (tx,ty,tz) 是物体原点在相机坐标系中的位置，
                       四元数 (qw,qx,qy,qz) 表示从相机坐标系到物体坐标系的旋转。

    返回:
        list of length 7: [tx_obj, ty_obj, tz_obj, qw_obj, qx_obj, qy_obj, qz_obj]
                          相机在物体坐标系下的位置和姿态（四元数表示从物体到相机的旋转）。
    """
    # 解析输入
    t_c_obj = np.asarray(pose_cam_obj[:3], dtype=float)
    q_c2o = np.asarray(pose_cam_obj[3:], dtype=float)
    # 归一化四元数
    q_c2o = q_c2o / np.linalg.norm(q_c2o)
    w, x, y, z = q_c2o

    # 1. 计算旋转矩阵 R_c2o
    R_c2o = quat_to_rot_matrix(q_c2o)

    # 2. 相机在物体坐标系中的位置
    t_o_cam = -R_c2o.T @ t_c_obj

    # 3. 相机在物体坐标系中的姿态（四元数共轭）
    q_o2c = np.array([w, -x, -y, -z])

    # 组合输出
    result = np.concatenate([t_o_cam, q_o2c])
    return result.tolist()

class SatellitePoseEstimationDataset:
    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='speedplus/'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        if split == 'train':
            img_name = os.path.join(self.root_dir, 'synthetic', 'train', 'images', img_name)
        elif split == 'validation':
            img_name = os.path.join(self.root_dir, 'synthetic', 'validation', 'images', img_name)
        elif split == 'sunlamp':
            img_name = os.path.join(self.root_dir, 'sunlamp', 'images', img_name)
        elif split == 'lightbox':
            img_name = os.path.join(self.root_dir, 'lightbox', 'images', img_name)
        else:
            print()
            # raise error?

        image = Image.open(img_name).convert('RGB')
        return image

    def get_image_name(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        if split == 'train':
            img_name = os.path.join(self.root_dir, 'synthetic', 'train', 'images', img_name)
        elif split == 'validation':
            img_name = os.path.join(self.root_dir, 'synthetic', 'validation', 'images', img_name)
        elif split == 'sunlamp':
            img_name = os.path.join(self.root_dir, 'sunlamp', 'images', img_name)
        elif split == 'lightbox':
            img_name = os.path.join(self.root_dir, 'lightbox', 'images', img_name)
        else:
            print()
            # raise error?

        # image = Image.open(img_name).convert('RGB')
        return img_name

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = project(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')
        return

    def visualize_q_r(self, img, q, r, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.imshow(img)

        xa, ya = project(q, r)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

    def visualize_box(self, i, ax=None):
        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test

        q, r = self.get_pose(i)
        xa, ya = project(q, r)

        # q = Quat.Quaternion(q[0], q[1], q[2], q[3])
        # z = Quat.Quaternion(0, 0, 0, 1)
        # q_ = q.inverse()
        # vector = q.__mul__(z).__mul__(q_)
        # v = np.array([vector.x, vector.y, vector.z])
        x1 = xa[0] + 2000 / np.linalg.norm(r)
        x2 = xa[0] - 2000 / np.linalg.norm(r)
        y1 = ya[0] + 2000 / np.linalg.norm(r)
        y2 = ya[0] - 2000 / np.linalg.norm(r)
        if x1 > 1920:
            x1 = 1920
        if x2 < 0:
            x2 = 0
        if y1 > 1200:
            y1 = 1200
        if y2 < 0:
            y2 = 0

        ax.add_line(plt.Line2D((x1, x1), (y1, y2), color='m'))
        ax.add_line(plt.Line2D((x2, x2), (y1, y2), color='r'))
        ax.add_line(plt.Line2D((x1, x2), (y1, y1), color='g'))
        ax.add_line(plt.Line2D((x1, x2), (y2, y2), color='b'))

        return

    def visualize_landmarks(self, p_axes, i, partition='train', ax=None):
        """ Visualizing image, with ground truth pose with axes projected to training image. """
        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            # print(q,r)
            x1, y1 = myproject(p_axes=p_axes, q=q, r=r)
            ax.plot(x1, y1, 'bo')
            # print(x1, y1)
            theta = 2 * acos(q[0])
            r1 = q[1] / sin(theta * 0.5) * theta
            r2 = q[2] / sin(theta * 0.5) * theta
            r3 = q[3] / sin(theta * 0.5) * theta
            print(r1, r2, r3)

        return x1, y1

    def visualize_q_r(self, img, q, r, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.imshow(img)

        xa, ya = project(q, r)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return

    def calculate_landmarks(self, p_axes, i, partition='train'):

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            x1, y1 = myproject(p_axes=p_axes, q=q, r=r)
        return x1, y1


# 可见性
# 获取基准点的下标,基准点是p[k]
def get_leftbottompoint(p):
    k = 0
    for i in range(1, len(p)):
        if p[i][1] < p[k][1] or (p[i][1] == p[k][1] and p[i][0] < p[k][0]):
            k = i
    return k


# 叉乘计算方法
def multiply(p1, p2, p0):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


# 获取极角，通过求反正切得出，考虑pi/2的情况
def get_arc(p1, p0):
    # 兼容sort_points_tan的考虑
    if (p1[0] - p0[0]) == 0:
        if ((p1[1] - p0[1])) == 0:
            return -1;
        else:
            return math.pi / 2
    tan = float((p1[1] - p0[1])) / float((p1[0] - p0[0]))
    arc = math.atan(tan)
    if arc >= 0:
        return arc
    else:
        return math.pi + arc


# 对极角进行排序,排序结果list不包含基准点
def sort_points_tan(p, pk):
    p2 = []
    for i in range(0, len(p)):
        p2.append({"index": i, "arc": get_arc(p[i], pk)})
    # print('排序前:',p2)
    p2.sort(key=lambda k: (k.get('arc')))
    # print('排序后:',p2)
    p_out = []
    for i in range(0, len(p2)):
        p_out.append(p[p2[i]["index"]])
    return p_out


def convex_hull(p):
    # p = list(set(p))
    # print('全部点:',p)
    k = get_leftbottompoint(p)
    pk = p[k]
    p.remove(p[k])
    # print('排序前去除基准点的所有点:',p,'基准点:',pk)

    p_sort = sort_points_tan(p, pk)  # 按与基准点连线和x轴正向的夹角排序后的点坐标
    # print('其余点与基准点夹角排序:',p_sort)
    p_result = [pk, p_sort[0]]

    top = 2
    for i in range(1, len(p_sort)):
        #####################################
        # 叉乘为正,向前递归删点;叉乘为负,序列追加新点
        while (multiply(p_result[-2], p_sort[i], p_result[-1]) > 0):
            p_result.pop()
        # 叉乘为负,序列追加新点
        p_result.append(p_sort[i])
    return p_result  # 测试


if has_pytorch:
    class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='datasets/', points=None, transform=None,
                     IS_VALID=False, padded=False, use_convex_hull=True,
                     consistency_layers=None):
            import albumentations as A
            # self.mask = A.Compose([A.CoarseDropout(max_holes=3, max_height=150, max_width=150, min_holes=2, min_height=50, min_width=50
            #              , fill_value=255, mask_fill_value=100,p=0.5)])
            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'validation', 'sunlamp', 'lightbox'}:
                raise ValueError(
                    'Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

            if split in {'train', 'validation'}:
                self.image_root = os.path.join(speed_root, 'synthetic', 'images')
                self.depth_root = os.path.join(speed_root, 'synthetic', 'depth')
                with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                    label_list = json.load(f)

            elif split in {'sunlamp','lightbox'}:
                self.image_root = os.path.join(speed_root, split, 'images')
                self.depth_root = os.path.join(speed_root, split, 'depth')
                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)


            self.sample_ids = [label['filename'] for label in label_list]
            self.train = split[-5:] == 'train'
            self.is_valid = IS_VALID
            if IS_VALID:
                self.train = False
            self.split = split
            self.Is_Fusion = False
            if self.Is_Fusion:
                self.region_root = os.path.join(speed_root, "synthetic", "region", "images")
                self.region_list = [imgname for imgname in os.listdir(self.region_root)]

            if self.train:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            elif self.split[-10:] == 'validation':
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            else:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            self.transform = transform
            self.padded = padded
            self.use_convex_hull = use_convex_hull
            self.points = points
            self.to_tensor = A.Compose([ToTensorV2(p=1.0)])
            self.consistency_layers = consistency_layers


        def get_labels(self):
            return self.labels

        def calculate_landmarks(self, p_axes, q, r, partition='train'):

            # no pose label for test
            if partition == 'train':
                x1, y1 = myproject(p_axes=p_axes, q=q, r=r)
            return [[x, y, 0] for x, y in zip(x1, y1)]

        def calculate_landmarks_distance(self, p_axes, q, r, partition='train'):

            # no pose label for test
            if partition == 'train':
                x1, y1, distance = myproject2(p_axes=p_axes, q=q, r=r)
            return [[x, y, 0] for x, y in zip(x1, y1)], distance

        def calculate_boxes_and_padded(self, y, padded=False):
            x1 = y[:, 0].min().item()
            x2 = y[:, 0].max().item()
            y1 = y[:, 1].min().item()
            y2 = y[:, 1].max().item()
            x_all = x2 - x1
            y_all = y2 - y1
            cx1, cx2 = max(x1, 0), min(x2, 1920)
            cy1, cy2 = max(y1, 0), min(y2, 1200)
            x_vis = cx2 - cx1
            y_vis = cy2 - cy1
            padded_ratio = 1 - (x_vis * y_vis) / (x_all * y_all) if x_all * y_all > 0 else 0
            x_add = x_all / 10
            y_add = y_all / 10
            box = [x1 - x_add, y1 - y_add, x2 + x_add, y2 + y_add]
            if not padded:
                box[0] = max(box[0], 0)
                box[1] = max(box[1], 0)
                box[2] = min(box[2], 1920)
                box[3] = min(box[3], 1200)
            return box, padded_ratio



        def calculate_world_coordinates(self, depth_map, pose, fx, fy, cx, cy):
            from scipy.spatial.transform import Rotation
            """
            输入:
                depth_map: (H, W) 深度图，单位米
                pose: 相机在世界坐标系下的位姿 [tx, ty, tz, qw, qx, qy, qz]
                fx, fy: 相机焦距（像素单位）
                cx, cy: 主点位置（像素单位）
            输出:
                world_coords: (H, W, 3)，世界坐标，深度无效的地方是 NaN
            """
            tx, ty, tz, qw, qx, qy, qz = cam_to_object_pose(pose)

            T = np.array([tx, ty, tz], dtype=np.float32)  # 平移
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # 四元数 -> 旋转矩阵

            H, W = depth_map.shape
            v, u = np.indices((H, W))
            u = u.astype(np.float32)
            v = v.astype(np.float32)

            # 初始化结果，全部设为 nan
            world_coords = np.full((H, W, 3), np.nan, dtype=np.float32)

            # 只保留有效的深度：0 < depth < 100
            valid_mask = (depth_map > 0) & (depth_map < 100)
            d = depth_map[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]

            # 计算相机坐标
            x_cam = (u_valid - cx) * d / fx
            y_cam = (v_valid - cy) * d / fy
            z_cam = d
            cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)  # (N, 3)

            # 相机系 -> 世界系
            world_coords_valid = (R @ cam_coords.T).T + T  # (N, 3)

            # 把有效的世界坐标填回对应位置
            world_coords[v_valid.astype(int), u_valid.astype(int), :] = world_coords_valid

            return world_coords

        def read_depth(self, path):
            import OpenEXR
            import Imath
            exr_file = OpenEXR.InputFile(path)
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = exr_file.channel('R', FLOAT)
            depth = np.frombuffer(r_str, dtype=np.float32).reshape(Camera.nv, Camera.nu)
            return depth

        def get_coors(self,depth_path,pose):
            # depth_path = os.path.join(self.depth_path, depth_name)
            width = Camera.nu
            height = Camera.nv
            depth_map = self.read_depth(depth_path)  # H, W
            depth_map = cv2.resize(depth_map, \
                                   (width, height), \
                                   interpolation=cv2.INTER_NEAREST)
            # self.cfg['model']['height']
            # self.cfg['model']['width']
            fx=Camera.fpx
            fy=Camera.fpy
            cx=Camera.K[0, 2]
            cy=Camera.K[1, 2]
            coors_ = self.calculate_world_coordinates(depth_map, pose, \
                                                 fx=fx, fy=fy,
                                                 cx=cx, cy=cy)  # 3, H, W
            coors = np.transpose(coors_, (2, 0, 1))
            return coors
        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            sample_id = self.sample_ids[idx]
            img_name = os.path.join(self.image_root, sample_id)

            pil_image = Image.open(img_name).convert('RGB')

            if self.labels is not None:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                kp, distance = self.calculate_landmarks_distance(p_axes=self.points, q=q, r=r)
                if self.use_convex_hull:
                    for index in range(len(kp)):
                        select_kp = distance[:-3] < distance[index]
                        kp_temp = [p for p, b in zip(kp, select_kp) if b == True]
                        kp_temp.append(kp[index])
                        if len(kp_temp) <= 3:
                            kp[index][2] = 1
                        else:
                            result = convex_hull(kp_temp)
                            for p in result:
                                if kp[index] == p:
                                    kp[index][2] = 1
                else:
                    for i in range(len(kp)):
                        kp[i][2] = 1

            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']

            np_image = np.array(pil_image)
            ymax, xmax, _ = np_image.shape

            # Layer 1: mark kp outside original image as invisible
            for i in range(len(kp)):
                if not (kp[i][0] > 0 and kp[i][1] > 0 and kp[i][0] < xmax and kp[i][1] < ymax):
                    kp[i][2] = 0
            target_dict = {}

            # --- compute bbox from original keypoints ---
            kp_orig = [[x, y, v] for x, y, v in kp]
            b, padded_ratio = self.calculate_boxes_and_padded(torch.tensor(kp_orig, dtype=torch.float32), padded=self.padded)
            target_dict["padded_ratio"] = padded_ratio
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            H_o, W_o = ymax, xmax

            is_aug = hasattr(self.transform, 'is_augmented') and self.transform.is_augmented

            # --- get coors/mask at full resolution ---
            if self.split == 'train' or self.split == 'validation':
                depth_path = os.path.join(self.depth_root, sample_id[:-4] + '0001.exr')
                coors = self.get_coors(depth_path, pose=[r[0], r[1], r[2], q[0], q[1], q[2], q[3]])
                mask = np.all(np.isfinite(coors), axis=0)

            # --- crop + resize image to 256x256 ---
            cx1, cy1 = max(x1, 0), max(y1, 0)
            cx2, cy2 = min(x2, W_o), min(y2, H_o)
            cropped_img = np_image[cy1:cy2, cx1:cx2]
            p_top = p_bottom = p_left = p_right = 0
            if self.padded:
                p_top, p_bottom = max(0, -y1), max(0, y2 - H_o)
                p_left, p_right = max(0, -x1), max(0, x2 - W_o)
                if p_top or p_bottom or p_left or p_right:
                    cropped_img = cv2.copyMakeBorder(cropped_img, p_top, p_bottom, p_left, p_right, cv2.BORDER_CONSTANT, value=0)
            resized_img = cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_LINEAR)

            # --- kp to 256 space for augment, crop space for none ---
            if is_aug:
                scale_x = 256.0 / (x2 - x1)
                scale_y = 256.0 / (y2 - y1)
                kp_trans = []
                for x, y, v in kp:
                    kp_trans.append([(x - x1) * scale_x, (y - y1) * scale_y, v])
            else:
                kp_trans = [[x - x1, y - y1, v] for x, y, v in kp]

            # --- crop coors/mask from full resolution, resize to 256 ---
            if self.split == 'train' or self.split == 'validation':
                cx1_o, cx2_o = max(x1, 0), min(x2, W_o)
                cy1_o, cy2_o = max(y1, 0), min(y2, H_o)
                coors_crop = coors[:, cy1_o:cy2_o, cx1_o:cx2_o]
                mask_crop = mask[cy1_o:cy2_o, cx1_o:cx2_o]
                if self.padded:
                    if p_top or p_bottom or p_left or p_right:
                        coors_crop = np.pad(coors_crop, ((0,0),(p_top,p_bottom),(p_left,p_right)), constant_values=np.nan)
                        mask_crop = np.pad(mask_crop, ((p_top,p_bottom),(p_left,p_right)), constant_values=0)
                coors_resized = np.transpose(cv2.resize(
                    np.transpose(coors_crop, (1, 2, 0)), (256, 256),
                    interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                mask_resized = cv2.resize(mask_crop.astype(np.uint8), (256, 256),
                                          interpolation=cv2.INTER_NEAREST)

                trans_image = self.transform(image=resized_img, keypoints=kp_trans,
                                              mask=mask_resized.astype(np.uint8),
                                              coors=np.transpose(coors_resized, (1, 2, 0)))
                new_coors = np.transpose(trans_image['coors'], (2, 1, 0))
                new_mask = trans_image['mask'].T
                target_dict["coors_gt"] = torch.tensor(new_coors)
                target_dict["mask_gt"] = torch.tensor(new_mask)
            else:
                trans_image = self.transform(image=resized_img, keypoints=kp_trans)
                # sunlamp/lightbox: compute coors_gt from depth, no augmentation
                if self.split in ('sunlamp', 'lightbox'):
                    depth_path = os.path.join(self.depth_root, sample_id[:-4] + '0001.exr')
                    coors = self.get_coors(depth_path, pose=[r[0], r[1], r[2], q[0], q[1], q[2], q[3]])
                    mask = np.all(np.isfinite(coors), axis=0)
                    cx1_o, cx2_o = max(x1, 0), min(x2, W_o)
                    cy1_o, cy2_o = max(y1, 0), min(y2, H_o)
                    coors_crop = coors[:, cy1_o:cy2_o, cx1_o:cx2_o]
                    mask_crop = mask[cy1_o:cy2_o, cx1_o:cx2_o]
                    coors_resized = np.transpose(cv2.resize(
                        np.transpose(coors_crop, (1, 2, 0)), (256, 256),
                        interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    mask_resized = cv2.resize(mask_crop.astype(np.uint8), (256, 256),
                                              interpolation=cv2.INTER_NEAREST)
                    target_dict["coors_gt"] = torch.tensor(coors_resized)
                    target_dict["mask_gt"] = torch.tensor(mask_resized)
            kp = trans_image['keypoints']
            final_image = self.transform.apply_norm(trans_image['image'])

            # --- keypoints to 256 space (train) or crop space (val) ---
            keypoints = []
            for x, y, v in kp:
                # in_bounds = x > 0 and y > 0 and x < (x2 - x1 - p_right) and y < (y2 - y1 - p_bottom) and x >= p_left and y >= p_top
                # keypoints.append([x, y, 1 if (v > 0 and in_bounds) else 0])
                keypoints.append([x, y, 0])
            k = torch.tensor(keypoints, dtype=torch.float32)
            k = torch.reshape(k, (-1, 3))

            b = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).reshape(1, 4)
            if self.split == 'train':
                target_dict["imageshape"] = torch.tensor([256, 256])
            else:
                target_dict["imageshape"] = torch.tensor([x2 - x1, y2 - y1], dtype=torch.float32)

            trans = self.to_tensor
            tt = trans(image=final_image)
            torch_image = tt['image']

            target_dict["boxes"] = b
            target_dict["labels"] = torch.ones(1, dtype=torch.int64)
            target_dict["keypoints"] = k.unsqueeze(0)
            torch_image = torch.transpose(torch_image, dim0=-2, dim1=-1)
            target_dict["q_gt"] = torch.tensor(q)
            target_dict["r_gt"] = torch.tensor(r)

            if self.consistency_layers is not None:
                variants = [torch_image]
                for layer in self.consistency_layers:
                    var = layer(torch_image.unsqueeze(0)).squeeze(0)
                    variants.append(var)
                return torch.stack(variants, dim=0), target_dict

            return torch_image, target_dict

else:
    class PyTorchSatellitePoseEstimationDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')


def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            den = np.sqrt((r[i, j] - g[i, j]) ** 2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg


points = np.array([[0.38,
                    0.385,
                    0.325,
                    1]
                      ,
                   [-0.37,
                    0.385,
                    0.325,
                    1]
                      ,
                   [0.38,
                    -0.385,
                    0.325,
                    1]
                      ,
                   [-0.37,
                    -0.385,
                    0.325,
                    1]
                      ,
                   [0.37,
                    0.30,
                    0,
                    1]
                      ,
                   [-0.37,
                    0.30,
                    0,
                    1]
                      ,
                   [0.355,
                    -0.265,
                    0,
                    1]
                      ,
                   [-0.37,
                    -0.30,
                    0,
                    1]
                      ,

                   [0.305,
                    -0.57,
                    0.255,
                    1]
                      ,
                   [0.55,
                    0.49,
                    0.26,
                    1]
                      ,
                   [-0.53,
                    0.48,
                    0.26,
                    1],
                   ])

# points = np.array([
#     [63038.3040090531,-65626.7342210694,54905.7341338851,1],
#     [62775.2814026001,65463.1291559213,54983.330145782,1],
#     [-62937.2192178545,65319.1922893251,54975.1920998896,1],
#     [-62800.4077847193,-65391.8693799259,54992.7599423682,1],
#     [62825.6149293275,-44774.0175340227,25.1176605517419,1],
#     [62899.9373677315,51629.0999246606,297.021114141113,1],
#     [-62657.9333359609,51842.9459335114,298.531666355505,1],
#     [-62524.0439660439,-44144.7970332338,517.586069664211,1],
#     [52159.0634884646,-98987.3183589307,42895.9483565026,1],
#     [92592.8519531914,83414.5186979327,43365.5933890869,1],
#     [-92997.2362991386,83175.5474630359,43323.5959547502,1]
#     ])/[200000,200000,200000,1]


import cv2

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, dic, view='box'):
    # image = image.copy()
    image = np.uint8(image)
    image = Image.fromarray(image)
    imagedraw = ImageDraw.Draw(image)
    if view == 'box':
        boxes = dic[:, :4]
        for x1, y1, x2, y2 in boxes:
            # for x1,y1,x2,y2 in k:
            imagedraw.rectangle([x1, y1, x2, y2], fill=None, outline="white", width=1)
    else:
        keypoints = dic["keypoints"]
        # print(boxes)
        print(keypoints)
        for k in keypoints:
            for x, y, v in k:
                if v:
                    imagedraw.ellipse((x - 10, y - 10, x + 10, y + 10), (0, 255, 255))
            # imagedraw.point((x,y),(0,255,255))
    image.show()


from tqdm import tqdm


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    for idx, (data, targets) in pbar:
        # print(data.shape)
        data = data.float()
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


try:
    from utils_datasets.rotation_parameter import points_trans_accord, quaternion2rot
    from utils_datasets.tools import splot
except ImportError:
    pass


def get_few_sample_train_set(loader, dis_list, num_viewpoint):
    # result_list = []
    temp1 = []
    temp2 = []
    rectangular_list = []

    for point in splot(r=dis_list, limit=num_viewpoint):
        rectangular_list.append(torch.tensor(point, dtype=float))
    rectangular = torch.stack(rectangular_list)

    for i in range(rectangular.shape[0]):
        temp1.append('1234567890123')
        temp2.append(1000.0)
    temp2 = torch.tensor(temp2, dtype=float)
    temp1 = np.array(temp1)
    pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    for idx, (data, targets) in pbar:
        index = 0
        for origin in targets["origin"]:
            quat, r = origin[:4], origin[-3:].view(3, -1)
            Rmats = quaternion2rot(quat)
            # for Rmat in Rmats:
            t = points_trans_accord(Rmats, r).squeeze()

            # 计算t与rectangular[:]的距离
            distance = ((rectangular - t) * (rectangular - t)).sum(-1)

            positive_index = distance < temp2
            temp2[positive_index] = distance[positive_index]
            temp1[positive_index] = targets["sample_id"][index]
            index += 1
            print(temp1[positive_index])
    return temp1

import torch.nn.functional as F


from cv2 import solvePnP, solvePnPRansac
def rotation_matrix_to_quaternion(r1, r2, r3):
    """
    r1, r2, r3: 每一行是旋转矩阵的行向量 (3,)
    返回: 四元数 (w, x, y, z)
    """
    from numpy import sin, cos, sqrt
    theta = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
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
def pose_calculate(cam_K, coors):
    object_points = []
    image_points = []
    H, W, _ = coors.shape
    for u in range(W):
        for v in range(H):
            # 提取有效像素点，忽略无效点（如NaN）
            if not np.isnan(coors[v, u, 0]):  # 如果3D坐标有效
                # 提取3D坐标
                x, y, z = coors[v, u, :3]
                object_points.append([x, y, z])
                image_points.append([u, v])
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
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
    return qvecs, tvecs

def visualize_dataset_sample(dataset, idx, save_path=None, coord_mode='channels'):
    """
    可视化 dataset 中的一个样本（适用于物体坐标系下的三维坐标场）。

    Args:
        dataset: PyTorchSatellitePoseEstimationDataset 实例
        idx: 样本索引
        save_path: 如果提供，保存图片到该路径；否则显示
        coord_mode: 坐标可视化模式，可选：
                    'channels' - 分别显示 X, Y, Z 三个通道（默认）
                    'rgb'      - 将归一化后的 (X,Y,Z) 编码为 RGB 彩色图
    """
    # 获取样本
    torch_image, target_dict = dataset[idx]

    # 1. 图像反归一化 (ImageNet 均值/标准差)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = torch_image.permute(2, 1, 0).cpu().numpy()   # (H, W, 3)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # 2. coors_gt: 原始形状 (3, W, H) -> 转为 (H, W, 3)
    coors = target_dict["coors_gt"].permute(2, 1, 0).cpu().numpy()      # (3, W, H)
    # coors = np.transpose(coors, (2, 1, 0))             # (H, W, 3)
    # 有效掩码 mask_gt: 原始形状 (W, H) -> 转为 (H, W)
    mask = target_dict["mask_gt"].permute(1, 0).cpu().numpy().astype(bool)  # (W, H)

    # mask = mask                                             # (H, W)
    # 创建带 NaN 的坐标数组（用于 channels 模式显示）
    coors_nan = coors.copy()
    coors_nan[~mask] = np.nan

    kp_full = target_dict["keypoints"].cpu().numpy().squeeze(0)
    imageshape = target_dict.get("imageshape")
    if imageshape is not None and imageshape[0].item() == 256:
        kp_display = kp_full.copy()
    else:
        crop_w = imageshape[0].item()
        crop_h = imageshape[1].item()
        kp_display = kp_full.copy()
        kp_display[:, 0] = kp_full[:, 0] * 256.0 / crop_w
        kp_display[:, 1] = kp_full[:, 1] * 256.0 / crop_h
    visible = kp_full[:, 2] > 0

    # 根据 coord_mode 创建不同数量的子图
    if coord_mode == 'channels':
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax_img, ax_x, ax_y, ax_mask, ax_overlay, ax_z = axes.flatten()
    else:  # 'rgb'
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_img, ax_coords_rgb, ax_mask, ax_overlay = axes.flatten()

    # ---------- 子图1: 原始图像 + 关键点 ----------
    ax_img.imshow(img)
    if visible.any():
        ax_img.scatter(kp_display[visible, 0], kp_display[visible, 1], c='red', s=20, marker='o', label='keypoints')
    ax_img.set_title(f"Image with keypoints (index {idx})")
    ax_img.axis('off')
    ax_img.legend()

    # ---------- 坐标可视化 ----------
    if coord_mode == 'channels':
        # 分别显示 X, Y, Z 通道
        channels = ['X (object)', 'Y (object)', 'Z (object)']
        for i, (ax, ch_name) in enumerate(zip([ax_x, ax_y, ax_z], channels)):
            ch_data = coors_nan[..., i]   # (H, W)
            # 计算有效数据的百分位数范围（忽略 NaN）
            valid_data = ch_data[~np.isnan(ch_data)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
            else:
                vmin, vmax = 0, 1
            im = ax.imshow(ch_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.set_title(ch_name)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:  # rgb mode
        # 将 (X,Y,Z) 归一化到 [0,1] 范围后作为 RGB 显示
        # 注意：仅在有效像素上计算 min/max，避免异常值影响
        coors_valid = coors[mask]   # (N, 3)
        if len(coors_valid) > 0:
            min_vals = coors_valid.min(axis=0)
            max_vals = coors_valid.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # 避免除零
            coors_norm = (coors - min_vals) / range_vals
        else:
            coors_norm = np.zeros_like(coors)
        # 将无效区域设为灰色 (0.5,0.5,0.5)
        rgb_img = coors_norm.copy()*255
        rgb_img[~mask] = 0.0
        ax_coords_rgb.imshow(rgb_img.astype(np.uint8))
        ax_coords_rgb.set_title("Object coordinates (RGB = X,Y,Z normalized)")
        ax_coords_rgb.axis('off')

    # ---------- 掩码显示 ----------
    ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title("Valid mask (from coors)")
    ax_mask.axis('off')

    # ---------- 叠加图: 图像 + 掩码(红色半透明) + 关键点 ----------
    img_overlay = img.copy()
    mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1).astype(np.float32)
    img_overlay = img_overlay * 0.6 + mask_rgb * 0.4
    ax_overlay.imshow(img_overlay)
    if visible.any():
        ax_overlay.scatter(kp_display[visible, 0], kp_display[visible, 1], c='lime', s=20, marker='o')
    ax_overlay.set_title("Image + mask (red=valid) + keypoints (green)")
    ax_overlay.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

    # ---------- 打印统计信息 ----------
    valid_coors = coors[mask]
    print(f"Sample {idx}:")
    print(f"  Image shape: {torch_image.shape}")
    print(f"  Valid coors ratio: {np.mean(mask):.3f}")
    if len(valid_coors) > 0:
        print(f"  Object coordinates range (X): [{valid_coors[:,0].min():.3f}, {valid_coors[:,0].max():.3f}]")
        print(f"  Object coordinates range (Y): [{valid_coors[:,1].min():.3f}, {valid_coors[:,1].max():.3f}]")
        print(f"  Object coordinates range (Z): [{valid_coors[:,2].min():.3f}, {valid_coors[:,2].max():.3f}]")
    print(f"  Keypoints shape: {kp_full.shape}")
    print(f"  q_gt: {target_dict['q_gt'].cpu().numpy()}")
    print(f"  r_gt: {target_dict['r_gt'].cpu().numpy()}")
    if 'boxes' in target_dict:
        print(f"  boxes: {target_dict['boxes'].cpu().numpy()}")


    ### aug test ###
    qvecs, tvecs = pose_calculate(Camera.K,  coors)
    qw, qx, qy, qz = target_dict['q_gt'].cpu().numpy()
    tx, ty, tz = target_dict['r_gt'].cpu().numpy()
    # tx, ty, tz, qw, qx, qy, qz = invert_pose(tx, ty, tz, qw, qx, qy, qz)
    print('pnp_t', tvecs)
    print('label_t', tx, ty, tz)

    print('pnp_q', qvecs)
    print('label_q', qw, qx, qy, qz)
    import torch
    pred_t = torch.tensor(tvecs).squeeze()
    pred_q = torch.tensor(qvecs).squeeze()
    gt_t = torch.tensor([tx, ty, tz])
    gt_q = torch.tensor([qw, qx, qy, qz])
    t_err, q_err = pose_error(pred_t, pred_q, gt_t, gt_q)
    print('t_err, q_err', t_err, q_err)
    ### aug test ###

if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    dataset_config = {
        'train_root_dir':  '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus',  # 修改为实际路径
    }
    # 假设 points 是必需的，如果不需要可设为 None 或空列表
    # points = None  # 或者提供具体的点集

    # 定义 transform（复制上面的 T 和 trans）
    import cv2
    import albumentations as A
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    config = {'TRAIN': {'P_AUG_SUN': 0.5}}  # 示例配置，根据实际调整

    # T = [
    #     # A.Resize(height=300, width=480, p=1),
    #     A.RandomBrightnessContrast(p=1),
    #     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=1,
    #                        border_mode=cv2.BORDER_CONSTANT, fill=255),
    #     A.OneOf([A.GaussNoise()], p=0.5),
    #     A.OneOf([
    #         A.MotionBlur(p=0.5),
    #         A.MedianBlur(blur_limit=3, p=0.5),
    #         A.Blur(blur_limit=3, p=0.5),
    #     ], p=1),
    #     A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=400,
    #                      num_flare_circles_range=(1, 2),
    #                      p=config['TRAIN']['P_AUG_SUN']),
    #     A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    # ]
    # trans = A.Compose(T,
    #                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    #                   additional_targets={
    #                       'mask': 'mask',
    #                       'coors': 'mask'
    #                   })
    from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
    trans = SpaceAugTransform('augbaseline')

    # 创建 dataset 实例
    dataset = PyTorchSatellitePoseEstimationDataset(
        split='train',
        speed_root=dataset_config['train_root_dir'],
        points=points,
        transform=trans,
        padded=True
    )

    # 可视化前几个样本
    for i in range(min(15, len(dataset))):
        visualize_dataset_sample(dataset, idx=i, save_path=f'sample_{i}.png', coord_mode='rgb')