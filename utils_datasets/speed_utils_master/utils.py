import numpy as np
import json
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import cv2
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


class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


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

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
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

    # projection to image plane
    x = Camera.K[0, 0] * x0 + Camera.K[0, 2]
    y = Camera.K[1, 1] * y0 + Camera.K[1, 2]

    return x, y

class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

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


if has_pytorch:
    class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='', points=None, transform=None):

            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'train_train', 'val', 'test', 'real_test'}:
                raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

            with open(os.path.join(speed_root, split + '.json'), 'r') as f:
                label_list = json.load(f)

            self.sample_ids = [label['filename'] for label in label_list]
            self.train = split[:5] == 'train'

            if self.train or split == 'val':
                self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            if split == 'val':
                self.image_root = os.path.join(speed_root, 'images/train')
            elif self.train:
                self.image_root = os.path.join(speed_root, 'images/train')
            else:
                self.image_root = os.path.join(speed_root, 'images', split)
            self.transform = transform
            self.points = points
        def calculate_landmarks(self, p_axes, q, r, partition='train'):

            # no pose label for test
            if partition == 'train':
                x1, y1 = myproject(p_axes=p_axes, q=q, r=r)
            return [(x,y)for x,y in zip(x1, y1)]

        def calculate_boxes(self, y, partition='train'):
            if partition == 'train':
                x1=9999
                x2=0
                y1=9999
                y2=0
                for b in y:
                    if b[0]<x1:x1 = b[0]
                    if b[0]>x2:x2 = b[0]
                    if b[1]<y1:y1 = b[1]
                    if b[1]>y2:y2 = b[1]
            return [x1, y1, x2, y2]
        def whitening(self, img):
            img = img/255.0
            m, dev = cv2.meanStdDev(img)  # 返回均值和方差，分别对应3个通道
            img[:, :] = (img[:, :] - m[0]) / (dev[0] + 1e-6)

            # 将 像素值 低于 值域区间[0, 255] 的 像素点 置0
            img = img * 255
            img *= (img > 0)
            # 将 像素值 高于 值域区间[0, 255] 的 像素点 置255
            img = img * (img <= 255) + 255 * (img > 255)
            img = img.astype(np.uint8)
            return img

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            sample_id = self.sample_ids[idx]
            img_name = os.path.join(self.image_root, sample_id)

            # note: despite grayscale images, we are converting to 3 channels here,
            # since most pre-trained networks expect 3 channel input
            pil_image = Image.open(img_name).convert('RGB')
            np_image = np.array(pil_image)
            if self.train:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                kp = self.calculate_landmarks(p_axes=self.points, q=q, r=r)
            else:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                kp = self.calculate_landmarks(p_axes=self.points, q=q, r=r)

            if self.transform is not None:
                trans_image = self.transform(image=np_image, keypoints=kp)
                torch_image=trans_image['image']
                kp=trans_image['keypoints']


                keypoints = []
                for x, y in kp:
                    if x > 0 and y > 0 and x < 1920 and y < 1200:
                        keypoints.append([x, y, 1])
                    else:
                        keypoints.append([x, y, 0])
                k = torch.tensor(keypoints, dtype=torch.float32)
                b = self.calculate_boxes(k)

                k = torch.reshape(k,(-1,3))
                b = torch.reshape(torch.tensor(b), (-1, 4))

                target_dict = {}
                target_dict["boxes"] = b
                target_dict["labels"] = torch.ones(1,dtype=torch.int64)
                target_dict["keypoints"] = k.unsqueeze(0)
                # target_dict = dict(boxes=b.type(torch.int32),labels=torch.ones(1),keypoints=y)
                torch_image = torch.transpose(torch_image, dim0=-2, dim1=-1)
            return torch_image, target_dict
else:
    class PyTorchSatellitePoseEstimationDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')

if has_tf:
    class KerasDataGenerator(Sequence):

        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

        def __init__(self, preprocessor, label_list, speed_root, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True):

            # loading dataset
            self.image_root = os.path.join(speed_root, 'images', 'train')

            # Initialization
            self.preprocessor = preprocessor
            self.dim = dim
            self.batch_size = batch_size
            self.labels = self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                                         for label in label_list}
            self.list_IDs = [label['filename'] for label in label_list]
            self.n_channels = n_channels
            self.shuffle = shuffle
            self.indexes = None
            self.on_epoch_end()

        def __len__(self):

            """ Denotes the number of batches per epoch. """

            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):

            """ Generate one batch of data """

            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        def on_epoch_end(self):

            """ Updates indexes after each epoch """

            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):

            """ Generates data containing batch_size samples """

            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size, 7), dtype=float)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                img_path = os.path.join(self.image_root, ID)
                img = keras_image.load_img(img_path, target_size=(224, 224))
                x = keras_image.img_to_array(img)
                x = self.preprocessor(x)
                X[i,] = x

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                y[i] = np.concatenate([q, r])

            return X, y
else:
    class KerasDataGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError('tensorflow.keras is not available! Please install tensorflow.')

KEYPOINT_COLOR = (0, 255, 0)  # Green
def vis_keypoints(image, dic, color=KEYPOINT_COLOR, diameter=15):
    # image = image.copy()
    keypoints = dic["keypoints"]
    boxes = dic["boxes"]
    # print(boxes)
    image = np.uint8(image)
    image = Image.fromarray(image)
    imagedraw = ImageDraw.Draw(image)
    for k in keypoints:
        for x, y ,v in k:
            imagedraw.ellipse((x-10,y-10,x+10,y+10),(0,255,255))
            # imagedraw.point((x,y),(0,255,255))
    # for x1,y1,x2,y2 in boxes:
    #     # for x1,y1,x2,y2 in k:
    #     imagedraw.rectangle([x1,y1,x2,y2], fill=None, outline="white", width=1)
    image.show()
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
                1]])
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    T = [
        A.Blur(p=0.1),
        A.MedianBlur(p=0.1),
        # A.ToGray(p=1),
        A.CLAHE(p=0.1),
        A.RandomBrightnessContrast(p=0.1),
        A.RandomGamma(p=0.1),
        # A.ImageCompression(quality_lower=75, p=0.1),
        # A.OpticalDistortion(distort_limit=0.5, p=1),

        # A.CoarseDropout(max_holes=3, max_height=150, max_width=150, min_holes=2, min_height=50, min_width=50
        #                 , fill_value=255, mask_fill_value=100),
        # A.RandomSizedCrop(min_max_height=(800,1200),height=1200, width=1920,p=0.1),
        # A.RandomGridShuffle(grid=(50,50),p=0.5),
        # A.HorizontalFlip(p=1),
        # A.VerticalFlip(p=1),
        # A.RandomSunFlare(flare_roi=(0,0,1,1),src_radius=150,num_flare_circles_lower=1,num_flare_circles_upper=4,p=1),
        # A.Cutout(num_holes=2, max_h_size=300, max_w_size=300,fill_value=255,p=1),
        # A.Normalize(),
        A.Resize(height=1200, width=1920, always_apply=True),
        ToTensorV2(p=1.0)
    ]  # transforms
    trans = A.Compose(T, keypoint_params=A.KeypointParams(format='xy',
                                                          remove_invisible=False),
                      # bbox_params=A.BboxParams(format='pascal_voc')
                      )

    # trans()
    dataset_root_dir = '/opt/dl_workspace/algorithm/02-test/XuanRan_pose_estimate/datasets/speed/'  # path to speed



    import torch
    from torch.utils.data import DataLoader



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PyTorchSatellitePoseEstimationDataset(split='val', speed_root=dataset_root_dir, points=points, transform=trans)
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        # sampler=data_sampler(train_set, shuffle=True, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=1,
        # collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )


    # mean, std = get_mean_std(train_loader)
    # print(mean)
    # print(std)

    for _ in range(5):
        img, dic = train_set.__getitem__(_+1)

        torch_image = torch.transpose(img, dim0=-1, dim1=0)
        print(len(dic['keypoints'][0]))
        vis_keypoints(torch_image, dic)
        # print(y)
    x=0