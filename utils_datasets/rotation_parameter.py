import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import math
def quaternion2euler(quat):
    Rmat = R.from_quat(quat)
    euler = Rmat.as_euler('xyz', degrees=True)
    return euler

def euler2quaternion(euler):
    Rmat = R.from_euler('xyz',euler,degrees=True)
    quaternion = Rmat.as_quat()
    return quaternion

def rotvector2rot(rotvector):
    Rmat = cv2.Rodrigues(rotvector)[0]
    return Rmat

def rot2rotvector(Rmat):
    rotvector = cv2.Rodrigues(Rmat)[0]
    return rotvector

def quaternion2rotvector(quats):
    Rmats = R.from_quat(quats) #qw, qx, qy, qz
    Rmats = Rmats.as_matrix()
    rotvectors = []
    for Rmat in Rmats:
        rotvector = cv2.Rodrigues(Rmat)[0]
        rotvectors.append(torch.tensor(rotvector).squeeze())
    rotvectors = torch.stack(rotvectors)
    return rotvectors

def quaternion2rot(quat):
    Rmat = R.from_quat(quat)
    return Rmat

def cross(vector1, vector2):
    vector1_x = vector1[:, :, 0]
    vector1_y = vector1[:, :, 1]
    vector1_z = vector1[:, :, 2]
    vector2_x = vector2[:, :, 0]
    vector2_y = vector2[:, :, 1]
    vector2_z = vector2[:, :, 2]
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return torch.stack((n_x, n_y, n_z))

def dot(vector1, vector2):
    return torch.dot(vector1,vector2)

def rotate(point, axis, angle):
    cos_angle = math.cos(angle)
    axis_dot_point = dot(axis, point)
    return point * cos_angle + cross(
        axis, point) * math.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)


def points_trans_accord(Rmat, t):
    #Rmat 3*3, t 3*n
    t1 = torch.mm(torch.tensor(Rmat.as_matrix()), t)
    return t1

if __name__ == "__main__":
    quat = torch.rand((12,4))
    Rmats=quaternion2rot(quat)
    # rot = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    # print(rot.as_matrix())
    t = torch.tensor([[1,-1,1], [-1,1,-1]],dtype=float).view(3,-1)
    for Rmat in Rmats:
        print(points_trans_accord(Rmat, t))