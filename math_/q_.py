import torch
import numpy as np
# 四元数乘法
def quatProduct(q1, q2):
    q= torch.tensor([q1[0]*q2[0] - q1[1]*q2[1] -q1[2]*q2[2]-q1[3]*q2[3],
     q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
     q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
     q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
    return q

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Return the conjugate of the quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return rotation_matrix

def rotate_vector_by_quaternion(v, q):
    """Rotate a vector by a quaternion."""
    q_conjugate = quaternion_conjugate(q)
    v_quaternion = np.concatenate((np.array([0]), v))
    rotated_v_quaternion = quaternion_multiply(quaternion_multiply(q, v_quaternion), q_conjugate)
    return rotated_v_quaternion[1:]

def quaternion_between(q1, q2):
    """Calculate the quaternion that rotates from q1 to q2."""
    q = quaternion_multiply(quaternion_conjugate(q1), q2)
    if np.array_equal(q, np.array([1, 0, 0, 0])):
        return np.array([1, 0, 0, 0])
    if q[0]<0:
        q=-q
    return q

def q2_given_q1_and_quaternion(q1, q_diff):
    """Calculate q2 given q1 and the quaternion representing the rotation from q1 to q2."""
    q2 = quaternion_multiply(q_diff, q1)
    return q2