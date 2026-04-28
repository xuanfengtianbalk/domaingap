import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image

class PerspectiveWarpWithROI(object):
    def __init__(self, angle_range=(-45, 45), roi=None):
        self.angle_range = angle_range
        self.roi = roi  # (x, y, w, h)

    def __call__(self, img, keypoints):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        img_np = np.array(img)

        M = None
        if self.roi:
            x, y, w, h = self.roi
            roi_img = img_np[y:y + h, x:x + w]
            warped_roi_img, M = self.warp_perspective(roi_img, angle)
            img_np[y:y + h, x:x + w] = warped_roi_img
        else:
            img_np, M = self.warp_perspective(img_np, angle)

        # Update keypoints
        if M is not None:
            keypoints = self.transform_keypoints(keypoints, M)

        return F.to_pil_image(img_np), keypoints

    def warp_perspective(self, img, angle):
        h, w = img.shape

        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst_pts = np.float32([
            [w * (-angle / 90), 0],
            [w * (1 + angle / 90), 0],
            [0, h],
            [w, h]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(img, M, (w, h))
        return warped_img, M

    def transform_keypoints(self, keypoints, M):
        keypoints = np.array([(kp[0], kp[1], 1) for kp in keypoints])
        keypoints = keypoints.dot(M.T)
        keypoints = keypoints[:, :2] / keypoints[:, 2, np.newaxis]
        return keypoints.tolist()


# # 示例使用
# from torchvision.transforms import Compose, ToTensor
# from PIL import Image
# # x0,y0,x1,y1=[100,300,1750,1200]
# roi = (100, 300, 1600, 1200)  # (x, y, width, height)
# transform = PerspectiveWarpWithROI(angle_range=(-30, 30), roi=roi)
#
#
#
# image_path = 'test.jpg'
# img = Image.open(image_path)
#
# keypoints = [(700, 700), (800, 900), (1100, 1100)]
# new_img, new_keypoints = transform(img, keypoints)
#
# print("Original Keypoints:", keypoints)
# print("Transformed Keypoints:", new_keypoints)
#
#
# # Plotting
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(img)
# axes[0].scatter(*zip(*keypoints), c='r', marker='o')
# axes[0].set_title("Original Image with Keypoints")
#
# axes[1].imshow(new_img)
# axes[1].scatter(*zip(*new_keypoints), c='r', marker='o')
# axes[1].set_title("Warped Image with Transformed Keypoints")
#
# plt.tight_layout()
# plt.savefig("comparison.png", dpi=300)
# plt.close(fig)