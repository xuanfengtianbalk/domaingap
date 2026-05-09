import torch
import numpy as np
import torch.nn.functional as F
def crop_tensor_image(image, bbox, padded=False):
    xmin=bbox[0]
    ymin=bbox[1]
    xmax=bbox[2]
    ymax=bbox[3]

    if not padded:
        # clip bbox to image boundaries, no padding
        xmin = int(torch.clamp(xmin, 0, image.shape[1]))
        ymin = int(torch.clamp(ymin, 0, image.shape[2]))
        xmax = int(torch.clamp(xmax, 0, image.shape[1]))
        ymax = int(torch.clamp(ymax, 0, image.shape[2]))
    else:
        # pad if bbox exceeds image
        pad_left = max(0, -xmin)
        pad_top = max(0, -ymin)
        pad_right = max(0, xmax - image.shape[1])
        pad_bottom = max(0, ymax - image.shape[2])
        if pad_left or pad_top or pad_right or pad_bottom:
            padding = [int(_) for _ in [pad_top, pad_bottom, pad_left, pad_right]]
            image = F.pad(image, padding, value=0)
        xmin = int(torch.clamp(xmin, 0, image.shape[1] - 1))
        ymin = int(torch.clamp(ymin, 0, image.shape[2] - 1))
        if pad_top:
            ymax = int(torch.clamp(ymax + pad_top, 0, image.shape[2]))
        else:
            ymax = int(torch.clamp(ymax, 0, image.shape[2]))
        if pad_right:
            xmax = int(torch.clamp(xmax + pad_right, 0, image.shape[1]))
        else:
            xmax = int(torch.clamp(xmax, 0, image.shape[1]))

    croped_image=image[:,xmin:xmax,ymin:ymax]
    return croped_image
def apply_random_cutout(img, min_ratio=0.0, max_ratio=0.3, p=0.5):
    """
    Apply black cutout on a random edge of the image tensor.

    :param img: Input image tensor [C, H, W]
    :param min_ratio: Minimum ratio of the image size to cutout
    :param max_ratio: Maximum ratio of the image size to cutout
    :return: Augmented image tensor
    """
    # Clone the input image tensor
    img_copy = img.clone()
    c, h, w = img_copy.size()
    if np.random.rand() < p:
        # Determine the edge and the size of the cutout
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        ratio = np.random.uniform(min_ratio, max_ratio)

        if edge == 'top':
            y1 = 0
            y2 = int(ratio * h)
            x1 = 0
            x2 = w
        elif edge == 'bottom':
            y1 = h - int(ratio * h)
            y2 = h
            x1 = 0
            x2 = w
        elif edge == 'left':
            y1 = 0
            y2 = h
            x1 = 0
            x2 = int(ratio * w)
        else:
            y1 = 0
            y2 = h
            x1 = w - int(ratio * w)
            x2 = w

        # Apply the cutout
        img_copy[:, y1:y2, x1:x2] = 0

    return img_copy

def apply_cutout(img, num_cutouts=1, min_ratio=0.1, max_ratio=0.2, p=0.5):
    """
    Apply cutout augmentation on the image tensor with a certain probability.

    :param img: Input image tensor [C, H, W]
    :param num_cutouts: Number of cutouts to apply
    :param min_ratio: Minimum ratio of the cutout size relative to the image size
    :param max_ratio: Maximum ratio of the cutout size relative to the image size
    :param p: Probability of applying cutout
    :return: Augmented image tensor
    """
    # Clone the input image tensor
    img_copy = img.clone()
    _, h, w = img_copy.size()

    # Apply cutout with probability p
    if np.random.rand() < p:
        # Generate random coordinates for the cutout
        for _ in range(num_cutouts):
            # Determine the size of the cutout
            ratio = np.random.uniform(min_ratio, max_ratio)
            size = int(ratio * min(h, w))

            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()

            # Determine the coordinates of the cutout
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)

            # Apply cutout
            img_copy[:, y1:y2, x1:x2] = 0

    return img_copy

def apply_black_border(img, min_ratio=0.8, max_ratio=1.0, p=0.5):
    """
    Apply black border on the image tensor with a certain probability.

    :param img: Input image tensor [C, H, W]
    :param min_ratio: Minimum ratio of the image size to keep
    :param max_ratio: Maximum ratio of the image size to keep
    :param p: Probability of applying black border
    :return: Augmented image tensor
    """
    # Apply black border with probability p
    if np.random.rand() < p:
        # Clone the input image tensor
        img_copy = img.clone()
        c, h, w = img_copy.size()

        # Determine the size of the image to keep
        ratio = np.random.uniform(min_ratio, max_ratio)
        new_h = int(ratio * h)
        new_w = int(ratio * w)

        # Create a black image
        black_img = torch.zeros((c, h, w))

        # Determine the coordinates to place the original image
        y1 = (h - new_h) // 2
        y2 = y1 + new_h
        x1 = (w - new_w) // 2
        x2 = x1 + new_w

        # Place the original image on the black image
        black_img[:, y1:y2, x1:x2] = img[:, y1:y2, x1:x2]

        return black_img
    else:
        return img


# # Example usage:
# # Assume 'img' is a PyTorch tensor of shape [3, 128, 128]
# # Apply cutout with probability 0.5
# augmented_img = apply_cutout(img, num_cutouts=2, min_ratio=0.1, max_ratio=0.2, p=0.5)
#
# # To display or save the image, you can convert the tensor to a PIL image
# from torchvision.transforms import ToPILImage
#
# to_pil = ToPILImage()
# pil_img = to_pil(augmented_img)
# pil_img.show()
