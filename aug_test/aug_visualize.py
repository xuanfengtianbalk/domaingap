"""Visualize SpaceAugTransform on cropped+resized 256x256 images (matching training pipeline)"""
import os, random, sys, json, numpy as np
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')
from PIL import Image
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
from utils_datasets.speedplus_utils_main.utils import (
    PyTorchSatellitePoseEstimationDataset, points as body_points, convex_hull
)

DATASET_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus'
OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/aug_test'

if __name__ == '__main__':
    random.seed(42)

    with open(os.path.join(DATASET_DIR, 'synthetic', 'train.json'), 'r') as f:
        labels = json.load(f)
    label_dict = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                  for label in labels}

    all_images = sorted([f for f in os.listdir(DATASET_DIR+'/synthetic/images') if f.endswith(('.jpg','.png')) and f in label_dict])
    selected = random.sample(all_images, min(10, len(all_images)))

    aug_types = ['augbaseline', 'aug1', 'aug2', 'aug3', 'aug4', 'styleaug', 'augmix']
    n_cols = 9  # crop + 7 augs + sunlamp

    ds = PyTorchSatellitePoseEstimationDataset(
        split='train', speed_root=DATASET_DIR, points=body_points)

    for idx, fname in enumerate(selected):
        path = os.path.join(DATASET_DIR+'/synthetic/images', fname)
        full_img = np.array(Image.open(path).convert('RGB'))
        H, W, _ = full_img.shape

        label = label_dict[fname]
        q, r = label['q'], label['r']
        kp, distance = ds.calculate_landmarks_distance(p_axes=body_points, q=q, r=r)

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

        kp_orig = []
        for x, y, view in kp:
            kp_orig.append([x, y, int(view)])
        bbox, _ = ds.calculate_boxes_and_padded(torch.tensor(kp_orig, dtype=torch.float32))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        cropped = full_img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)

        fig, axes = plt.subplots(1, n_cols, figsize=(28, 3))
        axes[0].imshow(resized); axes[0].set_title(f'Crop 256x256\n[{x1},{y1},{x2},{y2}]', fontsize=6); axes[0].axis('off')

        for j, at in enumerate(aug_types):
            trans = SpaceAugTransform(at, styleaug_p=1.0, normalize=False, to_gray=True)
            result = trans(image=resized.copy())
            axes[j+1].imshow(result['image'])
            axes[j+1].set_title(at, fontsize=6); axes[j+1].axis('off')

        sun_dir = os.path.join(DATASET_DIR, 'sunlamp', 'images')
        sun_f = random.choice(sorted(os.listdir(sun_dir)))
        sun_img = np.array(Image.open(os.path.join(sun_dir, sun_f)).convert('RGB').resize((480, 300)))
        axes[-1].imshow(sun_img); axes[-1].set_title('Sunlamp', fontsize=6); axes[-1].axis('off')

        out_path = os.path.join(OUT_DIR, f'aug_sample_{idx:02d}.png')
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f'[{idx+1}/10] {out_path}')

    print(f'\nDone! {OUT_DIR}/')
