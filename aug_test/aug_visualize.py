"""Visualize training augmentations via PyTorchSatellitePoseEstimationDataset pipeline.
All augmentation columns come directly from the dataset, matching __getitem__ exactly."""
import os, random, sys, numpy as np
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')
import torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
from utils_datasets.speedplus_utils_main.utils import (
    PyTorchSatellitePoseEstimationDataset, points as body_points)
from Hyperpose_net.losses.consistency import ConsistencyAugmentor

DATASET_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus'
OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/aug_test'
AUG_TYPES = ['augbaseline', 'aug3', 'aug4', 'styleaug', 'augmix']

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def denorm(tensor):
    """Reverse ImageNet normalization: float32 [C,H,W] ~[-2,2] → uint8 [H,W,C]."""
    arr = tensor.numpy().astype(np.float32) * STD + MEAN
    return (arr.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


if __name__ == '__main__':
    random.seed(42)

    # base dataset: crop + resize, no augmentation, no styleaug
    ds_none = PyTorchSatellitePoseEstimationDataset(
        split='train', speed_root=DATASET_DIR, points=body_points,
        transform=SpaceAugTransform('none', styleaug_p=0))

    # one dataset per augmentation type; identical __getitem__ path
    aug_datasets = {}
    for at in AUG_TYPES:
        sp = 1.0 if at == 'styleaug' else 0.0
        aug_datasets[at] = PyTorchSatellitePoseEstimationDataset(
            split='train', speed_root=DATASET_DIR, points=body_points,
            transform=SpaceAugTransform(at, styleaug_p=sp))

    augmentor = ConsistencyAugmentor([
        {'type': 'randconv', 'mix': True, 'p': 0.0},
        # {'type': 'aug', 'aug_type': 'augmix'}
    ])
    augmentor.train()  # p=0 → always apply RandConv

    n_cols = 8  # crop + 7 augs + randconv + sunlamp
    n_total = len(ds_none)
    indices = random.sample(range(n_total), min(10, n_total))

    for i, idx in enumerate(indices):
        # 1) raw crop from ds_none → denorm for display
        sample_none, _ = ds_none[idx]
        resized = denorm(sample_none)

        fig, axes = plt.subplots(1, n_cols, figsize=(31, 3))
        axes[0].imshow(resized)
        axes[0].set_title('Crop 256×256', fontsize=6); axes[0].axis('off')

        # 2) each augmentation column from its own dataset
        for j, at in enumerate(AUG_TYPES):
            sample_aug, _ = aug_datasets[at][idx]
            axes[j + 1].imshow(denorm(sample_aug))
            axes[j + 1].set_title(at, fontsize=6); axes[j + 1].axis('off')

        # 3) AugConsistencyLayer: full pipeline matching experiment A, denorm for display
        rc_out = augmentor(sample_none.unsqueeze(0))[0].squeeze(0)
        axes[-2].imshow(denorm(rc_out))
        axes[-2].set_title('AugConsistency', fontsize=6); axes[-2].axis('off')

        # 4) Sunlamp reference
        sun_dir = os.path.join(DATASET_DIR, 'sunlamp', 'images')
        sun_f = random.choice(sorted(os.listdir(sun_dir)))
        sun_img = plt.imread(os.path.join(sun_dir, sun_f))
        axes[-1].imshow(sun_img)
        axes[-1].set_title('Sunlamp', fontsize=6); axes[-1].axis('off')

        out_path = os.path.join(OUT_DIR, f'aug_sample_{i:02d}.png')
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f'[{i + 1}/{len(indices)}] {out_path}')

    print(f'\nDone! {OUT_DIR}/')
