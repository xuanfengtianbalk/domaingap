"""Visualize aug1-aug5 from DLR paper"""
import os, random, sys
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/synthetic/images'
OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/aug_test'

def build_aug1():
    return A.Compose([
        A.OneOf([A.GaussianBlur((3,7), p=1), A.MotionBlur((3,7), p=1)], p=1),
        A.Sharpen(p=0.5), A.Emboss(p=0.5),
        A.GaussNoise((0.01,0.05), p=0.5), A.CoarseDropout(num_holes_range=(1,8), p=0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
    ], p=1)

def build_aug2():
    return A.Compose([
        A.OneOf([A.GaussianBlur((3,7),p=0.7), A.MotionBlur((3,7),p=0.7)], p=0.7),
        A.Sharpen(p=0.5), A.Emboss(p=0.5),
        A.GaussNoise((0.01,0.05), p=0.5), A.CoarseDropout(num_holes_range=(1,8), p=0.5),
        A.RandomBrightnessContrast(0.3,0.3,p=0.5),
        A.Superpixels(p_replace=0.1, n_segments=100, p=0.5),
        A.PixelDropout(0.02, p=0.5),
    ], p=1)

def build_aug3():
    return A.Compose([
        A.OneOf([A.GaussianBlur((3,7),p=0.5), A.MotionBlur((3,7),p=0.5)], p=0.5),
        A.Sharpen(p=0.3), A.Emboss(p=0.3),
        A.GaussNoise((0.01,0.05), p=0.3), A.CoarseDropout(num_holes_range=(1,8), p=0.3),
        A.RandomBrightnessContrast(0.3,0.3,p=0.3),
        A.Superpixels(p_replace=0.1, n_segments=100, p=0.3),
        A.PixelDropout(0.02, p=0.3),
        A.RandomFog(0.2, p=0.5), A.RandomSnow(0.2, p=0.5),
        A.RandomSunFlare((0,0,1,0.5), src_radius=200, p=0.5),
        A.RandomBrightnessContrast(0.5,0.5,p=0.5),
    ], p=1)

def build_aug4():
    return A.Compose([
        A.OneOf([A.GaussianBlur((3,7),p=0.5), A.MotionBlur((3,7),p=0.5)], p=0.5),
        A.Sharpen(p=0.3), A.Emboss(p=0.3),
        A.GaussNoise((0.01,0.05), p=0.3), A.CoarseDropout(num_holes_range=(1,8), p=0.3),
        A.RandomBrightnessContrast(0.3,0.3,p=0.3),
        A.Superpixels(p_replace=0.1, n_segments=100, p=0.3),
        A.PixelDropout(0.02, p=0.3),
        A.RandomFog(0.2, p=0.5), A.RandomSnow(0.2, p=0.5),
        A.RandomSunFlare((0,0,1,0.5), src_radius=200, p=0.5),
        A.RandomBrightnessContrast(0.5,0.5,p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ], p=1)

def build_aug5():
    brightness = A.Compose([
        A.OneOf([A.RandomBrightnessContrast(0.2,0.2,p=1), A.ColorJitter(p=1)], p=1),
    ])
    blur = A.Compose([
        A.OneOf([A.GaussianBlur((3,7),p=1), A.MotionBlur((3,7),p=1), A.Sharpen(p=1)], p=1),
    ])
    corruptions = A.Compose([
        A.SomeOf([A.GaussNoise((0.01,0.05),p=1), A.RandomFog(0.2,p=1),
                  A.RandomSnow(0.2,p=1), A.RandomSunFlare((0,0,1,0.5),src_radius=200,p=1),
                  A.PixelDropout(0.02,p=1)], n=2),
    ])
    general = A.Compose([
        A.SomeOf([A.CoarseDropout(num_holes_range=(1,8),p=1), A.Emboss(p=1),
                  A.Superpixels(p_replace=0.1,n_segments=100,p=1),
                  A.HueSaturationValue(20,30,20,p=1)], n=2),
    ])
    return A.Compose([brightness, blur, corruptions, general], p=1)

if __name__ == '__main__':
    all_images = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(('.jpg','.png'))])
    selected = random.sample(all_images, min(10, len(all_images)))
    random.seed(42)

    augs = {'aug1': build_aug1(), 'aug2': build_aug2(), 'aug3': build_aug3(),
            'aug4': build_aug4(), 'aug5': build_aug5()}

    for idx, fname in enumerate(selected):
        path = os.path.join(DATASET_DIR, fname)
        img = np.array(Image.open(path).convert('RGB'))

        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=8)
        axes[0].axis('off')

        for j, (name, aug) in enumerate(augs.items()):
            result = aug(image=img)['image']
            axes[j+1].imshow(result)
            axes[j+1].set_title(name, fontsize=8)
            axes[j+1].axis('off')

        out_path = os.path.join(OUT_DIR, f'aug_sample_{idx:02d}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'[{idx+1}/10] {fname} -> {out_path}')

    print(f'\nDone! Output: {OUT_DIR}/')
