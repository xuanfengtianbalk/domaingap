"""Visualize SpaceAugTransform: aug1-4, styleaug, augmix"""
import os, random, sys, numpy as np
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform

DATASET_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus'
OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/aug_test'

if __name__ == '__main__':
    random.seed(42)
    all_images = sorted([f for f in os.listdir(DATASET_DIR+'/synthetic/images') if f.endswith(('.jpg','.png'))])
    selected = random.sample(all_images, min(10, len(all_images)))

    aug_types = ['aug1', 'aug2', 'aug3', 'aug4', 'styleaug', 'augmix']
    n_cols = 8  # orig + 6 augs + sunlamp target

    for idx, fname in enumerate(selected):
        path = os.path.join(DATASET_DIR+'/synthetic/images', fname)
        img = np.array(Image.open(path).convert('RGB'))

        fig, axes = plt.subplots(1, n_cols, figsize=(24, 3))
        axes[0].imshow(img); axes[0].set_title('Original', fontsize=6); axes[0].axis('off')

        for j, at in enumerate(aug_types):
            trans = SpaceAugTransform(at, styleaug_p=0.5, normalize=False, to_gray=True)  # p=1 for vis
            result = trans(image=img)
            axes[j+1].imshow(result['image'])
            axes[j+1].set_title(at, fontsize=6); axes[j+1].axis('off')

        # Sunlamp target
        sun_dir = os.path.join(DATASET_DIR, 'sunlamp', 'images')
        sun_f = random.choice(sorted(os.listdir(sun_dir)))
        sun_img = np.array(Image.open(os.path.join(sun_dir, sun_f)).convert('RGB').resize((480, 300)))
        axes[-1].imshow(sun_img); axes[-1].set_title('Sunlamp', fontsize=6); axes[-1].axis('off')

        out_path = os.path.join(OUT_DIR, f'aug_sample_{idx:02d}.png')
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f'[{idx+1}/10] {out_path}')

    print(f'\nDone! {OUT_DIR}/')
