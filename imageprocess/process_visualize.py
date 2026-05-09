"""Image processing: HistEq + CLAHE + Power-law on all 3 splits"""
import os, random, cv2, numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/imageprocess'

SOURCES = {
    'synthetic': '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/synthetic/images',
    'sunlamp': '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/sunlamp/images',
    'lightbox': '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/lightbox/images',
}

def hist_equalize(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def clahe_equalize(img, clip_limit=2.0, tile_size=(8,8)):
    """CLAHE - Contrast Limited Adaptive Histogram Equalization on V channel"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def power_law(img, gamma):
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, gamma)
    return (corrected * 255).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    random.seed(42)
    gamma_values = [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]
    n_per_split = 3
    total = 0

    for split_name, split_dir in SOURCES.items():
        all_images = sorted([f for f in os.listdir(split_dir) if f.endswith(('.jpg','.png'))])
        selected = random.sample(all_images, min(n_per_split, len(all_images)))

        for idx, fname in enumerate(selected):
            path = os.path.join(split_dir, fname)
            img = np.array(Image.open(path).convert('RGB'))
            he = hist_equalize(img)
            clahe1 = clahe_equalize(img, clip_limit=2.0, tile_size=(8,8))
            clahe2 = clahe_equalize(img, clip_limit=4.0, tile_size=(16,16))

            fig, axes = plt.subplots(3, 1, figsize=(16, 9))

            # Row 1: Histogram Equalization comparison
            axes[0].imshow(np.hstack([img, he, clahe1, clahe2]))
            axes[0].set_title(f'[{split_name}] {fname}  |  Original / HE / CLAHE(2.0,8x8) / CLAHE(4.0,16x16)', fontsize=8)
            axes[0].axis('off')

            # Row 2: Power-law
            gammas = [img] + [power_law(img, g) for g in gamma_values]
            axes[1].imshow(np.hstack(gammas))
            axes[1].set_title('Power-law Transform: Original + $\gamma$=0.3/0.5/0.7/1.5/2.0/3.0', fontsize=8)
            axes[1].axis('off')

            # Row 3: CLAHE with V channel histogram before/after
            axes[2].axis('off')
            # Show CLAHE on grayscale version for clarity
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_he = cv2.equalizeHist(gray)
            gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            axes[2].imshow(np.hstack([
                cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(gray_he, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB),
            ]))
            axes[2].set_title('Grayscale: Original / Global HE / CLAHE', fontsize=8)
            axes[2].axis('off')

            out_path = os.path.join(OUT_DIR, f'proc_{split_name}_{idx:02d}.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            total += 1
            print(f'[{total}/9] {out_path}')

    print(f'\nDone! {total} images saved to {OUT_DIR}/')
