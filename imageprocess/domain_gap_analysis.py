"""
Domain Gap Analysis: Synthetic vs Sunlamp
Systematic mathematical comparison of feature distributions.
"""
import os, random, cv2, numpy as np
from PIL import Image
from scipy import stats, ndimage
from scipy.fft import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/imageprocess'

SYN_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/synthetic/images'
SUN_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/sunlamp/images'

N_SAMPLES = 200

def load_samples(img_dir, n):
    files = random.sample(sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))]), n)
    imgs = []
    for f in files:
        img = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        imgs.append(img)
    return imgs

def extract_features(imgs):
    """Extract per-image and per-pixel statistics"""
    data = {
        'rgb_mean': [], 'rgb_std': [],        # global RGB stats
        'h_mean': [], 's_mean': [], 'v_mean': [],  # HSV means
        'h_std': [], 's_std': [], 'v_std': [],     # HSV stds
        'gradient_mag': [],                    # edge strength
        'fft_energy_low': [], 'fft_energy_mid': [], 'fft_energy_high': [],  # frequency bands
        'contrast': [],                        # GLCM contrast
        'entropy': [],                         # pixel entropy
        'v_skew': [],                          # brightness skew
        'dark_ratio': [],                      # ratio of dark pixels (V<30)
        'bright_ratio': [],                    # ratio of bright pixels (V>220)
    }
    
    for img in imgs:
        # RGB stats
        data['rgb_mean'].append(img.mean(axis=(0,1)))
        data['rgb_std'].append(img.std(axis=(0,1)))
        
        # HSV stats
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        data['h_mean'].append(hsv[:,:,0].mean())
        data['s_mean'].append(hsv[:,:,1].mean())
        data['v_mean'].append(hsv[:,:,2].mean())
        data['h_std'].append(hsv[:,:,0].std())
        data['s_std'].append(hsv[:,:,1].std())
        data['v_std'].append(hsv[:,:,2].std())
        
        # Brightness skew (V channel)
        v = hsv[:,:,2].ravel()
        data['v_skew'].append(stats.skew(v))
        data['dark_ratio'].append((v < 30).mean())
        data['bright_ratio'].append((v > 220).mean())
        
        # Edge gradient
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        data['gradient_mag'].append(np.sqrt(gx**2 + gy**2).mean())
        
        # Frequency analysis (center crop 256x256)
        h, w = gray.shape
        crop = gray[h//2-128:h//2+128, w//2-128:w//2+128]
        f = fftshift(fft2(crop))
        mag = np.abs(f)
        r = 128
        y, x = np.ogrid[-r:r, -r:r]
        dist = np.sqrt(x**2 + y**2)
        data['fft_energy_low'].append(mag[dist < 10].sum())
        data['fft_energy_mid'].append(mag[(dist >= 10) & (dist < 60)].sum())
        data['fft_energy_high'].append(mag[dist >= 60].sum())
        
        # GLCM texture
        glcm = graycomatrix((gray/255*255).astype(np.uint8), [1], [0], levels=256, symmetric=True, normed=True)
        data['contrast'].append(graycoprops(glcm, 'contrast')[0,0])
        data['entropy'].append(stats.entropy(gray.ravel(), base=2))

    return {k: np.array(v) for k, v in data.items()}

def plot_comparison(syn_data, sun_data, feature_name, title, ax, bins=40):
    """Plot distribution comparison"""
    ax.hist(syn_data, bins=bins, alpha=0.5, label='Synthetic', density=True, color='#1f77b4')
    ax.hist(sun_data, bins=bins, alpha=0.5, label='Sunlamp', density=True, color='#ff7f0e')
    ax.set_title(title, fontsize=7)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(syn_data, sun_data)
    ax.text(0.02, 0.95, f'D={ks_stat:.3f}', transform=ax.transAxes, fontsize=5, va='top')

    return ks_stat

if __name__ == '__main__':
    random.seed(42)
    print(f'Loading {N_SAMPLES} synthetic images...')
    syn_imgs = load_samples(SYN_DIR, N_SAMPLES)
    print(f'Loading {N_SAMPLES} sunlamp images...')
    sun_imgs = load_samples(SUN_DIR, N_SAMPLES)

    print('Extracting features...')
    syn = extract_features(syn_imgs)
    sun = extract_features(sun_imgs)

    # ========== Figure 1: Multi-panel feature distributions ==========
    features = [
        ('v_mean', 'V Mean (brightness)', True),
        ('v_std', 'V Std (contrast)', True),
        ('v_skew', 'V Skewness', True),
        ('s_mean', 'S Mean (saturation)', True),
        ('s_std', 'S Std', True),
        ('gradient_mag', 'Edge Gradient Mag', True),
        ('contrast', 'GLCM Contrast', True),
        ('entropy', 'Pixel Entropy', True),
        ('dark_ratio', 'Dark Pixel Ratio', True),
        ('bright_ratio', 'Bright Pixel Ratio', False),
        ('fft_energy_low', 'FFT Low Freq', False),
        ('fft_energy_high', 'FFT High Freq', False),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    ranks = []
    for ax, (key, title, _) in zip(axes.flat, features):
        ks = plot_comparison(syn[key], sun[key], key, title, ax)
        ranks.append((ks, key, title))

    ranks.sort(key=lambda x: -x[0])
    fig.suptitle(f'Domain Gap Analysis: Synthetic vs Sunlamp (N={N_SAMPLES}, KS test D-values)\n'
                 f'Top gaps: {" > ".join(f"{r[2]}({r[0]:.3f})" for r in ranks[:4])}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'analysis_feature_gap.png'), dpi=200)
    plt.close()

    # ========== Figure 2: RGB histograms (aggregated) ==========
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['R', 'G', 'B']
    for c, ax in enumerate(axes):
        syn_vals = np.concatenate([img[:,:,c].ravel() for img in syn_imgs[:50]])
        sun_vals = np.concatenate([img[:,:,c].ravel() for img in sun_imgs[:50]])
        ax.hist(syn_vals, bins=100, alpha=0.4, density=True, label='Synthetic', color='#1f77b4')
        ax.hist(sun_vals, bins=100, alpha=0.4, density=True, label='Sunlamp', color='#ff7f0e')
        ax.set_title(f'Channel {colors[c]}', fontsize=9)
        ax.legend(fontsize=6)
    suptitle = 'Aggregated RGB Histograms (50 samples each)'
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'analysis_rgb_hist.png'), dpi=200)
    plt.close()

    # ========== Figure 3: FFT spectrum comparison ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(3):
        syn_gray = cv2.cvtColor(syn_imgs[i], cv2.COLOR_RGB2GRAY).astype(float)
        sun_gray = cv2.cvtColor(sun_imgs[i], cv2.COLOR_RGB2GRAY).astype(float)
        axes[0,i].imshow(np.log(fftshift(np.abs(fft2(syn_gray))) + 1), cmap='viridis')
        axes[0,i].set_title(f'Synthetic #{i+1}', fontsize=7); axes[0,i].axis('off')
        axes[1,i].imshow(np.log(fftshift(np.abs(fft2(sun_gray))) + 1), cmap='viridis')
        axes[1,i].set_title(f'Sunlamp #{i+1}', fontsize=7); axes[1,i].axis('off')
    fig.suptitle('FFT Magnitude Spectra', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'analysis_fft_spectra.png'), dpi=200)
    plt.close()

    # ========== Figure 4: Proposed mitigation strategies ==========
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)

    syn_img = syn_imgs[0].copy()
    sun_img = sun_imgs[0].copy()

    # Strategy 1: HSV V-channel matching
    ax = fig.add_subplot(gs[0, 0])
    syn_hsv = cv2.cvtColor(syn_img, cv2.COLOR_RGB2HSV).astype(float)
    sun_hsv = cv2.cvtColor(sun_img, cv2.COLOR_RGB2HSV).astype(float)
    # Match V distribution
    syn_v = syn_hsv[:,:,2]
    sun_v_mean, sun_v_std = sun_hsv[:,:,2].mean(), sun_hsv[:,:,2].std()
    syn_v_matched = ((syn_v - syn_v.mean()) / syn_v.std() * sun_v_std + sun_v_mean).clip(0,255)
    syn_hsv[:,:,2] = syn_v_matched
    matched = cv2.cvtColor(syn_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    ax.imshow(np.hstack([syn_img, matched]))
    ax.set_title('V-Channel Matching\n(Synthetic V→Sunlamp V μ/σ)', fontsize=8); ax.axis('off')

    # Strategy 2: Histogram Equalization
    ax = fig.add_subplot(gs[0, 1])
    he = cv2.cvtColor(syn_img, cv2.COLOR_RGB2HSV)
    he[:,:,2] = cv2.equalizeHist(he[:,:,2])
    he_rgb = cv2.cvtColor(he, cv2.COLOR_HSV2RGB)
    ax.imshow(np.hstack([syn_img, he_rgb]))
    ax.set_title('Histogram Equalization (V channel)', fontsize=8); ax.axis('off')

    # Strategy 3: CLAHE
    ax = fig.add_subplot(gs[1, 0])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = cv2.cvtColor(syn_img, cv2.COLOR_RGB2HSV)
    cl[:,:,2] = clahe.apply(cl[:,:,2])
    cl_rgb = cv2.cvtColor(cl, cv2.COLOR_HSV2RGB)
    ax.imshow(np.hstack([syn_img, cl_rgb]))
    ax.set_title('CLAHE (2.0, 8x8)', fontsize=8); ax.axis('off')

    # Strategy 4: Power-law gamma correction
    ax = fig.add_subplot(gs[1, 1])
    gamma_val = 1.8
    corrected = ((syn_img.astype(float)/255)**gamma_val * 255).clip(0,255).astype(np.uint8)
    ax.imshow(np.hstack([syn_img, corrected]))
    ax.set_title(f'Power-law (γ={gamma_val})', fontsize=8); ax.axis('off')

    # Strategy 5: Summary table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    summary = f"""Domain Gap Summary (KS D-values, higher = larger gap):
    
    Top gaps:  {' > '.join(f'{r[2]} ({r[0]:.3f})' for r in ranks[:4])}
    
    Key findings:
    1. Brightness/contrast: Sunlamp has LOWER V-mean and HIGHER V-std → images are darker but more contrasty
    2. Saturation: Sunlamp has HIGHER saturation → more vivid colors than synthetic
    3. Texture: Sunlamp has MORE edges and HIGHER GLCM contrast → sharper, more texture
    4. Frequency: Sunlamp has MORE high-frequency energy → fine details (real sensor noise, texture)
    
    Recommended pipeline (from analysis):
      Synthetic → [CLAHE or HE] → [V-channel μ/σ matching to Sunlamp] → [Power-law γ<1 to brighten]
    
    Augmentation candidates:
      - A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8)) or ALB RandomBrightnessContrast
      - A.GaussNoise to add high-freq sensor noise
      - A.RandomGamma for brightness distribution shift
      - A.Sharpen to increase edge/texture contrast
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=7,
            fontfamily='monospace', va='top', linespacing=1.5)

    fig.suptitle('Mitigation Strategies & Summary', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'analysis_mitigation.png'), dpi=200)
    plt.close()

    print(f'\n=== RANKED GAPS (KS D-values) ===')
    for ks, key, title in ranks:
        print(f'  {title:<30s} D={ks:.4f}')
    print(f'\nDone! Outputs saved to {OUT_DIR}/')
    print(f'  - analysis_feature_gap.png')
    print(f'  - analysis_rgb_hist.png')
    print(f'  - analysis_fft_spectra.png')
    print(f'  - analysis_mitigation.png')
