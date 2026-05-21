"""PCA at each stage: encoder→reassemble→fusion→decoder"""
import sys, os, random, json, argparse, numpy as np
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')

import torch
import cv2
import albumentations as A
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run import load_config, build_model
from utils_datasets.speedplus_utils_main.utils import PyTorchSatellitePoseEstimationDataset, points as body_points, convex_hull
from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform

DATASET_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus'
DISTINCT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                   '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']

def get_crop(full_img, kp, ds):
    kp_orig = [[x, y, int(v)] for x, y, v in kp]
    bbox, _ = ds.calculate_boxes_and_padded(torch.tensor(kp_orig, dtype=torch.float32))
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return cv2.resize(full_img[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LINEAR)


def pca_plot(features_list, colors, stage_names, model_type, aug_type, m, out_path):
    n_stages = len(stage_names)
    cols = min(n_stages, 4)
    rows = (n_stages + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for s, (feat, name) in enumerate(zip(features_list, stage_names)):
        if feat is None:
            axes[s].text(0.5, 0.5, 'No features', ha='center', va='center')
            axes[s].set_title(name)
            continue
        feat_np = feat.cpu().numpy()
        feat_scaled = StandardScaler().fit_transform(feat_np)
        feat_pca = PCA(n_components=2).fit_transform(feat_scaled)
        for i in range(m):
            mask = np.array(colors) == i
            axes[s].scatter(feat_pca[mask, 0], feat_pca[mask, 1],
                            c=[DISTINCT_COLORS[i % len(DISTINCT_COLORS)]], s=15, alpha=0.6,
                            label=f'Img {i+1}')
        axes[s].set_title(name, fontsize=9)
        axes[s].set_xticks([]); axes[s].set_yticks([])
    for j in range(n_stages, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=m, fontsize=7)
    fig.suptitle(f'{model_type} | {aug_type} | m={m}', fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150); plt.close()
    print(f'Done: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uuid', type=str, required=True)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--aug_type', type=str, default='aug4')
    parser.add_argument('--out_name', type=str, default='',
                        help='Custom output filename (overrides auto-generated)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda:0')

    uuid_dir = os.path.join('workingdir', args.model_uuid)
    info = json.load(open(os.path.join(uuid_dir, 'train_info.json')))
    model_type = info['model_info']['TYPE']
    backbone_name = info['model_info']['BACKBONE_NAME']
    checkpoint = os.path.join(uuid_dir, 'model_final.pth')

    config = load_config('configs/cfg.yaml')
    config['MODEL']['TYPE'] = model_type
    config['MODEL']['BACKBONE_NAME'] = backbone_name
    model, bc = build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device).eval()

    with open(os.path.join(DATASET_DIR, 'synthetic', 'validation.json'), 'r') as f:
        labels = json.load(f)
    label_dict = {l['filename']: {'q': l['q_vbs2tango_true'], 'r': l['r_Vo2To_vbs_true']} for l in labels}
    all_images = [f for f in os.listdir(DATASET_DIR+'/synthetic/images')
                  if f.endswith(('.jpg','.png')) and f in label_dict]
    random.shuffle(all_images)

    ds = PyTorchSatellitePoseEstimationDataset(split='validation', speed_root=DATASET_DIR, points=body_points)
    trans = SpaceAugTransform(args.aug_type, styleaug_p=0.0, to_gray=True)
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    selected = all_images[:args.m]

    features_per_stage = {name: [] for name in ['enc_0', 'enc_1', 'enc_2', 'enc_3',
                                                  'reassemble', 'fusion', 'decoder']}
    colors = []
    collected = {name: [] for name in features_per_stage}

    def make_encoder_hook(layer_idx):
        def hook(module, input, output):
            _, cls = output[layer_idx]
            collected[f'enc_{layer_idx}'].append(cls.detach().cpu())
        return hook

    def make_decoder_hook():
        def hook(module, input, output):
            collected['decoder'].append(output.mean(dim=[-2, -1]).detach().cpu())
        return hook

    handles = []
    for li in range(4):
        handles.append(model.encoder.register_forward_hook(make_encoder_hook(li)))

    dec_key = next(iter(model.decoder))
    handles.append(model.decoder[dec_key].register_forward_hook(make_decoder_hook()))

    def reassemble_hook(module, input, output):
        collected['reassemble'].append(output[-1].mean(dim=[-2, -1]).detach().cpu())
    def fusion_hook(module, input, output):
        collected['fusion'].append(output.mean(dim=[-2, -1]).detach().cpu())

    handles.append(model.decoder[dec_key].reassemble_blocks.register_forward_hook(reassemble_hook))
    handles.append(model.decoder[dec_key].project.register_forward_hook(fusion_hook))

    for idx, fname in enumerate(selected):
        path = os.path.join(DATASET_DIR, 'synthetic', 'images', fname)
        img = np.array(Image.open(path).convert('RGB'))
        lbl = label_dict[fname]
        kp, distance = ds.calculate_landmarks_distance(p_axes=body_points, q=lbl['q'], r=lbl['r'])
        for i_ in range(len(kp)):
            sel = distance[:-3] < distance[i_]
            tmp = [p for p, b in zip(kp, sel) if b]
            tmp.append(kp[i_])
            if len(tmp) <= 3: kp[i_][2] = 1
            else:
                res = convex_hull(tmp)
                for p in res:
                    if kp[i_] == p: kp[i_][2] = 1
        crop = get_crop(img, kp, ds)

        batch_imgs = []
        for _ in range(args.n):
            aug = trans(image=crop.copy())['image']
            batch_imgs.append(norm(image=aug)['image'])
        all_tensors = [torch.from_numpy(b).permute(2, 0, 1) for b in batch_imgs]

        sub_batch = 50
        for start in range(0, len(all_tensors), sub_batch):
            chunk = torch.stack(all_tensors[start:start+sub_batch]).to(device)
            with torch.no_grad():
                model(chunk)
            del chunk
            torch.cuda.empty_cache()

        for li in range(4):
            features_per_stage[f'enc_{li}'].append(torch.cat(collected[f'enc_{li}']))
        features_per_stage['reassemble'].append(torch.cat(collected['reassemble']))
        features_per_stage['fusion'].append(torch.cat(collected['fusion']))
        features_per_stage['decoder'].append(torch.cat(collected['decoder']))

        for k in collected:
            collected[k].clear()

        colors.extend([idx] * args.n)

    for h in handles:
        h.remove()

    stage_names = ['enc_0', 'enc_1', 'enc_2', 'enc_3', 'reassemble', 'fusion', 'decoder']
    stage_labels = ['Encoder L0', 'Encoder L1', 'Encoder L2', 'Encoder L3',
                    'Reassemble', 'Fusion', 'Decoder']
    feature_tensors = [torch.cat(features_per_stage[s]) if features_per_stage[s] else None
                       for s in stage_names]

    out_path = args.out_name if args.out_name else \
        f'imageprocess/pca_stages_m{args.m}n{args.n}_{"+".join(model_type)}_{args.aug_type}.png'
    os.makedirs('imageprocess', exist_ok=True)
    pca_plot(feature_tensors, colors, stage_labels, '+'.join(model_type), args.aug_type, args.m, out_path)
