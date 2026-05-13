"""
Measure augmentation-induced label distortion.
PnP from TRANSFORMED kp/coords vs original pose.
"""
import sys, os, argparse, json, numpy as np
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')

import torch
from tqdm import tqdm
from run import load_config, build_dataset
from post_process import pose_calculats_from_kps, pose_calculats_from_coors, compute_pose_error
from utils_datasets.speedplus_utils_main.utils import points, Camera


def test_kps(loader, aug_type, n_max=500):
    err_ori, err_r, fail_cnt, good_cnt = [], [], 0, 0
    cnt = 0
    for samples, targets in tqdm(loader, desc=f'{aug_type} kps', total=n_max):
        r_gt = targets["r_gt"].squeeze()
        q_gt = targets["q_gt"].squeeze()
        cnt += 1
        for i in range(samples.shape[0]):
            kp = targets["keypoints"][i].squeeze(0).cpu().numpy()
            gtbbox = targets["boxes"][i].squeeze(0).cpu().numpy()
            kp[:, 0] += gtbbox[0]
            kp[:, 1] += gtbbox[1]
            valid = np.ones(len(kp), dtype=bool)
            pgt = np.array(points)
            pf = np.zeros((len(valid), 3), dtype=np.float32)
            pf[:, :2] = kp[valid, :2]; pf[:, 2] = 1
            is_true, qv, tv = pose_calculats_from_kps(Camera.K, pgt[valid], torch.as_tensor(pf))
            e_ori, _, e_r, _, fail, good = compute_pose_error(qv, tv, q_gt, r_gt, is_true)
            if fail: fail_cnt += 1
            else:
                eo, er = float(e_ori), float(e_r)
                if eo > 1 or er > 1:
                    sid = loader.dataset.sample_ids[cnt-1] if hasattr(loader.dataset, 'sample_ids') else f'sample_{cnt}'
                    print(f'  [!] kps {aug_type} #{cnt} {sid}: ori={eo:.1f}° trans={er:.3f}m')
                err_ori.append(eo); err_r.append(er)
                if good: good_cnt += 1
        if cnt >= n_max:
            break
    return err_ori, err_r, fail_cnt, good_cnt


def test_coords(loader, aug_type, n_max=200):
    err_ori, err_r, fail_cnt, good_cnt = [], [], 0, 0
    cnt = 0
    for samples, targets in tqdm(loader, desc=f'{aug_type} coords', total=n_max):
        r_gt = targets["r_gt"].squeeze()
        q_gt = targets["q_gt"].squeeze()
        cnt += 1
        for i in range(samples.shape[0]):
            if "coors_gt" not in targets or "mask_gt" not in targets:
                fail_cnt += 1; continue
            coors = targets["coors_gt"][i].clone()
            mask_gt = targets["mask_gt"][i].float()
            gtbbox = torch.round(targets["boxes"][i].squeeze(0))

            mask = mask_gt > 0.5
            mask = mask.expand_as(coors)
            coors[~mask] = float('nan')

            coors = coors.permute(2, 1, 0).cpu().detach().numpy()
            bbox_np = gtbbox.cpu().numpy()
            is_true, qv, tv = pose_calculats_from_coors(Camera.K, coors, bbox_np)
            e_ori, _, e_r, _, fail, good = compute_pose_error(qv, tv, q_gt, r_gt, is_true)
            if fail: fail_cnt += 1
            else:
                eo, er = float(e_ori), float(e_r)
                if eo > 1 or er > 1:
                    sid = loader.dataset.sample_ids[cnt-1] if hasattr(loader.dataset, 'sample_ids') else f'#{cnt}'
                    print(f'  [!] coords {aug_type} #{cnt} {sid}: ori={eo:.1f}° trans={er:.3f}m')
                err_ori.append(eo); err_r.append(er)
                if good: good_cnt += 1
        if cnt >= n_max:
            break
    return err_ori, err_r, fail_cnt, good_cnt


def print_stats(name, e_ori, e_r, fail, good):
    n = len(e_ori)
    if n == 0:
        print(f"  {name}: All failed PnP")
        return
    print(f"  {name}: n={n}, fail={fail}, 5°5cm={100*good/n:.1f}%")
    print(f"    OriErr:  mean={np.mean(e_ori):.1f}° med={np.median(e_ori):.1f}°")
    print(f"    TransErr: mean={np.mean(e_r):.3f}m med={np.median(e_r):.3f}m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, nargs='+', default=['none','aug4','augmix'])
    parser.add_argument('--mode', type=str, default='sunlamp')
    args = parser.parse_args()

    config = load_config('configs/cfg.yaml')

    for aug_type in args.aug:
        print(f"\n{'='*50}")
        print(f"  {aug_type} on {args.mode}")
        print(f"{'='*50}")

        dataset = build_dataset(config, args.mode, aug_type=aug_type)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        eo_k, er_k, f_k, g_k = test_kps(loader, aug_type)
        print_stats('keypoints', eo_k, er_k, f_k, g_k)

        dataset2 = build_dataset(config, args.mode, aug_type=aug_type)
        loader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=4)
        eo_c, er_c, f_c, g_c = test_coords(loader2, aug_type)
        print_stats('coordinates', eo_c, er_c, f_c, g_c)
