import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils_datasets.speedplus_utils_main.utils import points
import numpy as np
from Hyperpose_net.losses.kp_loss import heatmaps_to_keypoints

import math
import cv2
from cv2 import solvePnP, solvePnPRansac
from post_process import pose_calculats_from_coors, pose_calculats_from_kps, compute_pose_error

def valid_one_epoch(model, dataloader, model_type, criterion, device):
    model.eval()
    losses_epoch = []
    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Validing", ncols=80):
            target_list = []
            for i in range(samples.shape[0]):
                target_list.append({k: targets[k][i].to(device) for k in targets.keys()})
            gt_target = []
            imageshapes = []
            org_imgs_list = []
            coors_gt_list = []
            mask_gt_list = []
            target_dict = {}
            for img, target in zip(samples, target_list):
                gtbbox = torch.round(target["boxes"].squeeze(0))
                org_imgs_list.append(img.to(device))
                if 'coordinates' in model_type or 'coordinates_gs' in model_type:
                    coors_gt_list.append(target["coors_gt"].float().to(device))
                    mask_gt_list.append(target["mask_gt"].float().to(device))
                if 'keypoints_gs' in model_type:
                    gt_target.append(target['keypoints'])
                imageshapes.append(torch.tensor([gtbbox[2]-gtbbox[0], gtbbox[3]-gtbbox[1]]))
            with torch.amp.autocast('cuda'):
                inputs = torch.stack(org_imgs_list)
                if 'coordinates' in model_type or 'coordinates_gs' in model_type:
                    coors_gt = torch.stack(coors_gt_list)
                    mask_gt = torch.stack(mask_gt_list)
                    target_dict['coordinates'] = coors_gt
                    target_dict['mask'] = mask_gt
                if 'keypoints_gs' in model_type:
                    target_dict['keypoints_gs'] = torch.stack(gt_target)
                outputs = model(inputs)
                imageshapes = torch.stack(imageshapes)
                losses = criterion(outputs, imageshapes, target_dict)
            losses_epoch.append(losses.detach().cpu())
    return torch.tensor(losses_epoch).mean()


def eval_one_epoch(model, dataloader, model_type, criterion, K, device, bc=None):
    model.eval()

    result_dicts = {}
    for model_name in model_type:
        result_dicts[model_name] = []
    activate = torch.nn.Sigmoid()

    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Validing", ncols=80):
            rgt = torch.tensor(targets["r_gt"]).squeeze()
            qgt = torch.tensor(targets["q_gt"]).squeeze()
            gtkp = torch.tensor(targets["keypoints"].squeeze())
            target_list = []
            for i in range(samples.shape[0]):
                target_list.append({k: targets[k][i].to(device) for k in targets.keys()})
            model.zero_grad()
            org_imgs_list = []
            for img, target in zip(samples, target_list):
                gtbbox = torch.round(target["boxes"].squeeze(0))
                org_imgs_list.append(img.to(device))
                imageshape = torch.tensor([gtbbox[2]-gtbbox[0], gtbbox[3]-gtbbox[1]])
            with torch.cuda.amp.autocast():
                inputs = torch.stack(org_imgs_list)
                outputs = model(inputs)

            if 'keypoints_gs' in model_type:
                result1 = heatmaps_to_keypoints(outputs['keypoints_gs'], imageshape)
                p_all = result1[0][0].squeeze()

                p_all[:, 0] = p_all[:, 0] + gtbbox[0]
                p_all[:, 1] = p_all[:, 1] + gtbbox[1]

                pgt = np.array(points)
                p_ = torch.as_tensor(p_all)
                is_true, kps_qvecs, kps_tvecs = pose_calculats_from_kps(K, pgt, p_)
                print('test of keypoints_gs')
                err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose = compute_pose_error(
                    kps_qvecs, kps_tvecs, qgt, rgt, is_true
                )

                result_dict = {
                    'err_ori': err_ori_deg.tolist(), \
                    'los_r': err_r_abs.tolist(), \
                }
                result_dicts['keypoints_gs'].append(result_dict)


            if 'coordinates' in model_type:
                coormap_masked = outputs['coordinates'].clone()
                if hasattr(criterion, 'los_fnc') and hasattr(criterion.los_fnc.get('coordinates', None), 'post_process'):
                    coormap_masked = criterion.los_fnc['coordinates'].post_process(coormap_masked)
                mask_bool_est_ = activate(outputs['mask']) > 0.5
                mask_bool_est = mask_bool_est_.expand_as(coormap_masked).cpu()
                coormap_masked[~mask_bool_est] = float('nan')
                print('test of coordinates')
                is_true, coors_qvecs, coors_tvecs = pose_calculats_from_coors(K,coormap_masked.squeeze().permute(2,1,0).cpu().detach().numpy(), gtbbox.cpu().detach().numpy())

                err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose = compute_pose_error(
                    coors_qvecs, coors_tvecs, qgt, rgt, is_true
                )

                result_dict = {
                    'err_ori': err_ori_deg.tolist(), \
                    'los_r': err_r_abs.tolist(), \
                }
                result_dicts['coordinates'].append(result_dict)

            if 'coordinates_gs' in model_type and bc is not None:
                out = outputs['coordinates_gs'].detach()
                B, _, H, W = out.shape
                total_bins = bc.total_bins
                out = out.view(B, 3, total_bins, H, W).permute(0, 3, 4, 1, 2)
                probs = F.softmax(out, dim=-1)
                coormap_np = bc.bins_to_value(probs)
                coormap_value = torch.from_numpy(np.transpose(coormap_np, (0, 3, 1, 2))).float().to(device)
                if bc.use_mask and 'mask' in outputs:
                    mask_bool_est_ = activate(outputs['mask']) > 0.5
                    mask_bool_est = mask_bool_est_.expand_as(coormap_value).cpu()
                    coormap_value[~mask_bool_est] = float('nan')
                print('test of coordinates_gs')
                is_true, coors_qvecs, coors_tvecs = pose_calculats_from_coors(K, coormap_value.squeeze().permute(2,1,0).cpu().numpy(), gtbbox.cpu().numpy())
                err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose = compute_pose_error(
                    coors_qvecs, coors_tvecs, qgt, rgt, is_true)
                result_dict = {'err_ori': err_ori_deg.tolist(), 'los_r': err_r_abs.tolist()}
                result_dicts['coordinates_gs'].append(result_dict)

    return result_dicts
