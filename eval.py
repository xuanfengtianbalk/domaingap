import torch
from utils_datasets.speedplus_utils_main.my_augmentation import crop_tensor_image
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
        for samples, targets in tqdm(dataloader, desc="Validing"):
            target_list = []
            for i in range(samples.shape[0]):
                target_list.append({k: targets[k][i].to(device) for k in targets.keys()})
            model.zero_grad()
            gt_target=[]
            imageshapes=[]
            org_imgs_list=[]
            coors_gt_list = []
            mask_gt_list = []
            target_dict = {}
            for img, target in zip(samples, target_list):
            # for img, target in tqdm(zip(samples, target_list), desc="training", total=min(len(samples), len(target_list))):
                gtbbox=torch.round(target["boxes"].squeeze(0))

                org_imgs_list.append((crop_tensor_image(img.clone(), gtbbox.clone()).float()).to(device)\
                                     )

                if 'coordinates' in model_type:
                    coors_gt_list.append((crop_tensor_image(target["coors_gt"].clone(), gtbbox.clone()).float()).to(device) \
                                         )
                    mask_gt_list.append(
                        (crop_tensor_image(target["mask_gt"].clone().unsqueeze(0), gtbbox.clone()).float()).squeeze(
                            0).to(device) \
                        )
                if 'keypoints_gs' in model_type:
                    t = target['keypoints'] - torch.tensor([gtbbox[0], gtbbox[1], 0], device=device)
                    gt_target.append(t)
                # print(t)

                # padded_label.append(target["padded_ratio"]==0.0)
                imageshape = torch.tensor([gtbbox[2]-gtbbox[0],gtbbox[3]-gtbbox[1]])
                imageshapes.append(imageshape)
                # dc_list.append(target['style_idx'])
            with torch.cuda.amp.autocast():
                # 原本为loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                inputs = torch.stack(
                    [torch.nn.functional.interpolate(img_.unsqueeze(0), size=(256, 256), mode='nearest',
                                                     ).squeeze(0) for img_ in org_imgs_list])
                if 'coordinates' in model_type:
                    coors_gt = torch.stack(
                        [torch.nn.functional.interpolate(coors_gt_.unsqueeze(0), size=(256, 256), mode='nearest',
                                                         ).squeeze(0) for coors_gt_ in coors_gt_list])
                    mask_gt = torch.stack([torch.nn.functional.interpolate(mask_gt_.unsqueeze(0).unsqueeze(0),
                                                                           size=(256, 256), mode='nearest'
                                                                           ).squeeze(0).squeeze(0) for mask_gt_ in
                                           mask_gt_list])

                    target_dict['coordinates'] = coors_gt
                    target_dict['mask'] = mask_gt
                if 'keypoints_gs' in model_type:
                    target_dict['keypoints_gs'] = torch.stack(gt_target)
                outputs = model(inputs)
                imageshapes = torch.stack(imageshapes)
                # labels = torch.stack(gt_target)
                losses = criterion(outputs,imageshapes,target_dict)
            # losses = sum(loss for loss in loss_dict.values())
            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()
            losses_epoch.append(losses.item())
            # break
    # print('training/loss_all', )
    return torch.tensor(losses_epoch).mean()


def eval_one_epoch(model, dataloader, model_type, criterion, K, device):
    model.eval()

    result_dicts = {}
    for model_name in model_type:
        result_dicts[model_name] = []
    activate = torch.nn.Sigmoid()
    err_rot_list=[]
    err_trans_list=[]
    err_pose_list=[]

    miss_idx = 0
    fail_idx = 0
    deg_cm_5_5_idx = 0
    idx_load = 0
    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Validing"):
            # if idx_load > 10:
            #     break
            idx_load = idx_load + 1
            rgt = torch.tensor(targets["r_gt"]).squeeze()
            qgt = torch.tensor(targets["q_gt"]).squeeze()
            gtkp = torch.tensor(targets["keypoints"].squeeze())
            target_list = []
            for i in range(samples.shape[0]):
                target_list.append({k: targets[k][i].to(device) for k in targets.keys()})
            model.zero_grad()
            gt_target=[]
            imageshapes=[]
            org_imgs_list=[]
            for img, target in zip(samples, target_list):
            # for img, target in tqdm(zip(samples, target_list), desc="training", total=min(len(samples), len(target_list))):
                gtbbox=torch.round(target["boxes"].squeeze(0))

                org_imgs_list.append((crop_tensor_image(img.clone(), gtbbox.clone()).float()).to(device)\
                                     )

                t=target['keypoints']-torch.tensor([gtbbox[0],gtbbox[1],0],device=device)
                # print(t)
                gt_target.append(t)
                # padded_label.append(target["padded_ratio"]==0.0)
                imageshape = torch.tensor([gtbbox[2]-gtbbox[0],gtbbox[3]-gtbbox[1]])
                # imageshapes.append(imageshape)
                # dc_list.append(target['style_idx'])
            with torch.cuda.amp.autocast():
                # 原本为loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
                inputs = torch.stack([torch.nn.functional.interpolate(img_.unsqueeze(0), size=(256, 256), mode='bilinear',
                                                        align_corners=False).squeeze(0) for img_ in org_imgs_list])
                outputs = model(inputs)
                # imageshapes = torch.stack(imageshapes)
                # labels = torch.stack(gt_target)

                # losses = criterion(outputs,imageshapes,labels)


            if 'keypoints_gs' in model_type:
                result1 = heatmaps_to_keypoints(outputs['keypoints_gs'], imageshape)
                p_all = result1[0][0].squeeze()

                p_all[:, 0] = p_all[:, 0] + gtbbox[0]
                p_all[:, 1] = p_all[:, 1] + gtbbox[1]

                # kps = p_all.clone()
                pgt = np.array(points)
                p_ = torch.as_tensor(p_all)
                is_true, kps_qvecs, kps_tvecs = pose_calculats_from_kps(K, pgt, p_)
                print('test of keypoints_gs')
                # 每帧调用
                err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose = compute_pose_error(
                    kps_qvecs, kps_tvecs, qgt, rgt, is_true
                )

                result_dict = {
                    'err_ori': err_ori_deg.tolist(), \
                    'los_r': err_r_abs.tolist(), \
                    # 'q_est': kps_qvecs.tolist(), \
                    # 'q_gt': qgt.tolist(), \
                    # 'kps': kps.tolist(), \
                    # 'gt_kps': gtkp.tolist()
                }
                result_dicts['keypoints_gs'].append(result_dict)


            if 'coordinates' in model_type:
                coormap_masked = outputs['coordinates'].clone()
                mask_bool_est_ = activate(outputs['mask']) > 0.5
                mask_bool_est = mask_bool_est_.expand_as(coormap_masked).cpu()
                coormap_masked[~mask_bool_est] = float('nan')
                print('test of coordinates')
                is_true, coors_qvecs, coors_tvecs = pose_calculats_from_coors(K,coormap_masked.squeeze().permute(2,1,0).cpu().detach().numpy(), gtbbox.cpu().detach().numpy())

                # 每帧调用
                err_ori_deg, err_r_rel, err_r_abs, err_pose, inc_fail_miss, good_pose = compute_pose_error(
                    coors_qvecs, coors_tvecs, qgt, rgt, is_true
                )

                # err_pose_list.append(err_pose)
                # err_rot_list.append(err_ori_deg)
                # err_trans_list.append(err_r_rel)
                # if inc_fail_miss:
                #     fail_idx += 1
                #     miss_idx += 1
                # if good_pose:
                #     deg_cm_5_5_idx += 1

                result_dict = {
                    'err_ori': err_ori_deg.tolist(), \
                    'los_r': err_r_abs.tolist(), \
                    # 'q_est': coors_qvecs.tolist(), \
                    # 'q_gt': qgt.tolist(), \
                    # 'kps': kps.tolist(), \
                    # 'gt_kps': gtkp.tolist()
                }
                result_dicts['coordinates'].append(result_dict)


    # # print('training/loss_all', )
    # qmean_value = sum(err_rot_list) / len(err_rot_list)
    # print('--------eval---------')
    # print('qerr_mean',qmean_value)
    # tmean_value = sum(err_trans_list) / len(err_trans_list)
    # print('terr_mean', tmean_value)
    #
    # posemean_value = sum(err_pose_list) / len(err_pose_list)
    # print('loss_mean', posemean_value)
    # print('fail num:',fail_idx,'full num:', idx_load+1)
    # print("5deg/5cm:", 100*deg_cm_5_5_idx/(idx_load+1))

    return result_dicts