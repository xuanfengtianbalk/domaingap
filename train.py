import torch
from utils_datasets.speedplus_utils_main.my_augmentation import crop_tensor_image
from tqdm import tqdm
def train_one_epoch(model, dataloader, model_type, criterion, optimizer, scheduler, device):
    model.train()
    losses_epoch = []
    pbar=tqdm(dataloader, desc="Training")
    for idx, (samples, targets) in enumerate(pbar):
        target_list = []
        for i in range(samples.shape[0]):
            target_list.append({k: targets[k][i].to(device) for k in targets.keys()})
        model.zero_grad()
        gt_target=[]
        imageshapes=[]
        org_imgs_list=[]
        coors_gt_list=[]
        mask_gt_list=[]
        target_dict={}
        for img, target in zip(samples, target_list):
        # for img, target in tqdm(zip(samples, target_list), desc="training", total=min(len(samples), len(target_list))):

            gtbbox=torch.round(target["boxes"].squeeze(0))

            org_imgs_list.append((crop_tensor_image(img.clone(), gtbbox.clone()).float()).to(device)\
                                 )
            if 'coordinates' in model_type:
                coors_gt_list.append((crop_tensor_image(target["coors_gt"].clone(), gtbbox.clone()).float()).to(device) \
                                     )
                mask_gt_list.append((crop_tensor_image(target["mask_gt"].clone().unsqueeze(0), gtbbox.clone()).float()).squeeze(0).to(device) \
                                     )
            if 'keypoints_gs' in model_type:

                t=target['keypoints']-torch.tensor([gtbbox[0],gtbbox[1],0],device=device)
                gt_target.append(t)
            # print(t)

            # padded_label.append(target["padded_ratio"]==0.0)
            imageshape = torch.tensor([gtbbox[2]-gtbbox[0],gtbbox[3]-gtbbox[1]])
            imageshapes.append(imageshape)
            # dc_list.append(target['style_idx'])
        with torch.cuda.amp.autocast():
            # 原本为loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            inputs = torch.stack([torch.nn.functional.interpolate(img_.unsqueeze(0), size=(256, 256), mode='nearest',
                                                    ).squeeze(0) for img_ in org_imgs_list])
            if 'coordinates' in model_type:
                coors_gt = torch.stack([torch.nn.functional.interpolate(coors_gt_.unsqueeze(0), size=(256, 256), mode='nearest',
                                                        ).squeeze(0) for coors_gt_ in coors_gt_list])
                mask_gt = torch.stack([torch.nn.functional.interpolate(mask_gt_.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest'
                                                                       ).squeeze(0).squeeze(0) for mask_gt_ in mask_gt_list])

                target_dict['coordinates'] = coors_gt
                target_dict['mask'] = mask_gt
            if 'keypoints_gs' in model_type:

                target_dict['keypoints_gs'] = torch.stack(gt_target)

            outputs = model(inputs)
            imageshapes = torch.stack(imageshapes)
            # labels = torch.stack(gt_target)
            losses = criterion(outputs,imageshapes,target_dict)
        pbar.set_postfix(loss=f"{losses.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        # losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        losses_epoch.append(losses.item())
        # break
    # print('training/loss_all', )
    return torch.tensor(losses_epoch).mean()