import torch
from tqdm import tqdm
def train_one_epoch(model, dataloader, model_type, criterion, optimizer, scheduler, device):
    model.train()
    losses_epoch = []
    pbar = tqdm(dataloader, desc="Training", ncols=80)
    for idx, (samples, targets) in enumerate(pbar):
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
            imageshapes.append(target["imageshape"])
        inputs = torch.stack(org_imgs_list)
        if 'coordinates' in model_type or 'coordinates_gs' in model_type:
            coors_gt = torch.stack(coors_gt_list)
            mask_gt = torch.stack(mask_gt_list)
            target_dict['coordinates'] = coors_gt
            target_dict['mask'] = mask_gt
        if 'keypoints_gs' in model_type:
            target_dict['keypoints_gs'] = torch.stack(gt_target)
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            imageshapes = torch.stack(imageshapes)
            losses = criterion(outputs, imageshapes, target_dict)
        pbar.set_postfix(loss=f"{losses.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        losses_epoch.append(losses.detach().cpu())
    return torch.tensor(losses_epoch).mean()
