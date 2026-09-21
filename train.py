import torch
from tqdm import tqdm
def train_one_epoch(model, dataloader, model_type, criterion, optimizer, scheduler, device, l2sp=None):
    """l2sp: dict with keys alpha, beta, existing [(w, w0)], new [w] —
    L2-SP (ICML 2018) penalty, mirroring the official TensorFlow code:
        loss += alpha * sum(0.5 * ||w - w0||^2) over existing 'weights'
        loss += beta  * sum(0.5 * ||w||^2)    over new 'weights'
    """
    model.train()
    losses_epoch = []
    pbar = tqdm(dataloader, desc="Training", ncols=80)
    for idx, (samples, targets) in enumerate(pbar):
        model.zero_grad()
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
            if 'coordinates' in model_type or 'coordinates_gs' in model_type or 'coordinates_DER' in model_type or 'coordinates_gs_EDL' in model_type:
                coors_gt_list.append(target["coors_gt"].float().to(device))
                mask_gt_list.append(target["mask_gt"].float().to(device))
            if 'keypoints_gs' in model_type:
                gt_target.append(target['keypoints'])
            imageshapes.append(target["imageshape"])
        inputs = torch.stack(org_imgs_list)
        if 'coordinates' in model_type or 'coordinates_gs' in model_type or 'coordinates_DER' in model_type or 'coordinates_gs_EDL' in model_type:
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
        if l2sp is not None:
            # L2-SP (ICML 2018) decay applied directly to gradients, equivalent
            # to adding alpha/2*||w-w0||^2 + beta/2*||w||^2 to the loss
            # (official TF code computes decay terms in the loss; gradient
            # injection matches that for SGD-momentum and avoids graph bloat)
            for w, w0 in l2sp['existing']:
                grad = l2sp['alpha'] * (w - w0)
                if w.grad is None:
                    w.grad = grad
                else:
                    w.grad += grad
            for w in l2sp['new']:
                grad = l2sp['beta'] * w
                if w.grad is None:
                    w.grad = grad
                else:
                    w.grad += grad
        optimizer.step()
        scheduler.step()
        losses_epoch.append(losses.detach().cpu())
    return torch.tensor(losses_epoch).mean()
