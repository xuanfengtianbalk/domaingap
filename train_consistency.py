import torch
import torch.nn.functional as F
from tqdm import tqdm


def _heatmap_kl(branch, mean):
    """KL over spatial dims for keypoints heatmaps [B, K, H, W]."""
    B, K, H, W = branch.shape
    b = branch.reshape(B * K, H * W)
    m = mean.reshape(B * K, H * W)
    return F.kl_div(F.log_softmax(b, dim=1), F.softmax(m, dim=1), reduction='batchmean')


def _bin_kl(branch, mean, mask, total_bins):
    """KL over bin dim for coordinates_gs, masked by valid pixels."""
    B, _, H, W = branch.shape
    b = branch.reshape(B, 3, total_bins, H, W).permute(0, 3, 4, 1, 2)  # [B,H,W,3,tb]
    m = mean.reshape(B, 3, total_bins, H, W).permute(0, 3, 4, 1, 2)
    mask_bool = mask.squeeze(1) > 0.5  # [B, H, W]
    b = b[mask_bool]  # [N, 3, tb]
    m = m[mask_bool]
    if b.shape[0] == 0:
        return torch.tensor(0.0, device=branch.device)
    b = b.reshape(-1, total_bins)  # [N*3, tb]
    m = m.reshape(-1, total_bins)
    return F.kl_div(F.log_softmax(b, dim=1), F.softmax(m, dim=1), reduction='batchmean')


def train_one_epoch_randconv(model, dataloader, model_type, criterion, optimizer,
                              scheduler, device, randconv_layers=None, n_aug_branches=0,
                              bc=None, consistency_weight=0.1, n_branches=3):
    """Consistency training: aug branches pre-computed in dataset workers,
    randconv branches applied on-the-fly on GPU."""
    model.train()
    losses_epoch = []
    pbar = tqdm(dataloader, desc="Train", ncols=100)
    for (samples, targets) in pbar:

        # samples shape: [B, N, C, H, W] with aug, or [B, C, H, W] without
        has_variants = (samples.dim() == 5)
        if has_variants:
            B, N_img, C, H, W = samples.shape
            # original at index 0, aug variants at indices 1..n_aug_branches
            img_original = samples[:, 0].to(device)
            aug_variants = samples[:, 1:].to(device)  # [B, n_aug, C, H, W]
        else:
            img_original = samples.to(device)
            aug_variants = None

        # prepare targets once (shared across branches)
        target_list = [{} for _ in range(samples.shape[0])]
        for k in targets.keys():
            for i in range(samples.shape[0]):
                target_list[i][k] = targets[k][i].to(device)

        imageshapes_list = []
        coors_gt_list = []
        mask_gt_list = []
        gt_target_keypts = []
        for target in target_list:
            imageshapes_list.append(target["imageshape"])
            if 'coordinates' in model_type or 'coordinates_gs' in model_type:
                coors_gt_list.append(target["coors_gt"].float().to(device))
                mask_gt_list.append(target["mask_gt"].float().to(device))
            if 'keypoints_gs' in model_type:
                gt_target_keypts.append(target['keypoints'])

        imageshapes = torch.stack(imageshapes_list)
        target_dict = {}
        if 'coordinates' in model_type or 'coordinates_gs' in model_type:
            target_dict['coordinates'] = torch.stack(coors_gt_list)
            target_dict['mask'] = torch.stack(mask_gt_list)
        if 'keypoints_gs' in model_type:
            target_dict['keypoints_gs'] = torch.stack(gt_target_keypts)

        # ---- build all branches ----
        branches = []
        aug_idx = 0
        for j in range(n_branches):
            if randconv_layers is not None and j < len(randconv_layers):
                x = randconv_layers[j](img_original)
            elif aug_variants is not None and aug_idx < n_aug_branches:
                x = aug_variants[:, aug_idx]
                aug_idx += 1
            else:
                x = img_original
            with torch.amp.autocast('cuda'):
                outputs_j = model(x)
            branches.append(outputs_j)

        # convert autocast float16 → float32 for stable consistency loss
        branches = [{k: v.float() for k, v in b.items()} for b in branches]

        # ---- task loss on all branches ----
        # ---- task loss on first branch only (line 20) ----
        with torch.amp.autocast('cuda'):
            L_task = criterion(branches[0], imageshapes, target_dict)

        # ---- consistency loss: KL(ŷⱼ ‖ ȳ) (lines 19-20) ----
        L_cons = torch.tensor(0.0, device=device)
        if consistency_weight > 0:
            for key in branches[0].keys():
                mean = (sum(b[key] for b in branches) / float(n_branches)).detach()  # ȳ (line 19)

                if key == 'keypoints_gs':
                    for j in range(n_branches):
                        L_cons += _heatmap_kl(branches[j][key], mean)

                elif key == 'coordinates_gs' and bc is not None:
                    for j in range(n_branches):
                        L_cons += _bin_kl(
                            branches[j][key], mean,
                            target_dict['mask'], bc.total_bins)

                elif key == 'coordinates':
                    mask_bool = target_dict['mask'].squeeze(1) > 0.5
                    for j in range(n_branches):
                        b = branches[j][key].permute(0, 2, 3, 1)[mask_bool]
                        m = mean.permute(0, 2, 3, 1)[mask_bool]
                        if b.shape[0] > 0:
                            L_cons += F.mse_loss(b, m)

                elif key == 'mask':
                    for j in range(n_branches):
                        L_cons += F.mse_loss(branches[j][key], mean)

        total_loss = L_task + consistency_weight * L_cons

        pbar.set_postfix(L=f"{total_loss.item():.4f}",
                         T=f"{L_task.item():.4f}",
                         C=f"{L_cons.item():.2e}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses_epoch.append(total_loss.detach().cpu())

    return torch.tensor(losses_epoch).mean()
