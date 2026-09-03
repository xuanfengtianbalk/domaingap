import torch
from tqdm import tqdm


def train_one_epoch_stable(model, dataloader, model_type, criterion, optimizer,
                           scheduler, device, rff_layer=None, feature_net=None,
                           state=None, weight_steps=3, weight_lr=0.1,
                           enable=True):
    """StableNet-style training: sample reweighting with RFF decorrelation
    + global balancing. When enable=False, behaves like plain train_one_epoch."""
    model.train()
    losses_epoch = []
    pbar = tqdm(dataloader, desc="Training(Stable)", ncols=80)
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

        if not enable or state is None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                imageshapes_t = torch.stack(imageshapes)
                losses = criterion(outputs, imageshapes_t, target_dict)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            losses_epoch.append(losses.detach().cpu())
            pbar.set_postfix(loss=f"{losses.item():.4f}", lr=optimizer.param_groups[0]['lr'])
            continue

        # ── StableNet path ──
        with torch.amp.autocast('cuda'):
            outputs, features = model(inputs, return_features=True)

        # last-layer patch features → GAP → feature net → RFF
        patch_feats = features[-1][0].float()          # [B, C, h, w]
        z0 = torch.nn.functional.adaptive_avg_pool2d(patch_feats, 1).flatten(1)
        z = feature_net(z0)                            # [B, n_z]
        u = rff_layer(z)                               # [B, m]

        # per-sample task losses (criterion is scalar-mean; slice per sample)
        B = samples.shape[0]
        per_loss = []
        imageshapes_t = torch.stack(imageshapes)
        with torch.amp.autocast('cuda'):
            for i in range(B):
                out_i = {}
                for k, v in outputs.items():
                    if torch.is_tensor(v):
                        out_i[k] = v[i:i + 1]
                td_i = {}
                for k, v in target_dict.items():
                    if torch.is_tensor(v):
                        td_i[k] = v[i:i + 1]
                per_loss.append(criterion(out_i, imageshapes_t[i:i + 1], td_i))
        per_loss = torch.stack(per_loss).float()   # [B], keep graph for backward

        # learn sample weights w (inner SGD, minimize decorrelation)
        w = torch.ones(B, device=device)
        for _ in range(weight_steps):
            w = w.detach().clone()
            w.requires_grad_(True)
            L_indep = state.loss(w, z.detach(), u.detach())
            grad = torch.autograd.grad(L_indep, w)[0]
            w = (w - weight_lr * grad).detach().clamp(min=0.0)
            w = w / (w.mean() + 1e-8)

        w = w.detach()
        loss = (w * per_loss).sum() / B
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        state.update_global_B(w, z.detach(), u.detach())

        losses_epoch.append(loss.detach().cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
    return torch.tensor(losses_epoch).mean()
