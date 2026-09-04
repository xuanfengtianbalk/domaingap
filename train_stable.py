import torch
from tqdm import tqdm


def train_one_epoch_stable(model, dataloader, model_type, criterion, optimizer,
                           scheduler, device, rff=None, state=None, epoch=0,
                           num_f=1, epochb=20, lrbl=1.0, lambdap=70.0,
                           decay_pow=2, epochp=0, first_step_cons=1.0,
                           lambda_decay_rate=1, lambda_decay_epoch=5,
                           min_lambda_times=0.01, enable=True):
    """StableNet training aligned with official repo (xxgege/StableNet).

    enable=False → plain training (ablation control).
    """
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
        imageshapes_t = torch.stack(imageshapes)

        if not enable or state is None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
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

        B = samples.shape[0]

        # raw penultimate features (GAP), detached — no projection
        cfeatures = torch.nn.functional.adaptive_avg_pool2d(
            features[-1][0].float(), 1).flatten(1).detach()   # [B, d]

        # per-sample task losses (criterion per-image slices)
        per_loss = []
        with torch.amp.autocast('cuda'):
            for i in range(B):
                out_i = {k: v[i:i + 1] for k, v in outputs.items() if torch.is_tensor(v)}
                td_i = {k: v[i:i + 1] for k, v in target_dict.items() if torch.is_tensor(v)}
                per_loss.append(criterion(out_i, imageshapes_t[i:i + 1], td_i))
        per_loss = torch.stack(per_loss).float()              # [B]

        # learn sample weights (warmup: raw ones, official behavior)
        from Hyperpose_net.losses.stable_learning import weight_learner
        if epoch >= epochp:
            w, state = weight_learner(cfeatures, state, rff, num_f, epochb, lrbl,
                                      lambdap, decay_pow, epoch, idx, first_step_cons,
                                      lambda_decay_rate, lambda_decay_epoch, min_lambda_times)
        else:
            w = torch.ones(B, 1, device=device)

        loss = (per_loss * w.squeeze(1)).sum()               # no /B (official)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        state.update(cfeatures, w, epoch, idx)

        losses_epoch.append(loss.detach().cpu())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
    return torch.tensor(losses_epoch).mean()
