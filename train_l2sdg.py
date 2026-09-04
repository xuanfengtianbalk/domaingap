import torch
from tqdm import tqdm
from Hyperpose_net.losses.l2sdg import mmd_rbf


def train_one_epoch_l2sdg(model, dataloader, model_type, criterion, optimizer,
                          scheduler, device, wae=None, wae_optimizer=None,
                          beta=0.5, lambda_norm=1e-4, lambda_mmd=1e-3,
                          perturb_steps=1, enable=True):
    """L2SDG training (Qiao et al., CVPR 2020): WAE adversarial domain
    augmentation + meta-learning. enable=False -> plain training."""
    model.train()
    losses_epoch = []
    pbar = tqdm(dataloader, desc="Training(L2SDG)", ncols=80)
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

        if not enable or wae is None:
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

        B = samples.shape[0]
        inputs_detached = inputs.detach()

        # ── 1. inner: adversarial WAE update (maximize task loss on perturbed) ──
        for _ in range(perturb_steps):
            wae_optimizer.zero_grad()
            x_pert = wae(inputs_detached)
            with torch.amp.autocast('cuda'):
                outputs_pert = model(x_pert)
                L_adv = criterion(outputs_pert, imageshapes_t, target_dict)
            z = wae.encode(inputs_detached)
            z_prior = torch.randn(B, wae.latent_dim, device=device)
            wae_loss = -L_adv + lambda_norm * wae.delta_norm(inputs_detached) \
                       + lambda_mmd * mmd_rbf(z, z_prior)
            wae_loss.backward()
            wae_optimizer.step()
            del outputs_pert, L_adv, wae_loss, x_pert
        # adversarial backward polluted model grads — clear them
        optimizer.zero_grad()

        # ── 2. outer: meta update on updated-WAE perturbation (sequential
        #    backward: only ONE model graph alive at a time) ──
        with torch.no_grad():
            x_pert2 = wae(inputs_detached)
        with torch.amp.autocast('cuda'):
            outputs_pert2 = model(x_pert2)
            L_meta = criterion(outputs_pert2, imageshapes_t, target_dict)
        total_val = L_meta.item()
        L_meta.backward()
        del outputs_pert2, L_meta, x_pert2

        # ── 3. clean task loss (beta-weighted) ──
        with torch.amp.autocast('cuda'):
            outputs_orig = model(inputs)
            L_orig = criterion(outputs_orig, imageshapes_t, target_dict)
        total_val = total_val + beta * L_orig.item()
        (beta * L_orig).backward()
        del outputs_orig, L_orig

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses_epoch.append(torch.tensor(total_val))
        pbar.set_postfix(loss=f"{total_val:.4f}", lr=optimizer.param_groups[0]['lr'])
    return torch.tensor(losses_epoch).mean()
