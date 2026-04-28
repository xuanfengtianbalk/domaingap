"""
@author: Yuanhao Cai
@date:  2020.03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thresh1=1, thresh2=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = thresh1
        self.t2 = thresh2
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        batch_size = output.size(0)
        keypoint_num = output.size(1)
        loss = 0

        for i in range(batch_size):
            pred = output[i].reshape(keypoint_num, -1)
            gt = label[i].reshape(keypoint_num, -1)

            if not self.has_ohkm:
                weight = torch.gt(valid[i], self.t1).float()
                gt = gt * weight

            tmp_loss = self.calculate(pred, gt)

            if self.has_ohkm:
                tmp_loss = tmp_loss.mean(dim=1)
                weight = torch.gt(valid[i].squeeze(), self.t2).float()
                tmp_loss = tmp_loss * weight
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=0,
                        sorted=False)
                sample_loss = topk_val.mean(dim=0)
            else:
                sample_loss = tmp_loss

            loss = loss + sample_loss

        return loss / batch_size

#mymymy
import dsntnn

def keypointrcnn_loss(keypoint_logits, imageshapes, gt_keypoints, sigma_t):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    assert H == W

    heatmaps = []
    valid = []

    for gt_kp_in_image, imageshape in zip(gt_keypoints,imageshapes):
        # kp = gt_kp_in_image
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            gt_kp_in_image.squeeze(), imageshape, torch.tensor([H,W])
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    # keypoint_logits = keypoint_logits.reshape(N * K, H, W)
    keypoint_logits = keypoint_logits.reshape(N * K, H, W)
    keypoint_hm = dsntnn.flat_softmax(keypoint_logits[valid].unsqueeze(0))
    valid_targets = keypoint_targets[valid].view(-1,1)
    target_coords = torch.cat((valid_targets%H, torch.div(valid_targets, H, rounding_mode='trunc')),dim=-1)+0.5

    target_coords_norm = 2*target_coords.float()/H - 1
    js_losses_kp = dsntnn.js_reg_losses(keypoint_hm, target_coords_norm.unsqueeze(0), sigma_t=sigma_t)
    return dsntnn.average_loss(js_losses_kp)

def reverse_keypointrcnn_loss(keypoint_logits, imageshapes, gt_keypoints, sigma_t):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []

    # for gt_kp_in_image in gt_keypoints:
    #     # kp = gt_kp_in_image
    #
    #     heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
    #         gt_kp_in_image, imageshape, torch.tensor([H,W])
    #     )
    for gt_kp_in_image, imageshape in zip(gt_keypoints, imageshapes):
        # kp = gt_kp_in_image
        gt_kp_in_image[..., 2] = -gt_kp_in_image[..., 2] + 1
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            gt_kp_in_image.squeeze(), imageshape, torch.tensor([H, W])
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    # for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
    #     kp = gt_kp_in_image[midx]
    #     kp[..., 2] = -kp[..., 2] + 1
    #     heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
    #         kp, proposals_per_image, discretization_size
    #     )
    #     heatmaps.append(heatmaps_per_image.view(-1))
    #     valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    # keypoint_logits = keypoint_logits.view(N * K, H, W)
    keypoint_logits = keypoint_logits.reshape(N * K, H, W)
    keypoint_hm = dsntnn.flat_softmax(keypoint_logits[valid].unsqueeze(0))
    valid_targets = keypoint_targets[valid].view(-1,1)
    target_coords = torch.cat((valid_targets%H, valid_targets//H),dim=-1)+0.5
    target_coords_norm = 2*target_coords.float()/H - 1
    js_losses_kp = dsntnn.js_reg_losses(keypoint_hm, target_coords_norm.unsqueeze(0), sigma_t=sigma_t)
    return dsntnn.average_loss(js_losses_kp)

def keypoints_to_heatmap(keypoints, imageshape, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    heatmap_size=heatmap_size[0]
    # offset_x = 0
    # offset_y = 0
    scale_x = heatmap_size / imageshape[0]
    scale_y = heatmap_size / imageshape[1]


    # offset_x = offset_x[:, None]
    # offset_y = offset_y[:, None]
    # scale_x = scale_x[:, None]
    # scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == imageshape[0]
    y_boundary_inds = y == imageshape[1]

    x = x * scale_x
    x = x.floor().long()
    y = y * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

def heatmaps_to_keypoints(maps, imageshape):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    imageshape=imageshape.squeeze()
    scale = 1.005
    offset_x = 0
    offset_y = 0
    widths, heights = scale * imageshape

    offset_x = offset_x - 0.5*(scale-1)*imageshape[0]
    offset_y = offset_y - 0.5*(scale-1)*imageshape[1]

    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()
    num_keypoints = maps.shape[1]

    xy_preds = torch.zeros((1, 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((1, num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(1):
        roi_map_width = int(widths_ceil)
        roi_map_height = int(heights_ceil)

        width_correction = widths / roi_map_width
        height_correction = heights / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
        x_int = pos % w
        y_int = (pos - x_int) // w

        assert len(roi_map.size())==3
        roi_map_norm = dsntnn.flat_softmax(roi_map.unsqueeze(0))

        xy = dsntnn.dsnt(roi_map_norm).squeeze()
        xy_coord = (xy+1) * torch.tensor([roi_map_width,roi_map_height],device=xy.device) * 0.5

        x = xy_coord[:,0] * width_correction
        y = xy_coord[:,1] * height_correction

        xy_preds[i, 0, :] = x + offset_x
        xy_preds[i, 1, :] = y + offset_y
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map_norm[0][torch.arange(num_keypoints), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


class KeypointRCNNLoss(nn.Module):
    """
    关键点检测的联合损失，包含标准关键点损失和反向关键点损失。
    """
    def __init__(self, sigma):
        """
        Args:
            sigma (float): 控制高斯核宽度的超参数，用于软化标签。
        """
        super(KeypointRCNNLoss, self).__init__()
        self.sigma = sigma
    def reset_sigma(self, sigma):
        self.sigma = sigma
    def forward(self, outputs, imageshape, labels):
        """
        Args:
            outputs: 模型的输出，通常包含关键点热图或回归值。
            imageshape: 图像形状信息，可能用于坐标缩放或对齐。
            labels: 关键点的真实标签。

        Returns:
            torch.Tensor: 标量损失值。
        """
        # 标准关键点损失
        loss = keypointrcnn_loss(outputs, imageshape, labels, self.sigma)
        # 反向关键点损失（例如对称性约束）
        loss += reverse_keypointrcnn_loss(outputs, imageshape, labels, self.sigma)
        return loss

if __name__ == '__main__':
    a = torch.ones(1, 17, 12, 12)
    b = torch.ones(1, 17, 12, 12)
    c = torch.ones(1, 17, 1) * 2
    loss = JointsL2Loss()
    # loss = JointsL2Loss(has_ohkm=True)
    device = torch.device('cuda')
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    loss = loss.to(device)
    res = loss(a, c, b)
    print(res)


