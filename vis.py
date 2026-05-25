import matplotlib.pyplot as plt
from typing import List, Union, Optional, Tuple


import torch
import cv2
from tqdm import tqdm

import numpy as np
from post_process import mask_to_display
from Hyperpose_net.losses.kp_loss import heatmaps_to_keypoints

# ------------------ 通用可视化函数 ------------------
def visualize_image_list(
    images: List[np.ndarray],
    title: str = "Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    cols: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 100
):
    """
    将列表中的图像以网格形式显示或保存。

    Args:
        images: 图像列表，每个元素为 numpy 数组，形状 (H, W, 3) 或 (H, W)
        title: 总标题
        save_path: 保存路径（如 'output.png' 或 'output.pdf'），None 时不保存
        show: 是否显示图像
        cols: 网格列数
        figsize: 整个画布大小 (宽, 高)
        dpi: 分辨率
    """
    if not images:
        print("图像列表为空，无可视化内容。")
        return

    num = len(images)
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
    fig.suptitle(title, fontsize=16)

    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            # 假设是 RGB 图像，数值范围 0-1 或 0-255，自动适配
            if img.max() > 1.0:
                img = img / 255.0
            ax.imshow(img)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(num, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"可视化已保存至: {save_path}")
    if show:
        pass  # plt.show()
    else:
        plt.close(fig)


# ------------------ 仅做可视化的评估函数 ------------------
def eval_one_epoch_visualization(
    model,
    dataloader,
    model_type: List[str],
    device: torch.device,
    num_samples: int = 10,
    max_batches: Optional[int] = None,
    bc = None,  # BinConverter for coordinates_gs
) -> List[np.ndarray]:
    """
    仅进行前向推理，收集原图、标签和网络输出，生成可视化图像列表。

    Args:
        model: 神经网络模型
        dataloader: 数据加载器
        model_type: 模型输出类型列表，例如 ['keypoints_gs', 'coordinates']
        device: 设备
        num_samples: 最多收集多少张可视化图像（每个 batch 可能包含多张图）
        max_batches: 最多处理多少个 batch（None 表示全部）

    Returns:
        images_list: numpy 数组列表，每个元素是一张 RGB 图像，可用于 visualize_image_list
    """
    model.eval()
    activate = torch.nn.Sigmoid()

    collected_images = []   # 存储最终要可视化的图像

    batch_idx = 0
    with torch.no_grad():
        for samples, targets in tqdm(dataloader, desc="Visualizing"):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_idx += 1

            # 原始数据准备（与原始代码类似，但不再计算位姿）
            target_list = []
            for i in range(samples.shape[0]):
                target_list.append({k: targets[k][i].to(device) for k in targets.keys()})

            org_imgs_list = []
            gt_bboxes_list = []      # 存储每个样本的裁剪边界框
            gt_keypoints_list = []   # 存储每个样本的真实关键点（像素坐标）
            imageshapes_list = []    # 存储裁剪后的宽高

            for img, target in zip(samples, target_list):
                gtbbox = torch.round(target["boxes"].squeeze(0)).cpu().numpy().astype(int)
                gt_bboxes_list.append(gtbbox)

                org_imgs_list.append(img.to(device))

                kp_global = target['keypoints'].cpu().numpy()  # (N, 3) already in crop space
                gt_keypoints_list.append(kp_global)

                imageshape = target["imageshape"]
                print('kp_global:',kp_global)
                # print(imageshape)
                # print(gtbbox)
                imageshapes_list.append(imageshape)

            # 前向推理
            with torch.amp.autocast('cuda'):
                inputs = torch.stack(org_imgs_list).to(device)
                outputs = model(inputs)

            # 对 batch 中的每一张图像生成可视化
            for b in range(inputs.shape[0]):
                if len(collected_images) >= num_samples:
                    break

                img_resized = inputs[b].cpu().permute(2, 1, 0).numpy()
                img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

                gt_kp = gt_keypoints_list[b]
                print('gt_kp',gt_kp)
                gt_bbox = gt_bboxes_list[b]

                # 根据 model_type 生成不同的可视化子图
                if 'keypoints_gs' in model_type:
                    heatmap = outputs['keypoints_gs'][b]
                    pred_keypoints = heatmaps_to_keypoints(heatmap.unsqueeze(0), imageshapes_list[b])[0][0].cpu().numpy()

                    gt_global = gt_kp[0, :, :2].copy()
                    gt_global[:, 0] += gt_bbox[0]
                    gt_global[:, 1] += gt_bbox[1]
                    print('pred_keypoints',pred_keypoints)
                    pred_global = pred_keypoints[:, :2].copy()
                    pred_global[:, 0] += gt_bbox[0]
                    pred_global[:, 1] += gt_bbox[1]

                    canvas = np.ones((1200, 1920, 3), dtype=np.float32)
                    crop_img = img_resized.copy()
                    crop_h, crop_w = gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0]
                    paste_img = cv2.resize(np.clip(crop_img * 255, 0, 255).astype(np.uint8),
                                           (int(crop_w), int(crop_h)))
                    x1, y1 = int(gt_bbox[0]), int(gt_bbox[1])
                    cx1, cy1 = max(x1, 0), max(y1, 0)
                    cx2, cy2 = min(x1 + int(crop_w), 1920), min(y1 + int(crop_h), 1200)
                    px1, py1 = cx1 - x1, cy1 - y1
                    px2, py2 = px1 + (cx2 - cx1), py1 + (cy2 - cy1)
                    canvas[cy1:cy2, cx1:cx2] = paste_img[py1:py2, px1:px2].astype(np.float32) / 255.0

                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(canvas)
                    print(gt_global)
                    print(pred_global)
                    ax.scatter(gt_global[:, 0], gt_global[:, 1],
                               c='lime', marker='x', s=40, label='GT')
                    ax.scatter(pred_global[:, 0], pred_global[:, 1],
                               c='red', marker='o', s=20, label='Pred')
                    ax.legend()
                    ax.set_xlim(0, 1920)
                    ax.set_ylim(1200, 0)
                    ax.set_title('Keypoints (GS)')
                    ax.axis('off')
                    plt.tight_layout()

                    fig.canvas.draw()
                    img_arr = np.array(fig.canvas.buffer_rgba())[:, :, :3]
                    collected_images.append(img_arr)
                    plt.close(fig)

                if 'coordinates' in model_type:
                    # 坐标回归分支：预测的坐标图
                    coord_map = outputs['coordinates'][b]  # 假设形状 (C, H, W) 或 (H, W, C)
                    print(coord_map.shape)
                    mask_logits = outputs['mask'][b]       # 形状 (1, H, W) 或 (H, W)
                    # print(mask_logits.shape)
                    mask_bool = activate(mask_logits) > 0.5

                    # 将坐标图转为通道在最后一维的 numpy
                    if coord_map.dim() == 3 and coord_map.shape[0] == 3:
                        coord_np = coord_map.cpu().permute(2, 1, 0).numpy()
                    else:
                        coord_np = coord_map.cpu().numpy()  # 可能 (H,W,3) 或 (H,W,1)
                    # 归一化显示前三个通道
                    coord_vis = (coord_np - coord_np.min()) / (coord_np.max() - coord_np.min() + 1e-8)
                    if coord_vis.shape[-1] > 3:
                        coord_vis = coord_vis[..., :3]

                    # 创建子图：原图、坐标图、mask
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img_resized)
                    axes[0].set_title('Input')
                    axes[0].axis('off')

                    axes[1].imshow(coord_vis)
                    axes[1].set_title('Predicted Coordinates')
                    axes[1].axis('off')

                    axes[2].imshow(mask_to_display(mask_bool.cpu()), cmap='gray')
                    axes[2].set_title('Mask')
                    axes[2].axis('off')
                    # plt.show()
                    # plt.suptitle('Coordinates Regression')
                    # plt.tight_layout()
                    fig.canvas.draw()
                    img_arr = np.array(fig.canvas.buffer_rgba())[:, :, :3]
                    collected_images.append(img_arr)
                    plt.close(fig)

                if 'coordinates_gs' in model_type and bc is not None:
                    import torch.nn.functional as F
                    out = outputs['coordinates_gs'][b]  # (3*total_bins, H, W)
                    total_bins = bc.total_bins
                    out = out.view(3, total_bins, 256, 256).permute(2, 3, 0, 1)  # (256, 256, 3, total_bins)
                    probs = F.softmax(out, dim=-1)
                    coords_np = bc.bins_to_value(probs)  # (256, 256, 3) numpy
                    # print(coords_np.shape)
                    coords_vis = (coords_np - np.nanmin(coords_np)) / (np.nanmax(coords_np) - np.nanmin(coords_np) + 1e-8)

                    if bc.use_mask and 'mask' in outputs:
                        mask_bool = (activate(outputs['mask'][b]) > 0.5).cpu().numpy().squeeze()
                    else:
                        # mask derived from bins_to_value (NaN = background)
                        mask_bool = np.all(np.isfinite(coords_np), axis=-1)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img_resized)
                    axes[0].set_title('Input'); axes[0].axis('off')
                    axes[1].imshow(coords_np)
                    axes[1].set_title('Predicted Coords (GS bins)'); axes[1].axis('off')
                    axes[2].imshow(mask_bool, cmap='gray')
                    axes[2].set_title('Mask'); axes[2].axis('off')

                    fig.canvas.draw()
                    img_arr = np.array(fig.canvas.buffer_rgba())[:, :, :3]
                    collected_images.append(img_arr)
                    plt.close(fig)

            if len(collected_images) >= num_samples:
                break

    print(f"共收集 {len(collected_images)} 张可视化图像")
    return collected_images