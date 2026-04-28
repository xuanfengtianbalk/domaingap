import matplotlib.pyplot as plt
from typing import List, Union, Optional, Tuple


import torch
from utils_datasets.speedplus_utils_main.my_augmentation import crop_tensor_image
from tqdm import tqdm

import numpy as np
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
        plt.show()
    else:
        plt.close(fig)


# ------------------ 仅做可视化的评估函数 ------------------
def eval_one_epoch_visualization(
    model,
    dataloader,
    model_type: List[str],
    device: torch.device,
    num_samples: int = 10,          # 最多可视化多少张图（避免过多）
    max_batches: Optional[int] = None,  # 限制处理多少 batch
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
                # 获取地面真值边界框 (gtbbox)
                gtbbox = torch.round(target["boxes"].squeeze(0)).cpu().numpy().astype(int)
                gt_bboxes_list.append(gtbbox)

                # 裁剪原图（与原代码中 crop_tensor_image 一致）
                cropped = crop_tensor_image(img.clone(), torch.tensor(gtbbox).to(device)).float().to(device)
                org_imgs_list.append(cropped)

                # 获取真实关键点并转换到裁剪图像中的坐标（相对坐标）
                kp_global = target['keypoints'].cpu().numpy()  # (N, 3) or (N,2)
                # 计算相对裁剪框左上角的偏移
                kp_local = kp_global.copy()
                kp_local[:, 0] -= gtbbox[0]
                kp_local[:, 1] -= gtbbox[1]
                gt_keypoints_list.append(kp_local)

                # 裁剪区域的宽高
                imageshape = np.array([gtbbox[2]-gtbbox[0], gtbbox[3]-gtbbox[1]])
                imageshapes_list.append(imageshape)

            # 前向推理
            with torch.cuda.amp.autocast():
                # 将裁剪后的图像 resize 到 256x256（与原代码一致）
                inputs = torch.stack([
                    torch.nn.functional.interpolate(img_.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                    for img_ in org_imgs_list
                ]).to(device)
                outputs = model(inputs)

            # 对 batch 中的每一张图像生成可视化
            for b in range(inputs.shape[0]):
                if len(collected_images) >= num_samples:
                    break

                # 原始裁剪并 resize 后的图像 (256x256)
                img_resized = inputs[b].cpu().permute(1, 2, 0).numpy()
                img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

                gt_kp_local = gt_keypoints_list[b]   # 相对裁剪框的坐标
                # 关键点坐标需要缩放回 256x256 尺寸（因为原图被 resize 了）
                scale_x = 256.0 / imageshapes_list[b][0]  # 原裁剪宽度 -> 256
                scale_y = 256.0 / imageshapes_list[b][1]
                gt_kp_resized = gt_kp_local.copy()
                gt_kp_resized[:, 0] = gt_kp_local[:, 0] * scale_x
                gt_kp_resized[:, 1] = gt_kp_local[:, 1] * scale_y

                # 根据 model_type 生成不同的可视化子图
                if 'keypoints_gs' in model_type:
                    # 预测热图 -> 关键点
                    heatmap = outputs['keypoints_gs'][b]  # (C, H, W) 或 (H, W, C)? 根据实际情况
                    print(heatmap.shape)
                    # 假设热图形状 (C, 256, 256) 或 (256,256,C)，这里需要转成关键点坐标
                    # 使用 heatmaps_to_keypoints 函数（沿用原代码中的函数）
                    pred_keypoints = heatmaps_to_keypoints(heatmap.unsqueeze(0), imageshapes_list[b])[0][0]
                    # 同样缩放坐标到 256 尺寸以便叠加显示
                    pred_kp_resized = pred_keypoints.copy()
                    pred_kp_resized[:, 0] = pred_keypoints[:, 0] * scale_x
                    pred_kp_resized[:, 1] = pred_keypoints[:, 1] * scale_y

                    # 创建图像：原图 + GT关键点 + 预测关键点
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img_resized)
                    ax.scatter(gt_kp_resized[:, 0], gt_kp_resized[:, 1],
                               c='lime', marker='x', s=40, label='GT')
                    ax.scatter(pred_kp_resized[:, 0], pred_kp_resized[:, 1],
                               c='red', marker='o', s=20, label='Pred')
                    ax.legend()
                    ax.set_title('Keypoints (GS)')
                    ax.axis('off')
                    plt.tight_layout()

                    # 将 matplotlib figure 转为 numpy array
                    fig.canvas.draw()
                    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    collected_images.append(img_arr)
                    plt.close(fig)

                if 'coordinates' in model_type:
                    # 坐标回归分支：预测的坐标图
                    coord_map = outputs['coordinates'][b]  # 假设形状 (C, H, W) 或 (H, W, C)
                    print(coord_map.shape)
                    mask_logits = outputs['mask'][b]       # 形状 (1, H, W) 或 (H, W)
                    print(mask_logits.shape)
                    mask_bool = activate(mask_logits) > 0.5

                    # 将坐标图转为通道在最后一维的 numpy
                    if coord_map.dim() == 3 and coord_map.shape[0] == 3:
                        coord_np = coord_map.cpu().permute(1, 2, 0).numpy()
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

                    axes[2].imshow(mask_bool.cpu().numpy()[0], cmap='gray')
                    axes[2].set_title('Mask')
                    axes[2].axis('off')
                    plt.show()
                    # plt.suptitle('Coordinates Regression')
                    # plt.tight_layout()
                    # fig.canvas.draw()
                    # img_arr = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                    # img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    # collected_images.append(img_arr)
                    plt.close(fig)

            if len(collected_images) >= num_samples:
                break

    print(f"共收集 {len(collected_images)} 张可视化图像")
    return collected_images