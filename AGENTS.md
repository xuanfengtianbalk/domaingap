# Project: Domain Gap Analysis — DINOv3 Pose Estimation for SPEED+

## 项目概况
卫星姿态估计，基于 DINOv3 backbone，SPEED+ 数据集。三种模型类型：coordinates / keypoints_gs / coordinates_gs。

## 环境
- 本地: `/opt/dl_workspace/algorithm/04-myself/domaingap`, conda env `dinov3`
- 远程: `ssh -p 301 sat001@192.10.84.217`, 密码 `Admin@9000`, 项目 `~/lk/domaingap`, conda env `dinov3`
- GitHub: `xuanfengtianbalk/domaingap`

## 当前数据结构（最近一次重大重构：commit 0aed1d9）
- `__getitem__` 流程：原图(1920×1200) → 算 kp → 算 bbox → crop+resize 到 256×256 → augment → Normalize → 返回
- `target_dict["boxes"]` = 原始 bbox `[x1, y1, x2, y2]`（用于 PnP）
- `target_dict["keypoints"]` = crop 空间相对坐标 `[x - x1, y - y1]`
- `target_dict["coors_gt"]` = world coords，NEAREST resize，shape (3, 256, 256)
- `target_dict["mask_gt"]` = uint8, shape (256, 256)
- `imageshape` = `[x2-x1, y2-y1]`（原始 crop 尺寸）
- 所有 `A.Resize` 已从 augment pipeline 中移除

## 关键文件
- `run.py`: 入口，parse_args, Criterion, build_model, build_dataset, main
- `train.py`: train_one_epoch（简化版，无 crop_tensor_image/interpolate）
- `eval.py`: valid_one_epoch + eval_one_epoch（PnP evaluation）
- `vis.py`: 可视化（已适配新数据格式）
- `utils_datasets/speedplus_utils_main/utils.py`: PyTorchSatellitePoseEstimationDataset + Camera + calculate_boxes_and_padded
- `utils_datasets/speedplus_utils_main/space_aug.py`: SpaceAugTransform（已移除 Resize）
- `utils_datasets/speedplus_utils_main/my_augmentation.py`: crop_tensor_image（当前已不再被调用）
- `Hyperpose_net/losses/coors_loss.py`: CoorsLoss（L1 + post_process denormalize）
- `Hyperpose_net/losses/kp_loss.py`: keypointrcnn_loss（DSNT heatmap）
- `Hyperpose_net/losses/bin_converter.py`: BinConverter（coordinates_gs 用）
- `aug_test/aug_eval.py`: 增强效果测试（test_kps + test_coords + test_coords_gs）
- `aug_test/aug_visualize.py`: 增强可视化（crop+resize 后再增强）
- `configs/cfg.yaml`: 主配置
- `Create_Ushape_net.py`: 模型构建（本地用绝对路径，远程用 `./dinov3_main`）
- `post_process.py`: PnP 解算

## 远程同步注意事项
- 不要同步 `Create_Ushape_net.py`（本地/远程路径不同）
- 远程 `Create_Ushape_net.py` 的 `REPO_DIR='./dinov3_main'`, `sys.path.append('./dinov3_main')`

## Code Style
- 不要添加注释，除非用户要求

## 当前远程训练状态（2026-05-14）
- Wave 1: coordinates(GPU0) + keypoints_gs(GPU1), LR=2e-4, 50 epochs, dinov3_vitb16
- Wave 2: coordinates_gs mean(GPU0) + coordinates_gs sum(GPU1), LR=2e-4, 50 epochs, dinov3_vitb16
- Screen: `w1-coordinates`, `w1-kpgs`, `wave2-launcher`（自动等 w1 完成后启动 w2）

## 最近的 Commit
- `5c1971a` - fix: auto-detect Camera.speed_root for local/remote compatibility
- `f68be7b` - fix: NEAREST for coors resize, update aug_eval+aug_visualize
- `0aed1d9` - refactor: crop+resize to 256x256 before augmentations

---

## 完整对话手递 (Handoff)

### 对话总结

#### 1. 数据管线重构 (核心改动)
- **问题**: `crop_tensor_image` 在 train.py/eval.py 中裁剪，bbox 可能超出图像边界，且所有 pipeline 都有 `A.Resize(300,480)` 导致 label distortion
- **解决方案**: 将 bbox 计算、crop、resize(256×256) 全部前置到 `PyTorchSatellitePoseEstimationDataset.__getitem__` 中
- **影响文件**:
  - `utils.py`: `__getitem__` 重写 — 先算 bbox → crop+resize(256×256) → 再 augment
  - `run.py`: 移除所有 `A.Resize(300,480)`
  - `space_aug.py`: 移除所有 pipeline 中的 `A.Resize(300,480)`
  - `train.py`: 去掉 `crop_tensor_image` + `interpolate`；imageshape 用原始 crop 尺寸
  - `eval.py`: 同上简化；keypoint PnP 转换用 `+gtbbox[0]/[1]`（kp 在 crop 相对坐标）
  - `vis.py`: 去掉 `crop_tensor_image` + `interpolate`
  - `aug_eval.py`: 适配新格式；新增 `test_coords_gs`
  - `aug_visualize.py`: crop+resize 后再做增强

#### 2. NaN Loss 修复
- `coors_resized` 用 `INTER_LINEAR` 插值时，`coors_crop` 中有 NaN 像素，双线性插值跨 NaN 边界产生 NaN
- 改为 `cv2.INTER_NEAREST`

#### 3. bbox 边界截断
- `calculate_boxes_and_padded` 中 `box = [x1-x_add, y1-y_add, x2+x_add, y2+y_add]` 后添加 clamp 到 `[0,1920]×[0,1200]`

#### 4. Camera.speed_root 自动检测
- 本地路径 `/opt/dl_workspace/datasets/speedplus/speedplus/`，远程路径 `datasets/speedplus/speedplus/`
- 改为 fallback 自动检测

#### 5. 四元数归一化
- `compute_pose_error` 中 `qvecs`, `qgt`, `q_` 在 arccos 前归一化，防止 norm > 1 导致 NaN

#### 6. 远程训练配置
- 4 组训练，2 GPU，分两轮：
  - Wave 1: coordinates(GPU0) + keypoints_gs(GPU1)
  - Wave 2: coordinates_gs mean(GPU0) + coordinates_gs sum(GPU1)
- 参数: LR=2e-4, BS=16, 50 epochs, dinov3_vitb16, AUG_TYPE=none
- Wave2 由 `wave2-launcher` screen 自动等待 w1 完成后启动

#### 7. 本地测试
- coordinates(epoch=1, LR=2e-4) loss 正常 ~1.1→0.87
- keypoints_gs 和 coordinates_gs 顺序训练中也正常

### 关键决策
- `SAMPLE_RANGE` 默认 `[-0.7, 0.7]`（卫星模型 1.15×1.13×0.45m）
- `fg_threshold=0.2` in `bins_to_value`
- `USE_MASK=False` for coordinates_gs
- 所有 aug 中灰度变换在 Normalize 之前
- `coors_resized` 用 `INTER_NEAREST`（不能用 LINEAR，因为有 NaN）
- `imageshapes` = 原始 crop 尺寸（不是 256×256）
