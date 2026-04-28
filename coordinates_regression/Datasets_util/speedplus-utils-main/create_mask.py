import os
os.environ["ONNXRUNTIME_DEVICE"] = "GPU"
import cv2
import numpy as np
from pathlib import Path
from rembg import remove
from PIL import Image
import argparse
import io


def extract_neural_radiance_fields_masks(input_dir, output_dir, batch_size=10):
    """
    为NeRF训练提取符合要求的掩码

    要求:
    - 必须是仅包含黑白像素的单通道图像
    - 必须与训练图像的分辨率相同
    - 黑色区域表示需要忽略的区域
    - 所有图像都必须带有掩码

    Args:
        input_dir: 输入图像目录
        output_dir: 输出掩码目录
        batch_size: 批处理大小（用于进度显示）
    """

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    print(f"在 {input_dir} 中找到 {len(image_files)} 个图像文件")

    if len(image_files) == 0:
        print("未找到任何图像文件，请检查输入目录路径")
        return

    # 处理每个图像
    successful_count = 0
    failed_count = 0

    for i, image_path in enumerate(image_files):
        try:
            # 显示进度
            if i % batch_size == 0:
                print(f"处理进度: {i}/{len(image_files)}")

            # 生成输出文件名
            output_filename = f"{image_path.stem}_mask.png"
            output_path = Path(output_dir) / output_filename

            # 如果掩码文件已存在，跳过
            if output_path.exists():
                print(f"跳过已存在的文件: {output_filename}")
                successful_count += 1
                continue

            # 获取原始图像尺寸
            original_image = Image.open(image_path)
            original_size = original_image.size  # (width, height)

            # 使用 rembg 移除背景
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)

            # 将结果转换为 PIL Image
            mask_image = Image.open(io.BytesIO(output_data))

            # 确保掩码与原始图像尺寸相同
            if mask_image.size != original_size:
                mask_image = mask_image.resize(original_size, Image.Resampling.LANCZOS)

            # 转换为单通道灰度图像
            if mask_image.mode == 'RGBA':
                # 提取alpha通道
                alpha_channel = mask_image.split()[-1]
                mask_array = np.array(alpha_channel)
            else:
                # 转换为灰度
                mask_array = np.array(mask_image.convert('L'))

            # 二值化处理：确保只有黑白像素
            # 使用阈值将图像转换为纯黑白（0和255）
            _, binary_mask = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)

            # 确保黑色区域(0)表示忽略，白色区域(255)表示保留
            # rembg通常生成的是前景为白色，背景为黑色，这符合要求

            # 保存为单通道PNG图像
            cv2.imwrite(str(output_path), binary_mask)

            # 验证保存的图像是否符合要求
            saved_mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            if saved_mask is not None:
                unique_vals = np.unique(saved_mask)
                if len(unique_vals) > 2 or (0 not in unique_vals and 255 not in unique_vals):
                    print(f"警告: {output_filename} 可能不是纯黑白图像")

            successful_count += 1
            print(f"成功生成符合NeRF要求的掩码: {output_filename}")

        except Exception as e:
            failed_count += 1
            print(f"处理 {image_path.name} 时出错: {str(e)}")

    print(f"\n处理完成!")
    print(f"成功: {successful_count}, 失败: {failed_count}")
    print(f"符合NeRF要求的掩码保存到: {output_dir}")


def verify_mask_requirements(mask_dir, image_dir):
    """
    验证生成的掩码是否符合NeRF要求

    Args:
        mask_dir: 掩码目录
        image_dir: 原始图像目录
    """

    mask_files = list(Path(mask_dir).glob("*_mask.png"))
    image_files = list(Path(image_dir).glob("*.*"))

    print(f"\n验证掩码文件...")
    print(f"找到 {len(mask_files)} 个掩码文件")
    print(f"找到 {len(image_files)} 个图像文件")

    # 检查每个图像是否有对应的掩码
    image_stems = {Path(f).stem for f in image_files}
    mask_stems = {Path(f).stem.replace('_mask', '') for f in mask_files}

    missing_masks = image_stems - mask_stems
    if missing_masks:
        print(f"警告: {len(missing_masks)} 个图像缺少掩码:")
        for stem in list(missing_masks)[:5]:  # 只显示前5个
            print(f"  - {stem}")
        if len(missing_masks) > 5:
            print(f"  ... 还有 {len(missing_masks) - 5} 个")
    else:
        print("✓ 所有图像都有对应的掩码")

    # 检查掩码格式
    valid_count = 0
    invalid_count = 0

    for mask_path in mask_files:
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # 检查是否为单通道
            if len(mask.shape) != 2:
                print(f"✗ {mask_path.name} 不是单通道图像")
                invalid_count += 1
                continue

            # 检查是否只包含黑白像素
            unique_vals = np.unique(mask)
            if not set(unique_vals).issubset({0, 255}):
                print(f"✗ {mask_path.name} 包含非黑白像素值: {unique_vals}")
                invalid_count += 1
                continue

            # 检查尺寸是否与对应图像匹配
            image_stem = mask_path.stem.replace('_mask', '')
            image_candidates = list(Path(image_dir).glob(f"{image_stem}.*"))
            if image_candidates:
                image_path = image_candidates[0]
                image = cv2.imread(str(image_path))
                if image is not None and mask.shape != image.shape[:2]:
                    print(f"✗ {mask_path.name} 尺寸不匹配: 掩码{mask.shape} vs 图像{image.shape[:2]}")
                    invalid_count += 1
                    continue

            valid_count += 1

        except Exception as e:
            print(f"✗ 验证 {mask_path.name} 时出错: {str(e)}")
            invalid_count += 1

    print(f"\n验证结果:")
    print(f"✓ 符合要求的掩码: {valid_count}")
    print(f"✗ 不符合要求的掩码: {invalid_count}")

    return invalid_count == 0


def create_compatible_masks_for_all_images(base_path, datasets=None):
    """
    为所有数据集创建符合NeRF要求的掩码

    Args:
        base_path: 基础路径
        datasets: 数据集列表，如果为None则使用默认列表
    """

    if datasets is None:
        datasets = ["synthetic", "sunlamp", "lightbox"]

    all_successful = True

    for dataset in datasets:
        input_dir = os.path.join(base_path, dataset, "images")
        output_dir = os.path.join(base_path, dataset, "masks")

        if os.path.exists(input_dir):
            print(f"\n{'=' * 50}")
            print(f"处理数据集: {dataset}")
            print(f"{'=' * 50}")

            extract_neural_radiance_fields_masks(input_dir, output_dir)

            # 验证生成的掩码
            if not verify_mask_requirements(output_dir, input_dir):
                all_successful = False
                print(f"警告: {dataset} 数据集的部分掩码不符合要求")
        else:
            print(f"跳过不存在的目录: {input_dir}")

    if all_successful:
        print("\n🎉 所有数据集的掩码都符合NeRF要求!")
    else:
        print("\n⚠️  部分掩码不符合要求，请检查上述警告")


def main():
    parser = argparse.ArgumentParser(description='为NeRF训练提取符合要求的掩码')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像目录路径')
    parser.add_argument('--output', '-o', type=str, default='masks',
                        help='输出掩码目录路径 (默认: masks)')
    parser.add_argument('--batch_size', '-b', type=int, default=10,
                        help='批处理大小，用于进度显示 (默认: 10)')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='处理完成后验证掩码是否符合要求')

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    batch_size = args.batch_size
    verify = args.verify

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"生成符合NeRF要求的掩码...")

    # 提取掩码
    extract_neural_radiance_fields_masks(input_dir, output_dir, batch_size)

    # 验证掩码
    if verify:
        verify_mask_requirements(output_dir, input_dir)


if __name__ == "__main__":
    # 直接使用示例
    base_path = "/opt/dl_workspace/algorithm/04-myself/domaingap/datasets/speedplus/speedplus/"  # 修改为您的实际路径

    # 方法1: 处理单个数据集
    # input_dir = os.path.join(base_path, "synthetic", "images")
    # output_dir = os.path.join(base_path, "synthetic", "masks")
    # extract_neural_radiance_fields_masks(input_dir, output_dir)

    # 方法2: 批量处理所有数据集
    create_compatible_masks_for_all_images(base_path)