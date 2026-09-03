import argparse
import yaml
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import warnings
warnings.filterwarnings('ignore', message='Error fetching version info')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchvision import transforms
from utils_datasets.speedplus_utils_main.utils import *
from train import train_one_epoch
from eval import valid_one_epoch, eval_one_epoch

# # from visuals import
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run training/visualization/testing based on config.")
#     parser.add_argument('--mode', default='train', choices=['train', 'visualization', 'sunlamp', 'lightbox'],
#                         help='Task to execute: train, visualization, or test')
#     parser.add_argument('--config', type=str, default='configs/cfg.yaml',
#                         help='Path to configuration file (YAML)')
#     parser.add_argument('--resume', type=bool, default=True,
#                         help='resume training')
#     return parser.parse_args()

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_dataset(config, split, aug_type='none', padded=True):
    """根据配置构建数据集，split可以是'train'/'val'/'test'"""
    dataset_config = config['DATASET']

    if dataset_config['NAME'] == 'speedplus':
        from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
        if split == 'train':
            trans = SpaceAugTransform(aug_type,
                styleaug_p=config['TRAIN'].get('STYLEAUG_P', 0.5),
                styleaug_alpha=config['TRAIN'].get('STYLEAUG_ALPHA', 0.3))
        elif aug_type != 'none':
            trans = SpaceAugTransform(aug_type,
                styleaug_p=0, styleaug_alpha=0)
        else:
            trans = SpaceAugTransform('none',
                styleaug_p=0, styleaug_alpha=0)

        dataset = PyTorchSatellitePoseEstimationDataset(split=split, speed_root=dataset_config['train_root_dir'], points=points,
                                                               transform=trans, padded=padded,
                                                               use_convex_hull=config['TRAIN'].get('USE_CONVEX_HULL', True))

    return dataset


def build_model(config):
    from Create_Ushape_net import Create_Ushape_Net
    """根据配置构建模型"""
    model_config = config['MODEL']

    #构建U形模型
    Encode_Type = model_config['BACKBONE']
    output_channel = model_config['OUTPUT_CHANNELS']
    BACKBONE_NAME = model_config['BACKBONE_NAME']
    assert config['MODEL']['OUTPUT_CHANNELS'] == config['DATASET']['NUM_KEYPOINTS'], f"OUTPUT_CHANNELS 不等于 NUM_KEYPOINTS"

    # Create bin converter for coordinates_gs
    bc = None
    if 'coordinates_gs' in model_config.get('TYPE', []) or 'coordinates_gs_EDL' in model_config.get('TYPE', []):
        from Hyperpose_net.losses.bin_converter import BinConverter
        bc_cfg = model_config.get('BIN_CONVERTER', {})
        bc = BinConverter(
            sample_range=bc_cfg.get('SAMPLE_RANGE', [-0.5, 0.5]),
            n_per_unit=bc_cfg.get('N_PER_UNIT', 30),
            sigma_factor=bc_cfg.get('SIGMA_FACTOR', 1.5),
            use_mask=bc_cfg.get('USE_MASK', True),
            loss_reduction=bc_cfg.get('LOSS_REDUCTION', 'mean'),
        )
        # 3 channels (x,y,z) per keypoint × bin count
        output_channel = 3 * bc.total_bins

    model = Create_Ushape_Net(Encode_Type=Encode_Type, Decode_Type=model_config['TYPE'], output_channel=output_channel, BACKBONE_NAME=BACKBONE_NAME,model_config=model_config)

    return model, bc

def build_optimizer(config, model):
    """根据配置构建优化器"""
    train_config = config['TRAIN']
    lr = float(train_config['LR'])
    optimizer_name = train_config.get('OPTIM', 'SGD')
    print('OPTIM: ',optimizer_name)
    param_groups = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=5e-2, betas=(0.9, 0.95))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer


EMBED_DIM_MAP = {
    'dinov3_vits16': 384,
    'dinov3_vitb14': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vitg14': 1536,
}


def get_backbone_embed_dim(model, backbone_name=None):
    """Resolve encoder feature channel dim (for Stable Learning feature net)."""
    if backbone_name in EMBED_DIM_MAP:
        return EMBED_DIM_MAP[backbone_name]
    m = model
    while m is not None:
        enc = getattr(m, 'encoder', None)
        if enc is not None:
            bb = getattr(enc, 'backbone', None)
            if bb is not None and hasattr(bb, 'embed_dim'):
                return bb.embed_dim
        m = getattr(m, 'module', None)
    return 384


def get_warmup_scheduler(optimizer: Optimizer, warmup_steps: int = 1000, warmup_start_factor: float = 0.0):
    """
    返回一个 LambdaLR 调度器，在前 warmup_steps 步内将学习率从
    warmup_start_factor * base_lr 线性增加到 base_lr。

    参数:
        optimizer: 优化器
        warmup_steps: 预热步数
        warmup_start_factor: 预热起始学习率相对于 base_lr 的比例（默认 0.0）
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性增加：从 start_factor 到 1.0
            return warmup_start_factor + (1.0 - warmup_start_factor) * (current_step / warmup_steps)
        return 1.0  # 预热结束后保持原学习率（可后续再使用其他调度器）

    return LambdaLR(optimizer, lr_lambda)

def set_seed(seed=42):
    # random.seed(seed)  # Python 内置随机
    np.random.seed(seed)  # NumPy 随机
    torch.manual_seed(seed)  # PyTorch CPU 随机
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机（当前 GPU）
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机（多卡）

    # 可选：设置 cuDNN 为确定性算法（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import uuid
def generate_folder_name():
    return str(uuid.uuid4())
from typing import Any, Dict, List

def setup_ddp():
    """Initialize DDP if launched with torchrun, returns (rank, world_size, local_rank)."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0
def _path_exists_and_is_leaf(config: Dict, path: str) -> bool:
    """
    检查点分隔路径是否存在于配置中，且最后一级不是字典（叶子节点）。
    返回 True 表示存在且可覆盖。
    """
    keys = path.split('.')
    d = config
    for i, k in enumerate(keys):
        if not isinstance(d, dict) or k not in d:
            return False  # 中间节点缺失
        if i == len(keys) - 1:
            # 最后一级，检查是否为字典
            return not isinstance(d[k], dict)
        d = d[k]
    return False  # 理论上不会执行到这里


def update_config_from_args(config: Dict, args_list: List[str]) -> Dict:
    """
    从命令行参数列表更新配置字典，仅支持完整路径覆盖（如 --TRAIN.LR 1e-4），
    且只更新配置中已存在的叶子节点。

    Args:
        config: 从YAML加载的配置字典。
        args_list: 命令行参数列表（例如 sys.argv[1:] 或 parse_known_args 返回的未知参数）。

    Returns:
        更新后的配置字典。
    """
    # 1. 解析 args_list 为覆盖字典 {key_path: value_str}
    overrides = {}
    i = 0
    n = len(args_list)
    while i < n:
        arg = args_list[i]
        if not arg.startswith('--'):
            i += 1
            continue

        # 处理 --key=value 形式
        if '=' in arg:
            key, value = arg[2:].split('=', 1)
            overrides[key] = value
            i += 1
            continue

        # 处理 --key value 形式
        key = arg[2:]
        if i + 1 < n and not args_list[i + 1].startswith('--'):
            value = args_list[i + 1]
            overrides[key] = value
            i += 2
        else:
            # 无值参数视为标志，但此处建议显式传递值
            print(f"警告：忽略无值的标志参数 {arg}，如需设置请使用 '--{key} True'")
            i += 1

    # 2. 类型转换函数（根据原值类型或自动检测）
    def convert_value(value_str: str, ref_value: Any = None) -> Any:
        # 有参考值时按参考类型转换（必须在自动检测之前）
        if ref_value is not None:
            if isinstance(ref_value, bool):
                return value_str.lower() in ('true', 'yes', '1')
            if isinstance(ref_value, int):
                return int(value_str)
            if isinstance(ref_value, float):
                return float(value_str)
            return value_str

        # 无参考值：自动检测
        if value_str.lower() in ('true', 'yes', '1'):
            return True
        if value_str.lower() in ('false', 'no', '0'):
            return False

        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str

    # 3. 应用覆盖，仅对已存在的叶子节点更新
    for key, value_str in overrides.items():
        if '.' not in key:
            print(f"警告：忽略参数 '{key}'，因为它不是完整路径（缺少点号），"
                  f"如需覆盖请使用类似 '--TRAIN.LR 1e-4' 的格式。")
            continue

        # 检查路径是否存在且为叶子节点
        if not _path_exists_and_is_leaf(config, key):
            print(f"警告：忽略参数 '{key}'，因为配置中不存在该完整路径或指向非叶子节点。")
            continue

        keys = key.split('.')
        # 导航到父节点
        d = config
        for k in keys[:-1]:
            d = d[k]   # 路径已确保存在，无需检查

        last_key = keys[-1]
        old_value = d[last_key]
        new_value = convert_value(value_str, old_value)

        d[last_key] = new_value

    return config


from Hyperpose_net.losses.kp_loss import KeypointRCNNLoss
from Hyperpose_net.losses.coors_loss import CoorsLoss, FocalLoss
class Criterion:
    def __init__(self, model_type, bc=None, evi_loss=None, evi_cls_loss=None):
        self.model_type = model_type
        self.los_fnc = {
            'keypoints_gs': KeypointRCNNLoss(sigma=3),
            'coordinates': CoorsLoss(),
            'mask': nn.BCEWithLogitsLoss(),
        }
        self.bc = bc  # BinConverter for coordinates_gs
        self.evi_loss = evi_loss
        self.evi_cls_loss = evi_cls_loss



    def forward(self, outputs,imageshapes,target_dict):
        """
        计算损失
        Args:
            pred: 模型预测值 (logits 或原始输出)
            target: 标签 (与 loss 函数要求一致)
        Returns:
            loss: 标量张量
        """
        total_loss = 0.0
        if 'keypoints_gs' in self.model_type:
            total_loss += self.los_fnc['keypoints_gs'](outputs['keypoints_gs'],imageshapes,target_dict['keypoints_gs'])
        if 'coordinates' in self.model_type:
            mask = (target_dict['mask'] > 0.5)
            total_loss += self.los_fnc['coordinates'](
                outputs['coordinates'].permute(0,2,3,1)[mask],
                target_dict['coordinates'].permute(0,2,3,1)[mask])
            total_loss += self.los_fnc['mask'](outputs['mask'].squeeze(1),target_dict['mask'])
        if 'coordinates_gs' in self.model_type:
            mask = (target_dict['mask'] > 0.5)
            out = outputs['coordinates_gs']
            total_bins = self.bc.total_bins
            B, _, H, W = out.shape
            out = out.view(B, 3, total_bins, H, W).permute(0, 3, 4, 1, 2)
            gt = target_dict['coordinates'].permute(0, 2, 3, 1)
            valid_out = out[mask]
            valid_gt = gt[mask]
            if valid_out.shape[0] > 0:
                total_loss += self.bc.loss_js(valid_out, valid_gt)
            invalid_out = out[~mask]
            if invalid_out.shape[0] > 0:
                probs = invalid_out.softmax(dim=-1)
                expected = (probs * self.bc.bin_centers).sum(dim=-1)
                total_loss += 0.1 * (expected ** 2).mean()
            if self.bc.use_mask:
                total_loss += self.los_fnc['mask'](outputs['mask'].squeeze(1),target_dict['mask'])
        if 'coordinates_DER' in self.model_type:
            mask = (target_dict['mask'] > 0.5)
            c    = outputs['c'].permute(0,2,3,1)[mask]
            logl = outputs['logl'].permute(0,2,3,1)[mask]
            loga = outputs['loga'].permute(0,2,3,1)[mask]
            logb = outputs['logb'].permute(0,2,3,1)[mask]
            gt   = target_dict['coordinates'].permute(0,2,3,1)[mask]
            if c.shape[0] > 0:
                total_loss += self.evi_loss(c, logl, loga, logb, gt)
            total_loss += self.los_fnc['mask'](outputs['mask'].squeeze(1), target_dict['mask'])
        if 'coordinates_gs_EDL' in self.model_type:
            mask = (target_dict['mask'] > 0.5)
            out = outputs['coordinates_gs']
            total_bins = self.bc.total_bins
            B, _, H, W = out.shape
            logits = out.view(B, 3, total_bins, H, W).permute(0, 3, 4, 1, 2)  # [B,H,W,3,K]
            gt = target_dict['coordinates'].permute(0, 2, 3, 1)                # [B,H,W,3]
            valid_logits = logits[mask]      # [N, 3, K]
            valid_gt = gt[mask]              # [N, 3]
            if valid_logits.shape[0] > 0:
                for c_dim in range(3):
                    target_probs = self.bc.value_to_bins(valid_gt[:, c_dim])
                    total_loss += self.evi_cls_loss(valid_logits[:, c_dim, :], target_probs)
            invalid_logits = logits[~mask]
            if invalid_logits.shape[0] > 0:
                probs = invalid_logits.softmax(dim=-1)
                expected = (probs * self.bc.bin_centers).sum(dim=-1)
                total_loss += 0.1 * (expected ** 2).mean()
        return total_loss

    # 也可以直接使用 __call__ 让实例像函数一样调用
    def __call__(self, outputs,imageshapes,target_dict):
        return self.forward(outputs,imageshapes,target_dict)


def parse_args():
    """示例参数解析器，使用 parse_known_args 捕获未知参数用于覆盖配置。"""
    parser = argparse.ArgumentParser(description="Run training/visualization/testing based on config.")
    parser.add_argument('--mode', default='train', choices=['train', 'visualization', 'sunlamp', 'lightbox', 'evaluate', 'validation'],
                        help='Task to execute: train, visualization, sunlamp, lightbox, or evaluate')
    parser.add_argument('--config', type=str, default='configs/cfg.yaml',
                        help='Path to configuration file (YAML)')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                    help='GPU device IDs to use (e.g., --gpu 0 1 2)')
    parser.add_argument('--model_type', nargs='+',
                        help='List of model types: coordinates, softass, keypoints_gs, keypoints_set')
    parser.add_argument('--train_script', type=str, default='train',
                        choices=['train', 'train_consistency', 'train_stable', 'train_l2sdg'],
                        help='Which training script to use')
    parser.add_argument('--train_backbone', action='store_true', default=False,
                        help='Unfreeze dinov3 backbone for training (default: frozen)')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Workingdir UUID path for evaluate mode')
    parser.add_argument('--padded', action='store_true', default=False,
                        help='Allow bbox to extend beyond image boundaries (pad with black)')
    # 使用 parse_known_args 以接受额外参数
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    rank, world_size, local_rank = setup_ddp()
    is_master = (rank == 0)

    # 解析命令行参数（已知和未知）
    args, unknown = parse_args()
    config = load_config(args.config)
    if args.model_type is not None:
        config['MODEL']['TYPE'] = args.model_type
    config['MODEL']['TRAIN_BACKBONE'] = args.train_backbone
    config['MODEL']['MIXSTYLE'] = config['TRAIN'].get('MIXSTYLE', False)
    config['MODEL']['MIXSTYLE_P'] = config['TRAIN'].get('MIXSTYLE_P', 0.5)
    config['MODEL']['MIXSTYLE_ALPHA'] = config['TRAIN'].get('MIXSTYLE_ALPHA', 0.1)
    config = update_config_from_args(config, unknown)
    set_seed(42 + rank)

    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        gpu_ids = args.gpu
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # 构建模型
    model, bc = build_model(config)

    model.to(device)
    if bc is not None:
        bc.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
        if is_master:
            print(f"Using DDP on {world_size} GPUs")
    elif len(args.gpu) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print(f"Using DataParallel on GPUs: {args.gpu}")

    if args.mode == 'evaluate':
        checkpoint = torch.load(f'workingdir/{args.resume_path}/model_final.pth', map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        end_path_name = f'workingdir/{args.resume_path}'
    elif args.mode != 'train' or args.resume is True:
        if args.resume_path:
            checkpoint = torch.load(f'workingdir/{args.resume_path}/model_final.pth', map_location='cuda:0')
        else:
            checkpoint = torch.load('./workingdir/c04d815b-814e-47e5-bcba-8519be4c92dc/model_final.pth', map_location='cuda:0')
        model.load_state_dict(checkpoint, strict=True)

    if args.mode == 'train':
        # build consistency layers before dataset (aug layers go into DataLoader workers)
        consistency_layers = None
        randconv_layers = None
        n_aug_branches = 0
        if args.train_script == 'train_consistency':
            from Hyperpose_net.losses.consistency import (RandConvLayer,
                   AugConsistencyLayer, ConsistencyAugmentor)
            n_branches = config['TRAIN'].get('RAND_CONV_N_BRANCHES', 3)
            ctype = config['TRAIN'].get('CONSISTENCY_TYPE', 'randconv')
            if ctype == 'randconv':
                randconv_layers = nn.ModuleList([RandConvLayer(
                    mix=config['TRAIN'].get('RAND_CONV_MIX', False),
                    p=config['TRAIN'].get('RAND_CONV_P', 0.5)).to(device)
                    for _ in range(n_branches)])
            elif ctype == 'aug':
                augs = nn.ModuleList([AugConsistencyLayer(
                    aug_type='augmix').train() for _ in range(n_branches)])
                n_aug_branches = n_branches
            elif ctype == 'patch_mask':
                from Hyperpose_net.losses.consistency import PatchMaskLayer
                augs = nn.ModuleList([PatchMaskLayer(
                    mask_ratio=config['TRAIN'].get('RAND_CONV_MASK_RATIO', 0.5)).train()
                    for _ in range(n_branches)])
                n_aug_branches = n_branches

        train_dataset = build_dataset(config, 'train', aug_type=config['TRAIN'].get('AUG_TYPE', 'none'), padded=args.padded)
        if n_aug_branches > 0:
            train_dataset.consistency_layers = augs
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           shuffle=True) if world_size > 1 else None
        train_loader = DataLoader(train_dataset, batch_size=config['TRAIN']['BATCH_SIZE'],
                                  shuffle=(train_sampler is None), num_workers=8,
                                  pin_memory=True, persistent_workers=True,
                                  sampler=train_sampler, multiprocessing_context='spawn')
        val_dataset = build_dataset(config, 'validation')
        val_loader = DataLoader(val_dataset, batch_size=config['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=8)
        val_loader_eval = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        sunlamp_dataset = build_dataset(config, 'sunlamp')
        sunlamp_loader = DataLoader(sunlamp_dataset, batch_size=1, shuffle=False, num_workers=1)

        lightbox_dataset = build_dataset(config, 'lightbox')
        lightbox_loader = DataLoader(lightbox_dataset, batch_size=1, shuffle=False, num_workers=1)

        end_path_name = config['TRAIN']['SAVE_DIR'] + "/" + generate_folder_name()
        if is_master:
            os.mkdir(end_path_name)
            print(f'Files Saving in Path {end_path_name}')

    elif args.mode == 'sunlamp':
        # 可视化可以使用任意split，比如'train'或'val'，根据需要
        vis_dataset = build_dataset(config, 'sunlamp')
        vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=1)
        # vis_dataset = build_dataset(config, 'validation')
        # vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=1)
    elif args.mode == 'lightbox':
        # 可视化可以使用任意split，比如'train'或'val'，根据需要
        vis_dataset = build_dataset(config, 'lightbox')
        vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=1)
    elif args.mode == 'validation':
        vis_dataset = build_dataset(config, 'validation')
        vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=1)
    elif args.mode == 'evaluate':
        pass  # dataset built inside evaluate handler
    else:  # test
        test_dataset = build_dataset(config, args.mode)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


    if args.mode == 'train':

        valid_losses=[]
        train_losses=[]

        evi_loss = None
        evi_cls_loss = None
        if 'coordinates_DER' in config['MODEL']['TYPE']:
            from Hyperpose_net.losses.evidential_loss import EvidentialLossSumOfSquares
            evi_cfg = config.get('EVIDENTIAL', {})
            evi_loss = EvidentialLossSumOfSquares(
                lamb=float(evi_cfg.get('lamb', 1e-3)),
                active=evi_cfg.get('active', 'exp'))
        if 'coordinates_gs_EDL' in config['MODEL']['TYPE']:
            from Hyperpose_net.losses.evidential_loss import EvidentialClassificationLoss
            evi_cls_cfg = config.get('EVIDENTIAL_CLS', {})
            evi_cls_loss = EvidentialClassificationLoss(
                lamb=float(evi_cls_cfg.get('lamb', 1e-3)))

        criterion = Criterion(model_type=config['MODEL']['TYPE'], bc=bc, evi_loss=evi_loss, evi_cls_loss=evi_cls_loss)
        optimizer = build_optimizer(config, model)
        scheduler = get_warmup_scheduler(optimizer, warmup_steps=1000)

        # ── Stable Learning components ──
        stable_state = None
        rff_layer = None
        feature_net = None
        if args.train_script == 'train_stable':
            from Hyperpose_net.losses.stable_learning import FeatureNet, RFFLayer, StableNetState
            sl_cfg = config.get('STABLE_LEARNING', {})
            n_z = int(sl_cfg.get('n_z', 512))
            rff_dim = int(sl_cfg.get('rff_dim', 1000))
            in_dim = get_backbone_embed_dim(model, config['MODEL']['BACKBONE_NAME'])
            feature_net = FeatureNet(in_dim, n_z).to(device)
            rff_layer = RFFLayer(n_z, rff_dim, float(sl_cfg.get('rff_sigma', 1.0))).to(device)
            stable_state = StableNetState(n_z, rff_dim, float(sl_cfg.get('beta', 0.9)), device=device)
            optimizer.add_param_group({'params': feature_net.parameters()})
            if is_master:
                print(f"[Stable Learning] in_dim={in_dim} n_z={n_z} rff_dim={rff_dim} enable={sl_cfg.get('enable', False)}")

        # ── L2SDG components ──
        wae = None
        wae_optimizer = None
        if args.train_script == 'train_l2sdg':
            from Hyperpose_net.losses.l2sdg import WAE
            l2_cfg = config.get('L2SDG', {})
            wae = WAE(latent_dim=int(l2_cfg.get('latent_dim', 128)),
                      epsilon=float(l2_cfg.get('epsilon', 8.0))).to(device)
            wae_optimizer = torch.optim.AdamW(wae.parameters(), lr=float(l2_cfg.get('wae_lr', 1e-3)))
            if is_master:
                print(f"[L2SDG] latent={l2_cfg.get('latent_dim', 128)} epsilon={l2_cfg.get('epsilon', 8.0)} enable={l2_cfg.get('enable', False)}")

        epochs = config['TRAIN']['MAX_EPOCH']
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if args.train_script == 'train_consistency':
                from train_consistency import train_one_epoch_randconv
                train_loss = train_one_epoch_randconv(model, train_loader, config['MODEL']['TYPE'], criterion, optimizer, scheduler, device, randconv_layers=randconv_layers, n_aug_branches=n_aug_branches, bc=bc, consistency_weight=config['TRAIN'].get('RAND_CONV_CONSISTENCY', 0.1), n_branches=n_branches)
            elif args.train_script == 'train_stable':
                from train_stable import train_one_epoch_stable
                sl_cfg = config.get('STABLE_LEARNING', {})
                train_loss = train_one_epoch_stable(model, train_loader, config['MODEL']['TYPE'], criterion, optimizer, scheduler, device,
                                                    rff_layer=rff_layer, feature_net=feature_net, state=stable_state,
                                                    weight_steps=int(sl_cfg.get('weight_steps', 3)),
                                                    weight_lr=float(sl_cfg.get('weight_lr', 0.1)),
                                                    enable=bool(sl_cfg.get('enable', False)))
            elif args.train_script == 'train_l2sdg':
                from train_l2sdg import train_one_epoch_l2sdg
                l2_cfg = config.get('L2SDG', {})
                train_loss = train_one_epoch_l2sdg(model, train_loader, config['MODEL']['TYPE'], criterion, optimizer, scheduler, device,
                                                   wae=wae, wae_optimizer=wae_optimizer,
                                                   beta=float(l2_cfg.get('beta', 0.5)),
                                                   lambda_norm=float(l2_cfg.get('lambda_norm', 1e-4)),
                                                   lambda_mmd=float(l2_cfg.get('lambda_mmd', 1e-3)),
                                                   perturb_steps=int(l2_cfg.get('perturb_steps', 1)),
                                                   enable=bool(l2_cfg.get('enable', False)))
            else:
                train_loss = train_one_epoch(model, train_loader, config['MODEL']['TYPE'], criterion, optimizer, scheduler, device)
            if is_master:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

            val_loss = valid_one_epoch(model, val_loader, config['MODEL']['TYPE'], criterion, device)
            if is_master:
                print(f"Validation Loss: {val_loss:.4f}")

            train_losses.append(train_loss.item())
            valid_losses.append(val_loss.item())

            # train_losses.append(0)
            # valid_losses.append(0)

            # torch.save(model.state_dict(), end_path_name + f'/e_{epoch}.pth')
        # 保存模型等操作
        if is_master:
            SAVE_DATA={'valid_losses':valid_losses,
                       'train_losses':train_losses,
                       "model_info": config['MODEL'],
                       "train_info": config['TRAIN']
                       }
            # 写入JSON文件
            with open(end_path_name+'/train_info.json', 'w', encoding='utf-8') as f:
                json.dump(SAVE_DATA, f)
            torch.save(model.state_dict(), end_path_name+'/model_final.pth')

            print('testing on sunlamp...')
            result_dict = eval_one_epoch(model, sunlamp_loader, config['MODEL']['TYPE'], criterion, Camera.K, device, bc=bc, evi_cls_threshold=config.get("EVIDENTIAL_CLS", {}).get("threshold", 0.0), evi_threshold=config.get("EVIDENTIAL", {}).get("threshold", 0.0))
            for name, data in result_dict.items():
                file_path = f"{end_path_name}/sunlamp_result_{name}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f)

            print('testing on lightbox...')
            result_dict = eval_one_epoch(model, lightbox_loader, config['MODEL']['TYPE'], criterion, Camera.K, device, bc=bc, evi_cls_threshold=config.get("EVIDENTIAL_CLS", {}).get("threshold", 0.0), evi_threshold=config.get("EVIDENTIAL", {}).get("threshold", 0.0))
            for name, data in result_dict.items():
                file_path = f"{end_path_name}/lightbox_result_{name}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f)

            print('testing on validation...')
            result_dict = eval_one_epoch(model, val_loader_eval, config['MODEL']['TYPE'], criterion, Camera.K, device, bc=bc, evi_cls_threshold=config.get("EVIDENTIAL_CLS", {}).get("threshold", 0.0), evi_threshold=config.get("EVIDENTIAL", {}).get("threshold", 0.0))
            for name, data in result_dict.items():
                file_path = f"{end_path_name}/validation_result_{name}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f)


    elif args.mode == 'evaluate':
        from Hyperpose_net.losses.kp_loss import KeypointRCNNLoss
        for mode in ['lightbox', 'sunlamp', 'validation']:
            dataset = build_dataset(config, mode)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            criterion = KeypointRCNNLoss(sigma=3)
            result_dict = eval_one_epoch(model, loader, config['MODEL']['TYPE'], criterion, Camera.K, device, bc=bc, evi_cls_threshold=config.get("EVIDENTIAL_CLS", {}).get("threshold", 0.0), evi_threshold=config.get("EVIDENTIAL", {}).get("threshold", 0.0))
            for name, data in result_dict.items():
                with open(f'{end_path_name}/{mode}_result_{name}.json', 'w') as f:
                    json.dump(data, f)
            print(f'{mode} eval saved to {end_path_name}/')

    elif args.mode == 'sunlamp' or args.mode == 'lightbox' or args.mode == 'validation':
        # 可视化示例：显示几个预测结果
        import matplotlib.pyplot as plt
        model.eval()
        with torch.no_grad():
            from vis import eval_one_epoch_visualization
            vis_images = eval_one_epoch_visualization(model, vis_loader, config['MODEL']['TYPE'], device, bc=bc, evi_cls_threshold=config.get('EVIDENTIAL_CLS', {}).get('threshold', 0.0))
            import os as _os; _os.makedirs('visuals', exist_ok=True)
            save_dir = f'visuals/{args.resume_path}'
            _os.makedirs(save_dir, exist_ok=True)
            from PIL import Image as _Image
            for i, img_arr in enumerate(vis_images):
                _Image.fromarray(img_arr).save(f'{save_dir}/vis_{args.mode}_{i:02d}.png')
            print(f'Saved {len(vis_images)} images to {save_dir}/')

    else:  # test
        from Hyperpose_net.losses.kp_loss import KeypointRCNNLoss
        criterion = KeypointRCNNLoss(sigma=3)
        result_dict=eval_one_epoch(model, test_loader, criterion, Camera.K, device)
        with open(f'./save_text/{args.mode}_result.json', 'w') as f:
            json.dump(result_dict, f)
        # print(f"Test Accuracy: {accuracy:.4f}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()