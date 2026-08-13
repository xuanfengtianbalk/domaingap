"""Utilities for loading models, data, and extracting coordinate maps."""

import sys, os, torch, torch.nn.functional as F, numpy as np, json, glob

# Project root: analysis/../ = project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'dinov3_main'))

from run import load_config as load_train_config, build_model
from utils_datasets.speedplus_utils_main.utils import (
    PyTorchSatellitePoseEstimationDataset, points as body_points)
from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
from torch.utils.data import DataLoader, Subset, random_split


DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'speedplus', 'speedplus')
WORKINGDIR = os.path.join(PROJECT_ROOT, 'workingdir')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'cfg.yaml')


def load_model(uuid: str, model_type: str, backbone: str, device: str):
    """Load a trained model from workingdir/{uuid}/model_final.pth"""
    config = load_train_config(CONFIG_PATH)
    config['MODEL']['TYPE'] = [model_type]
    config['MODEL']['BACKBONE_NAME'] = backbone

    # read PEFT config from train_info if present
    pattern = os.path.join(WORKINGDIR, f'{uuid}*', 'train_info.json')
    matches = glob.glob(pattern)
    if matches:
        info = json.load(open(matches[0]))
        peft_cfg = info['model_info'].get('PEFT', {})
        if peft_cfg.get('method', 'none') != 'none':
            config['MODEL']['PEFT'] = peft_cfg

    model, bc = build_model(config)
    pattern = os.path.join(WORKINGDIR, f'{uuid}*', 'model_final.pth')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Checkpoint not found: {pattern}")
    ckpt = torch.load(matches[0], map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    if bc is not None:
        bc.to(device)
    return model, bc


def get_model_type_from_traininfo(uuid: str):
    """Read train_info.json and return model_type list."""
    pattern = os.path.join(WORKINGDIR, f'{uuid}*', 'train_info.json')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"train_info.json not found: {pattern}")
    info = json.load(open(matches[0]))
    return info['model_info']['TYPE'], info['model_info']['BACKBONE_NAME']


def build_dataloader(uuid: str, split: str, max_samples=None, batch_size=1):
    """Build a DataLoader for evaluation (no augmentation)."""
    ds = PyTorchSatellitePoseEstimationDataset(
        split=split, speed_root=DATASET_DIR, points=body_points,
        transform=SpaceAugTransform('none', styleaug_p=0))
    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def build_mlp_dataloader(uuid: str, aug_type: str, max_samples=None, mlp_batch=16):
    """Build train/val DataLoaders with SpaceAugTransform (like run.py build_dataset)."""
    ds = PyTorchSatellitePoseEstimationDataset(
        split='validation', speed_root=DATASET_DIR, points=body_points,
        transform=SpaceAugTransform(aug_type, styleaug_p=0, styleaug_alpha=0))
    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=mlp_batch, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=False,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=mlp_batch, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=False,
                            persistent_workers=True)
    return train_loader, val_loader, n_total, n_train


def extract_full(model, bc, sample_batch, device, model_type) -> dict:
    """Run model forward and extract all available prediction information.

    Returns a dict with keys depending on model_type:
      coords, epi_var, alea_var  (DER)
      coords, bin_probs, bin_centers, entropy  (gs)
      coords  (plain coordinates / others)
    """
    x = sample_batch.to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(x)

    result: dict = {}

    if model_type == 'coordinates':
        result['coords'] = out['coordinates'].cpu()
    elif model_type == 'coordinates_DER':
        result['coords'] = out['c'].cpu()
        # compute NIG uncertainty from logl/loga/logb
        logl = out['logl'].cpu().numpy()
        loga = out['loga'].cpu().numpy()
        logb = out['logb'].cpu().numpy()
        a = np.exp(loga) + 1.0 + 1e-6
        b = np.exp(logb) + 1e-6
        v = np.exp(logl) + 1e-6
        result['alea_var'] = b / (a - 1 + 1e-12)
        result['epi_var'] = b / ((a - 1 + 1e-12) * (v + 1e-12))
    elif model_type == 'coordinates_gs':
        logits = out['coordinates_gs'].cpu()
        B, _, H, W = logits.shape
        tb = bc.total_bins
        logits_reshaped = logits.view(B, 3, tb, H, W).permute(0, 3, 4, 1, 2)
        probs = F.softmax(logits_reshaped, dim=-1)
        coords_np = (probs.cpu().numpy() * bc.bin_centers.cpu().numpy()).sum(axis=-1)
        result['coords'] = torch.from_numpy(np.transpose(coords_np, (0, 3, 1, 2)))
        result['bin_probs'] = probs.cpu().numpy()
        result['bin_centers'] = bc.bin_centers.cpu().numpy()
        eps = 1e-12
        result['entropy'] = -(probs.cpu().numpy() * np.log(probs.cpu().numpy() + eps)).sum(axis=-1)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return result
