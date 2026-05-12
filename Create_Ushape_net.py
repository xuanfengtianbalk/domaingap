import os
import torch
import torch.nn as nn


# # 2. 解决路径索引问题 (如果有必要)
import sys
# # 确保 dinov2_wrapper 所在的父目录在路径里
sys.path.append('/opt/dl_workspace/algorithm/04-myself/domaingap/dinov3_main')

from dinov3.eval.depth.models import build_depther
from dinov3.eval.depth.models.encoder import BackboneLayersSet as bls

def Create_Ushape_Net(Encode_Type='dinov3',Decode_Type=["keypoints_gs"], output_channel=11,BACKBONE_NAME='dinov3_vits16', model_config=None):
    if Encode_Type == 'dinov3':
        REPO_DIR = '/opt/dl_workspace/algorithm/04-myself/domaingap/dinov3_main'
        WEIGHT_DICT = {'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                       'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
                       'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                       'dinov3_vitl16_sat': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                       'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                       'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
                       }

        WEIGHT = REPO_DIR + '/checkpoints/' + WEIGHT_DICT[BACKBONE_NAME]
        encoder = torch.hub.load(REPO_DIR, BACKBONE_NAME, source='local', weights=WEIGHT)
    else:
        raise ValueError("Unknown Encode_Type!")
    depther_model = build_depther(
        encoder,
        backbone_out_layers=(
            bls.FOUR_EVEN_INTERVALS  # One of BackboneLayersSet(Enum) or a list of indices e.g. [0, 1, 2, 3]
        ),
        n_output_channels=256,
        # use_backbone_norm=config.use_backbone_norm,
        use_batchnorm=True,
        use_cls_token=False,
        # adapt_to_patch_size="center_padding",
        head_type='dpt',
        autocast_dtype=torch.float32,
        # min_depth=config.min_depth,
        # max_depth=config.max_depth,
        # bins_strategy=config.bins_strategy,
        # norm_strategy=config.norm_strategy,
        Decode_Type=Decode_Type,
        output_channel = output_channel,
        activate = model_config['ACTIVATE'],
        train_backbone = model_config.get('TRAIN_BACKBONE', False),
        mixstyle = model_config.get('MIXSTYLE', False),
        mixstyle_p = model_config.get('MIXSTYLE_P', 0.5),
        mixstyle_alpha = model_config.get('MIXSTYLE_ALPHA', 0.1)
    )
    return depther_model