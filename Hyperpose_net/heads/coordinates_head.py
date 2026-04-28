from ..encoders.dinov2_wrapper import Dinov2Wrapper
from mmseg.registry import MODELS
from mmengine.model import BaseModule
@MODELS.register_module()
class Dinov2Backbone(BaseModule):
    def __init__(self,
                 model_name='dinov2_vits14',
                 modulation_dim=None,
                 freeze=True,
                 checkpoint_path=None,  # 增加：本地权重路径
                 out_indices=[0, 1, 2, 3],
                 patch_size=14,
                 init_cfg=None):
        super().__init__(init_cfg)

        # --- 核心改动：实例化你指定的 Dinov2Wrapper ---
        # 如果有本地权重，我们先不让它自动下载预训练权重
        pretrained = True if checkpoint_path is None else False

        self.wrapper = Dinov2Wrapper(
            model_name=model_name,
            modulation_dim=modulation_dim,
            freeze=freeze
        )

        # --- 增加：手动加载本地权重逻辑 ---
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"--- 正在从本地加载权重: {checkpoint_path} ---")
            # 注意：如果 wrapper 内部结构是 self.model，需要对应加载
            # state_dict = torch.load(checkpoint_path, map_location='cpu')
            # 兼容性处理：如果 state_dict 包含 'model' key 则解包
            # 1. 加载权重，加入 weights_only=False 消除警告
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # 2. 如果权重是官方打包格式，提取内部 model 字典
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # 3. 【关键控制点】获取当前模型结构中实际定义的参数名
            model_keys = self.wrapper.model.state_dict().keys()

            # 4. 【关键控制点】只保留模型中存在的 key，过滤掉多余的 mask_token 或 register_tokens
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}

            # 打印被过滤掉的信息，确保改动透明
            unused_keys = [k for k in state_dict.keys() if k not in model_keys]
            if unused_keys:
                print(f"--- [提示] 已自动过滤权重中不匹配的键: {unused_keys} ---")

            # 5. 执行加载。由于已经手动过滤，即使 strict=True 也会很安全
            # 但为了防止未来模型结构微调，建议使用 strict=False
            self.wrapper.model.load_state_dict(filtered_dict, strict=False)
            print("--- [成功] 权重已成功加载至 Backbone ---")

        self.out_indices = out_indices
        self.patch_size = patch_size
        self.embed_dim = self.wrapper.model.embed_dim

    def forward(self, x, mod=None):
        # 1. 调用你提供的 wrapper 获取 Token 序列 [B, L, C]
        # 注意：L = 1 (CLS) + (H/14 * W/14)
        tokens = self.wrapper(x, mod=mod)

        batch_size = x.shape[0]
        h_feat, w_feat = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        # 2. 移除 CLS Token 并重塑为特征图 [B, C, H, W]
        # tokens[:, 1:] 对应 x_norm_patchtokens
        patch_tokens = tokens[:, 1:, :].permute(0, 2, 1).reshape(batch_size, self.embed_dim, h_feat, w_feat)

        # 3. 因为你的 wrapper 只返回了最后一层特征
        # 为了适配 UperNet 等需要 4 层输入的 Head，我们将单层特征复制/返回
        # 这种做法在自监督模型微调中很常见
        outs = []
        for i in self.out_indices:
            outs.append(patch_tokens.contiguous())

        return tuple(outs)


def create_net(cfg):



    model_cfg = dict(
        type='EncoderDecoder',
        backbone=dict(
            type='Dinov2Backbone',
            model_name='dinov2_vits14',
            checkpoint_path='./backbone_checkpoint/dinov2_vits14_pretrain.pth',
            freeze=True),
        decode_head=dict(
            type=cfg["MODEL"]["DECODE_HEAD"],
            in_channels=[384, 384, 384, 384],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=cfg["DATASET"]["NUM_KEYPOINTS"],  # 假设 150 类
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )