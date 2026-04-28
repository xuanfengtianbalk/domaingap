# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel
# import segmentation_models_pytorch as smp
# from encoders.dinov2_wrapper import Dinov2Wrapper
# # ------------------------------------------------------------
# # 完整的分割模型：DINOv2 编码器 + U‑Net 解码器
# # ------------------------------------------------------------
# class DinoV2Unet(nn.Module):
#     def __init__(self,
#                  model_name='facebook/dinov2-small',
#                  num_classes=21,
#                  input_size=224,
#                  modulation_dim=None,
#                  freeze_encoder=True,
#                  checkpoint_path=None):
#         super().__init__()
#         self.input_size = input_size
#
#         # 1. 初始化编码器包装器
#         self.wrapper = Dinov2Wrapper(
#             model_name=model_name,
#             modulation_dim=modulation_dim,
#             freeze=freeze_encoder,
#             input_size=input_size
#         )
#
#         # 2. 本地权重加载（用户提供的逻辑）
#         if checkpoint_path and os.path.exists(checkpoint_path):
#             print(f"--- 正在从本地加载权重: {checkpoint_path} ---")
#             state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
#             if 'model' in state_dict:
#                 state_dict = state_dict['model']
#
#             model_keys = self.wrapper.model.state_dict().keys()
#             filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
#             unused_keys = [k for k in state_dict.keys() if k not in model_keys]
#             if unused_keys:
#                 print(f"--- [提示] 已自动过滤权重中不匹配的键: {unused_keys} ---")
#
#             self.wrapper.model.load_state_dict(filtered_dict, strict=False)
#             print("--- [成功] 权重已成功加载至 Backbone ---")
#
#         # 3. 构建解码器
#         n_blocks = len(self.wrapper.selected_layers)                 # 特征层数
#         encoder_channels = [self.wrapper.hidden_size] * n_blocks     # 每层通道数相同
#         decoder_channels = [256, 128, 64, 32][:n_blocks]             # 解码器通道，长度与层数一致
#
#         self.decoder = smp.unet.decoder.UnetDecoder(
#             encoder_channels=encoder_channels,
#             decoder_channels=decoder_channels,
#             n_blocks=n_blocks,
#             use_batchnorm=True,
#             center=False,
#             attention_type=None          # 可选用 'scse' 等
#         )
#
#         # 4. 分割头
#         self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
#
#     def forward(self, x):
#         # 编码器输出特征列表（从浅到深）
#         features = self.wrapper(x)           # list of tensors
#         # 解码器期望输入从深到浅，因此反转
#         features_rev = features[::-1]
#
#         # 解码器前向
#         decoder_out = self.decoder(*features_rev)
#
#         # 分割头
#         out = self.segmentation_head(decoder_out)
#
#         # 上采样到原图大小
#         out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
#         return out
#
#
# # ------------------------------------------------------------
# # 测试代码
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     # 配置参数
#     model_name = "facebook/dinov2-small"        # 可改为 base / large 等
#     num_classes = 21
#     input_size = 224
#     batch_size = 2
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 实例化模型（不加载本地权重，如需测试请传入 checkpoint_path）
#     model = DinoV2Unet(
#         model_name=model_name,
#         num_classes=num_classes,
#         input_size=input_size,
#         freeze_encoder=True,
#         checkpoint_path=None          # 例如 "/path/to/your/weights.pth"
#     ).to(device)
#
#     # 打印模型基本信息
#     print(model)
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"可训练参数数量: {trainable_params}")
#     print(f"总参数数量: {total_params}")
#
#     # 创建随机输入
#     dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
#
#     # 前向传播
#     with torch.no_grad():
#         output = model(dummy_input)
#
#     print(f"输出张量形状: {output.shape}")   # 期望 (batch_size, num_classes, input_size, input_size)
#
#     # 损失函数与优化器示例（您可在此替换为自己的实现）
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
#
#     print("测试完成！")
#
