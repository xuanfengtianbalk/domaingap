import sys
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')
import torch
from Create_Ushape_net import Create_Ushape_Net

model_config = {
    'ACTIVATE': None,
    'PEFT': {'method': 'lora', 'lora_rank': 1, 'lora_alpha': 1},
}
model = Create_Ushape_Net(
    Encode_Type='dinov3', Decode_Type=['coordinates_DER'],
    output_channel=11, BACKBONE_NAME='dinov3_vitl16', model_config=model_config)

from dinov3.eval.depth.models.peft import LoRAMergedLinear
layers = [m for m in model.modules() if isinstance(m, LoRAMergedLinear)]
print(f'LoRA layers: {len(layers)}')
m = layers[0]
print(f'A shape {tuple(m.lora_A.shape)} B shape {tuple(m.lora_B.shape)}')
print(f'scaling = {m.scaling} (alpha={m.lora_alpha}, r={m.rank})')
print(f'A std = {m.lora_A.std().item():.5f} (kaiming_uniform expected ~{((2/ (2 * m.in_features)) ** 0.5):.4f} std for uniform ~ {2*((6/m.in_features)**0.5)/ (2*3**0.5):.4f})')
print(f'B abs sum = {m.lora_B.abs().sum().item():.8f} (must be 0)')
print(f'merge_AB norm = {m.merge_AB().norm().item():.8f} (must be 0 at init)')
ind = m.lora_ind.view(3, -1)
print(f'enable_lora chunks (q,k,v): {ind[:, 0].tolist()}')
print(f'W requires_grad = {m.W.requires_grad} (must be False)')
print(f'bias requires_grad = {m.bias.requires_grad} (official: True)')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'trainable {trainable:,} / {total:,}')
backbone_trainable = sum(
    p.numel() for n, p in model.encoder.backbone.named_parameters() if p.requires_grad)
print(f'backbone trainable params: {backbone_trainable:,}')
expected = sum(
    (m.rank * 2 * m.in_features + m.rank * (m.out_features // 3) * 2)
    for m in layers)
print(f'expected lora params: {expected:,}')
assert m.lora_B.abs().sum().item() == 0.0
assert m.merge_AB().norm().item() == 0.0
assert m.W.requires_grad is False
assert m.bias.requires_grad is True
assert ind[:, 0].tolist() == [True, False, True]
print('ALL CHECKS PASSED')
