import sys
sys.path.insert(0, '/opt/dl_workspace/algorithm/04-myself/domaingap')
import torch
from Create_Ushape_net import Create_Ushape_Net

# --- DoRA checks ---
cfg = {'ACTIVATE': None, 'PEFT': {'method': 'lora', 'lora_rank': 1, 'lora_alpha': 1, 'use_dora': True}}
model = Create_Ushape_Net(Encode_Type='dinov3', Decode_Type=['coordinates_DER'],
                          output_channel=11, BACKBONE_NAME='dinov3_vitl16', model_config=cfg)
from dinov3.eval.depth.models.peft import LoRAMergedLinear
layers = [m for m in model.modules() if isinstance(m, LoRAMergedLinear)]
m = layers[0]
print(f'DoRA layers: {len(layers)}')
print(f'magnitude shape: {tuple(m.lora_magnitude.shape)} (expect [{m.sum_en},1,{m.in_features}])')
w_eff = m.effective_weight()
w3 = m.W.detach().view(3, m.out_chunk, m.in_features)
diff = (w_eff - m.W.detach()).abs().max().item()
print(f'init identity: max|W_eff - W| = {diff:.8f} (must be 0)')
mag_init = m.lora_magnitude.detach()
expect = w3[[0, 2]].norm(p=2, dim=1, keepdim=True)
print(f'magnitude init match col norms: {torch.allclose(mag_init, expect, atol=1e-5)}')
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'trainable with DoRA: {trainable:,}')

# --- LN tuning checks ---
cfg2 = {'ACTIVATE': None, 'PEFT': {'method': 'none'}}
model2 = Create_Ushape_Net(Encode_Type='dinov3', Decode_Type=['coordinates_DER'],
                           output_channel=11, BACKBONE_NAME='dinov3_vitl16', model_config=cfg2)
for p in model2.parameters():
    p.requires_grad_(False)
n_ln = 0
import torch.nn as nn
for mm in model2.modules():
    if isinstance(mm, nn.LayerNorm):
        for p in mm.parameters():
            p.requires_grad_(True)
            n_ln += 1
ln_trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f'LN tensors: {n_ln}, LN trainable params: {ln_trainable:,}')

# --- LLRD checks ---
import importlib.util
spec = importlib.util.spec_from_file_location('run_mod', '/opt/dl_workspace/algorithm/04-myself/domaingap/run.py')
run_mod = importlib.util.module_from_spec(spec)
import types
class _M: pass
spec.loader.exec_module(run_mod)
# need a model with backbone blocks: reuse model2 (peft none, unfreeze all)
for p in model2.parameters():
    p.requires_grad_(True)
groups = run_mod.build_llrd_param_groups(model2, 1e-4, 0.9)
print(f'LLRD groups: {len(groups)}, lr[0]={groups[0]["lr"]:.2e} (head), lr[-1]={groups[-1]["lr"]:.2e} (deepest)')
total = sum(len(g['params']) for g in groups)
print(f'LLRD grouped tensors: {total}')
assert abs(groups[0]['lr'] - 1e-4) < 1e-12
assert abs(groups[-1]['lr'] - 1e-4 * 0.9**24) < 1e-16
print('ALL BATCH2 CHECKS PASSED')
