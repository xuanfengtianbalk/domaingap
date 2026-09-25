# 微调方法对比轮总结（2026-09，DINOv3 预训练 + 新头协议）

## 1. 背景与协议

此前的 StableNet / LaSt-ViT 路线全部为 null 后，本轮改为**微调方法平级对比**：LoRA 本身作为方法之一，与其他微调方法（L2-SP、LP-FT、WiSE-FT、Model Soups）同协议竞争。

**协议**（与方法无关的固定项）：

- 起点：DINOv3 预训练 backbone（`dinov3_vitl16_pretrain_lvd1689m`）+ 全新 coordinates_DER 头
- 50 epoch，batch 16，augmix，seed 42
- lr：cosine 退火 **1e-4 → 2e-6**，warmup 1000 步
- 每个方法自带微调配置（与 LoRA 平级）

## 2. LoRA 实现修正（本轮前置工作）

旧实现（e24d72fb 所用）经审计**不符合 microsoft/LoRA 官方代码**：A 初始化 `randn×(0.1/in)`（无出处，为历史自定义）、无 α/r 缩放、qkv 全加（q,k,v 共享低秩对）、bias 冻结。

本轮按官方 `loralib/layers.py` 逐行移植修复：

| 项 | 旧实现 | 忠实版（本轮） |
|---|---|---|
| 位置 | fused qkv 全加 | MergedLinear，enable_lora=[True,False,True]（仅 q,v） |
| A 初始化 | randn×(0.1/in)≈1e-4 | kaiming_uniform(a=√5) |
| 缩放 | 无 | α/r（α=r=1） |
| bias | 冻结 | 可训练 |
| dropout | 无 | 参数化（默认 0） |

验证：初始 ΔW=0（前向≡冻结模型）、仅 q/v 有增量、参数计数精确匹配。历史产物（e24d72fb 及全部旧 runs）保留，标注为"旧 LoRA 变体"。

## 3. 方法实现依据（全部对照原作者代码）

| 方法 | 依据 | 关键配置 |
|---|---|---|
| LoRA | microsoft/LoRA `layers.py` | r=1, α=1, q,v |
| L2-SP | holyseven/TransferLearningClassification `network_base.py` mode 1 + `train.sh` | α∈{0.1, 0.01}, β=0.01, SGD momentum 0.9, wd=0, 仅 'weights' |
| LP-FT | 论文协议（ICLR22 Kumar et al.，无公开 repo）：LP 冻结训头 → 全量；ImageNet 各半 epoch → 25+25 | 阶段1 冻结 encoder 训头（AdamW），阶段2 全量联合 |
| WiSE-FT | mlfoundations/wise-ft `_merge`：θ=(1−α)θ₀+αθ₁ | 落到 LoRA 有效权重 = W + α·BA（scale lora_B），α∈{0.3,0.5,0.7}，零训练 |
| Model Soups | mlfoundations/model-soups uniform soup：state_dict 等权平均 | 成员 {L2-SP 0.1, L2-SP 0.01, LP-FT} |

## 4. 结果总表（angle mean±std / dist mean）

| Run | sunlamp | lightbox | val |
|---|---|---|---|
| 旧 LoRA 锚点 e24d72fb（50ep 固定 1e-4） | 4.47±9.88 / 0.1139 | 3.65±12.04 / 0.0781 | 0.87±2.08 / 0.0248 |
| LoRA 忠实锚点（50ep 退火） | 4.86±11.06 / 0.1266 | 3.74±12.02 / 0.0802 | 0.74±2.12 / 0.0210 |
| LP-FT 阶段1（仅训头 25ep） | 7.80±17.55 / 0.1947 | 8.11±22.92 / 0.1472 | 1.37±3.52 / 0.0429 |
| **LP-FT 最终（25+25）** | **4.22±9.49 / 0.1070** | **2.89±10.50 / 0.0609** | **0.46±0.62 / 0.0131** |
| L2-SP α=0.01 | 17.17±26.55 / 0.3529 | 12.38±22.86 / 0.2457 | 3.96±6.74 / 0.0762 |
| L2-SP α=0.1 | 24.09±31.77 / 0.4875 | 21.05±32.06 / 0.3961 | 6.64±10.34 / 0.1288 |
| WiSE-FT α=0.3 | 26.70±40.42 / 0.5128 | 20.88±38.31 / 0.4224 | 3.10±6.99 / 0.0825 |
| WiSE-FT α=0.5 | 9.94±20.81 / 0.2360 | 9.19±24.11 / 0.1816 | 1.86±3.85 / 0.0546 |
| WiSE-FT α=0.7 | 5.03±12.36 / 0.1535 | 4.86±15.99 / 0.1102 | 1.18±2.55 / 0.0462 |
| Soup 均匀平均（3 模型） | 105.63±40.61 / 5.8366 | 104.96±40.61 / 5.8483 | 103.45±41.47 / 5.9320 |

## 5. 判定

### 5.1 LP-FT：唯一有效，三指标全超锚点

- 两阶段（25ep 冻结训头 → 25ep 全量联合）在退火协议下全面优于 LoRA 锚点：val **0.46±0.62**（项目历史最佳）、lightbox **2.89**（历史最佳）、sunlamp 4.22
- 机制解读：先训头消除"随机头与特征联合漂移"的耦合（论文核心论点），再全量微调——与论文预测一致
- 阶段1 单独（7.80/8.11/1.37）证明"仅训头不够"，阶段2 的联合微调是收益来源

### 5.2 LoRA 忠实修正：无增益

忠实版（4.86/3.74/0.74）与旧变体（4.47/3.65/0.87）互有胜负——修正实现没有带来突破。注意此处同时改变了实现与 lr 调度两个变量，无法单独归因。与历史观察"rank 变化几乎无影响"一致：瓶颈不在 LoRA 实现细节。

### 5.3 L2-SP：双崩（协议不相容，非方法无效）

- α=0.01 → 17.17°，α=0.1 → 24.09°，α 越大越崩
- 原因：官方 L2-SP 协议 SGD lr=**0.01**（9000 iter + 2/3 处 ×0.1）；本协议退火 1e-4→2e-6，lr 差 100 倍。SGD 无逐参数自适应，1e-4 训全新头本身不足，β=0.01 又持续把头拉向 0 → 任务拟合失败（终 loss -2.25/-1.96 vs 锚点 -3.21）
- 结论：**官方 L2-SP 协议与本轮 lr 退火约束不相容**；在官方 lr 档位下未测

### 5.4 WiSE-FT：单调恢复但不超过锚点

- α=0.3 → 26.70°、0.5 → 9.94°、0.7 → 5.03°，随 α 单调回升到接近锚点（4.86）
- 解释：锚点的头是与**完整 LoRA 增量**联合训练的；缩掉增量后头与特征错配。且回归头没有 zero-shot 对应物，"zero-shot 侧"只有 encoder——插值曲线被头-增量错配主导
- 附带发现：α=0.3 的崩坏反证 **LoRA 增量对位姿拟合是真贡献**（非装饰）

### 5.5 Soup：失效（组成无效，非公平测试）

105.63° 崩坏。成员 3 个中 2 个（L2-SP×2）本身已崩，等权平均被拖垮。**这不是对 Model Soups 的公平检验**——前提（成员模型质量相近）不成立。若要公平测 Soups，需要多个健康的全量微调模型。

## 6. 已知局限

- 所有 run 单 seed（42）；LP-FT 的胜出需多 seed 验证
- L2-SP 仅在本协议 lr 档位下测试；官方 lr 档位（0.01）未测
- 本轮协议（50ep 退火）与旧协议（续训 10ep）不可直接对比；跨协议结论需谨慎
- Soup 组成失效（见 5.5）

## 7. 事件记录

- 首次启动的 WiSE-FT/Soup 评估因链式脚本中 `$COMMON` 的 `--mode train` 覆盖 `--mode evaluate`（argparse 取最后一个）而跑成 1-epoch 从零训练，产出 3 个垃圾 run（`e47c32f7`、`1b4e50d4`、`38a16b54`），已删除并以修正命令重跑（`_v2`/`_v3` 日志）
- Soup 首跑因 state_dict 中 Long 型张量无法 `mean()` 崩溃，已修复（Long/bool 张量取首值）

## 8. 复现命令

```bash
# LoRA 忠实锚点
python -u run.py --gpu 0 --mode train --train_script train --model_type coordinates_DER \
  --MODEL.BACKBONE_NAME dinov3_vitl16 --TRAIN.BATCH_SIZE 16 --TRAIN.AUG_TYPE augmix \
  --TRAIN.LR_SCHEDULE anneal --TRAIN.LR_WARMUP 1000 --seed 42 \
  --MODEL.PEFT.method lora --MODEL.PEFT.lora_rank 1 --MODEL.PEFT.lora_alpha 1 \
  --TRAIN.LR 1e-4 --TRAIN.LR_END 2e-6 --TRAIN.MAX_EPOCH 50

# L2-SP（α=0.1 / 0.01）
python -u run.py --gpu 0 --mode train --train_script train --model_type coordinates_DER \
  --MODEL.BACKBONE_NAME dinov3_vitl16 --TRAIN.BATCH_SIZE 16 --TRAIN.AUG_TYPE augmix \
  --TRAIN.LR_SCHEDULE anneal --TRAIN.LR_WARMUP 1000 --seed 42 \
  --train_backbone --MODEL.PEFT.method none --TRAIN.OPTIM SGD --TRAIN.WEIGHT_DECAY 0 \
  --TRAIN.L2SP_ALPHA 0.1 --TRAIN.L2SP_BETA 0.01 \
  --TRAIN.LR 1e-4 --TRAIN.LR_END 2e-6 --TRAIN.MAX_EPOCH 50

# LP-FT 阶段1（25ep 冻结训头）→ 阶段2（25ep 联合，resume 阶段1 uuid）
python -u run.py --gpu 1 ... --train_backbone --freeze_backbone --MODEL.PEFT.method none \
  --TRAIN.LR 1e-4 --TRAIN.LR_END 5.1e-5 --TRAIN.MAX_EPOCH 25
python -u run.py --gpu 1 ... --resume --resume_path <s1_uuid> --train_backbone --MODEL.PEFT.method none \
  --TRAIN.LR 5.1e-5 --TRAIN.LR_END 2e-6 --TRAIN.MAX_EPOCH 25

# WiSE-FT（零训练，α∈{0.3,0.5,0.7}）
python -u run.py --gpu 0 --model_type coordinates_DER --MODEL.BACKBONE_NAME dinov3_vitl16 \
  --MODEL.PEFT.method lora --MODEL.PEFT.lora_rank 1 --MODEL.PEFT.lora_alpha 1 --seed 42 \
  --mode evaluate --resume_path 25e06910-052f-4a3c-93b9-3612999915b7 --wise_alpha 0.5

# Soup（零训练）
python -u run.py --gpu 1 --model_type coordinates_DER --MODEL.BACKBONE_NAME dinov3_vitl16 \
  --no_peft --seed 42 --mode evaluate \
  --soup_paths 803126e6-2b60-425c-b719-688be43c3f66 46bf9c64-9e6c-40d0-a67b-3802b6c2ec9e 240f0e28-f3b8-4ef1-a4a8-095040e764cd
```

---

# 批 2（固定 lr=1e-4 协议）

## 9. 协议变更

按指示：后续所有 run **不再退火**，lr 固定 1e-4（warmup 1000 步后恒定）。同时 LP-FT 在固定 lr 下重跑一遍。

## 10. 批 2 结果总表

| Run | sunlamp | lightbox | val |
|---|---|---|---|
| **LLRD decay=0.9** | **4.29** | **2.87** | **0.77±0.68** |
| LN tuning | 6.62 | 4.68 | 0.96 |
| DoRA r=1 | 6.05 | 4.25 | 0.97 |
| LP-FT 重跑 s1（仅训头，固定 1e-4） | 8.61 | 8.10 | 1.52 |
| LP-FT 重跑 s2（25+25，固定 1e-4） | 12.94 | 5.11 | 0.87 |
| 参照：LP-FT 退火版 | 4.22 | 2.89 | 0.46 |
| 参照：LoRA 忠实锚点 | 4.86 | 3.74 | 0.74 |

## 11. 批 2 判定

### 11.1 LLRD：第二个有效方法

- 逐层 lr 衰减（decay=0.9，26 组：头/decoder 1e-4，顶层 block 1e-4，逐层 ×0.9，底层 8.9e-6）达到 LP-FT 级别：lightbox 2.87（并列历史最佳）、sunlamp 4.29、val 0.77
- 单阶段、无冻结切换、实现简单（参数分组）——与 LP-FT 并列为本项目唯二有效方法
- 机制与全项目经验自洽："encoder 必须动，但要有节制"——LLRD 用逐层 lr 梯度实现了这个节制

### 11.2 LP-FT 固定 lr 重跑崩了：退火是 LP-FT 的必要组成部分

- 固定 1e-4 版 sunlamp 12.94 vs 退火版 4.22——联合阶段在恒定 1e-4 下训过头（终 loss -2.91 vs 退火版 -3.64）
- 结论：**1e-4→2e-6 cosine 退火不是多余的协议冗余，是 LP-FT 生效的关键**

### 11.3 LN tuning 无效（但优于纯冻结）

- 6.62/4.68/0.96：差于 LoRA 锚点（4.86/3.74/0.74）
- 但优于"完全冻结仅训头"（8.61/8.10/1.52 或 7.80/8.11/1.37）——backbone 的 LN 层提供了少量有效适配
- 结论：仅 LN 的 backbone 适配不足；LLRD/LoRA/LP-FT 的 backbone 参与度都高于 LN

### 11.4 DoRA 无效：确认 A 类"结构无感"假设

- 6.05/4.25/0.97，差于 vanilla LoRA 锚点（4.86/3.74/0.74）
- 与"rank 无感、忠实修正无增益"一致：低秩家族的结构改进（幅值/方向分解）在本任务同样无感
- 该假设至此获得 3 条独立证据

## 12. 两批合并后的最终结论

```
有效（按效果排序）:
  1. LP-FT 两阶段（退火协议必需）  4.22 / 2.89 / 0.46
  2. LLRD 逐层 lr 衰减（固定 1e-4） 4.29 / 2.87 / 0.77

无效:
  LoRA（忠实/旧变体、DoRA）、L2-SP（lr 协议不相容）、WiSE-FT（无 zero-shot 头）、
  Soup（成员失效）、LN tuning（backbone 适配不足）

跨方法规律:
  - "encoder 必须动，但要有节制"是全部有效方法的共同特征
    （LP-FT: 头先行+退火联合; LLRD: 浅层近冻结/深层满速）
  - "节制"的实现方式有两种且都有效：时间维度（退火）+ 空间维度（逐层衰减）
  - A 类结构差异（低秩家族）三次证伪：rank 无感、忠实修正无增益、DoRA 无增益
```

## 13. 批 2 事件记录

- DoRA 首跑因 `effective_weight()` inplace 修改计算图张量崩溃（backward 报 version mismatch）；改为 `torch.cat` 无 inplace 组装后修复（`5e319db`）
- LN tuning 首跑误冻结全部参数（含随机初始化的 DER 头，loss 卡 11.76 无法学习）；修正为"backbone 冻结仅 LN + 头/decoder 正常训练"，垃圾 run 已删
- LN tuning 等待脚本 `pgrep -f "MAX_EPOCH 25"` 自匹配自身进程导致死等，改为直接启动

## 14. 批 2 复现命令

```bash
COMMON="--mode train --train_script train --model_type coordinates_DER --MODEL.BACKBONE_NAME dinov3_vitl16 --TRAIN.BATCH_SIZE 16 --TRAIN.AUG_TYPE augmix --seed 42 --TRAIN.LR_WARMUP 1000 --TRAIN.LR 1e-4 --TRAIN.MAX_EPOCH 50"

# LLRD
python -u run.py --gpu 1 --train_backbone --MODEL.PEFT.method none --TRAIN.LLRD_DECAY 0.9 $COMMON

# LN tuning
python -u run.py --gpu 0 --no_peft --ln_tune $COMMON

# DoRA
python -u run.py --gpu 1 --MODEL.PEFT.method lora --MODEL.PEFT.lora_rank 1 --MODEL.PEFT.lora_alpha 1 --MODEL.PEFT.use_dora true $COMMON

# LP-FT 固定 lr 重跑（25+25）
python -u run.py --gpu 0 --train_backbone --freeze_backbone --MODEL.PEFT.method none --TRAIN.MAX_EPOCH 25 $COMMON
python -u run.py --gpu 0 --resume --resume_path <s1_uuid> --train_backbone --MODEL.PEFT.method none --TRAIN.MAX_EPOCH 25 $COMMON
```
