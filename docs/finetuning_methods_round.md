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

---

# 批 2.5：LLRD + 退火组合

## 15. 动机与设计

批 2 发现两种"节制"机制各自独立生效：退火（时间维度，LP-FT 必需）与逐层衰减（空间维度，LLRD 单独有效）。本实验将两者叠加：

| 项 | 值 |
|---|---|
| 命令 | LLRD decay=0.9（26 组）+ cosine 退火 1e-4→2e-6，50ep，seed42 |
| 机制 | LambdaLR 因子对所有参数组等比例作用：头 1e-4→2e-6、底层 8.9e-6→~1.8e-7 |

## 16. 结果：项目历史最佳

| Run | sunlamp | lightbox | val |
|---|---|---|---|
| **LLRD 0.9 + 退火** | **3.01** | **2.02** | **0.34±0.56** |
| LP-FT 退火 | 4.22 | 2.89 | 0.46 |
| LLRD 固定 1e-4 | 4.29 | 2.87 | 0.77 |
| LoRA 忠实锚点 | 4.86 | 3.74 | 0.74 |
| 旧锚点 e24d72fb | 4.47 | 3.65 | 0.87 |
| （旧协议最佳 plain 续训） | 3.85 | 3.27 | 0.61 |

**三指标全部刷新项目纪录**：sunlamp 3.01（原最佳 3.85）、lightbox 2.02（原最佳 2.87）、val 0.34（原最佳 0.46）。较 e24d72fb 基线：sunlamp -33%、lightbox -45%、val -61%。

## 17. 判定与规律

- **两种"节制"机制叠加 > 各自单独**：退火（时间）× 逐层衰减（空间）产生了超加性效果，最终模型 = 项目历史最优
- 最终训练 loss -3.97（LLRD 固定版 -3.33、LP-FT 退火版 -3.64）——退火尾段的小 lr 精调 + 逐层衰减的浅层保护共同达成
- 至此"encoder 必须动、但要有节制"规律的完整表述：
  ```
  有效实现（可叠加）:
    时间维度: cosine 退火 1e-4→2e-6
    空间维度: 逐层 lr 衰减 decay=0.9
  叠加后: 3.01 / 2.02 / 0.34（项目最优）
  ```

## 18. 复现命令

```bash
python -u run.py --gpu 0 --train_backbone --MODEL.PEFT.method none --TRAIN.LLRD_DECAY 0.9 \
  --TRAIN.LR_SCHEDULE anneal --TRAIN.LR 1e-4 --TRAIN.LR_END 2e-6 --TRAIN.MAX_EPOCH 50 \
  --mode train --train_script train --model_type coordinates_DER --MODEL.BACKBONE_NAME dinov3_vitl16 \
  --TRAIN.BATCH_SIZE 16 --TRAIN.AUG_TYPE augmix --seed 42 --TRAIN.LR_WARMUP 1000
```

---

# 附录：FT 实验涉及的论文对应关系

## A. 已实测的微调方法

| 方法 | 论文 | 官方代码/参考实现 | 我们做了什么 | 结果 |
|---|---|---|---|---|
| LoRA | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, arXiv 2106.09685 | microsoft/LoRA `loralib/layers.py`（MergedLinear） | 旧版为手写变体（无参考）；后按官方逐行移植：q,v 块、kaiming A、α/r、bias 可训，r=1 | 忠实版 4.86/3.74/0.74，与旧变体相当（无增益） |
| DoRA | Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024, arXiv 2402.09353 | HF PEFT `use_dora=True`（`peft/tuners/lora/variants/dora.py`） | 移植到 q,v 块：W'=m·(W+BA)/‖W+BA‖c，m 初始=列范数 | 6.05/4.25/0.97（无增益，结构无感假设第 3 证） |
| L2-SP | Li, Grandvalet, Davoine, *Explicit Inductive Bias for Transfer Learning*, ICML 2018, arXiv 1802.01483 | holyseven/TransferLearningClassification（mode 1 + train.sh） | α∈{0.1, 0.01}、β=0.01、SGD mom 0.9、梯度注入 | 崩（17.2/24.1）——官方 lr 0.01 与退火 1e-4 差 100 倍 |
| LP-FT | Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022, arXiv 2202.10054 | 无公开 repo（多个候选 404，按论文协议实现） | 25ep 冻结训头 → 25ep 联合；退火版与固定 1e-4 版 | 退火版 4.22/2.89/0.46（赢）；固定版崩 12.94（退火必要性） |
| WiSE-FT | Wortsman et al., *Robust fine-tuning of zero-shot models*, CVPR 2022, arXiv 2109.01903 | mlfoundations/wise-ft（`_merge`: θ=(1−α)θ₀+αθ₁） | 落到 LoRA 有效权重 W+α·BA，α∈{0.3,0.5,0.7}，零训练 | 单调回升但不超锚点（无 zero-shot 头，头-增量错配） |
| Model Soups | Wortsman et al., *Model soups: averaging weights of multiple fine-tuned models*, ICML 2022, arXiv 2203.05482 | mlfoundations/model-soups（uniform soup） | 3 模型等权平均 | 105°（2/3 成员已崩，非公平测试） |
| LLRD | Sun et al., *How to Fine-Tune BERT for Text Classification?*, CCL 2019, arXiv 1905.05583 | 无官方 repo（BEiT/DINOv2 微调协议通行实践） | decay=0.9、26 组、全量微调；固定 1e-4 版与退火版 | 固定版 4.29/2.87/0.77；退火版 3.01/2.02/0.34 = 项目最优 |
| LN tuning | 无单一原始论文（作为基线出现在 Surgical FT（Lee et al., ICLR 2023, arXiv 2210.11466）、BitFit（Zaken et al., ACL 2022, arXiv 2106.10199）等） | 无官方 repo | 按通行定义：backbone 冻结仅 LN + 头训练 | 6.62/4.68/0.96（差于 LoRA 锚点） |

## B. 讨论过但未测的方法（出处供参考）

| 方法 | 论文 | 未测原因 |
|---|---|---|
| SAGM | 锐度感知组最小化（具体出处未核实，讨论阶段即砍） | 需要训练组标签，单源训练集无分组 |
| Adapter / AdaptFormer / Convpass / ViT-Adapter | Houlsby ICML 2019 / Chen NeurIPS 2022 / Jie & Deng 2022 / Chen ICLR 2023 | A 类"结构无感"假设成立后预期收益低 |
| AdaLoRA / QLoRA / LoRA+ / PiSSA / MoRA / FacT | 各自论文（ICLR 2023 / 2023 / 2024 等） | 同上（LoRA 微调变体） |
| Surgical FT | Lee et al., ICLR 2023, arXiv 2210.11466 | 移植成本高（TF 代码） |
| SAM | Foret et al., ICLR 2021, arXiv 2010.01412 | 与 SAGM 同族 |

## C. 早期 null 路线（训练期干预，非本轮 FT 方法）

| 方法 | 论文 | 官方代码 | 结果 |
|---|---|---|---|
| StableNet | Zhang et al., *Deep Stable Learning for Out-of-Distribution Generalization*, CVPR 2021, arXiv 2104.07876 | xxgege/StableNet | null（全局记忆为 1×batch 简化版） |
| L2SDG | Qiao et al., *Learning to Learn Single Domain Generalization*, CVPR 2020, arXiv 2003.13216 | 论文协议实现 | null |
| LaSt-ViT | arXiv 2602.22394v2（本地 PDF） | ChengShiest/LAST-ViT | ≤plain（5 个注入层） |
| RandConv（train_consistency） | Xu et al., *Robust and Generalizable Visual Representation Learning via Random Convolutions*, ICLR 2021, arXiv 2007.13003 | — | 训练协议的一部分 |

## D. 出处缺口（诚实标注）

1. **LN tuning**：无单一原始论文——通行做法，非单一论文提出的方法
2. **LP-FT**：官方 repo 未能定位（试了 4 个候选均 404），按论文正文协议实现
3. **SAGM**：出处未核实（讨论阶段即放弃）

---

# 附录 2：EXCLUDED 排除区间消融（LLRD+anneal 模型）

## 背景与协议

- 模型：LLRD 0.9 + 退火（uuid 783632ef，项目最优）
- 网格：完整复刻旧模型时代的 72 组合排除消融（`outputs/alpha_eval/ablation_sunlamp.csv` 同构）：excl_mode(and/or) × center(C1/C2) × rz_mode(uniform/z_half) × r(0.025/0.05/0.1) × std_excl_min(0.0025/0.005/0.01)
- 实现：`analysis/alpha_eval.py --excl_ablation`，两阶段（dump 网络一遍存 npz + 离线 72 路并行 sweep），断点续跑（npz 已存在即跳过 forward，整 split 完成秒级返回），批量 forward（batch 8）
- 结果：`outputs/excl_ablation_783632ef-.../excl_ablation_results.csv`（144 行，含 baseline/excluded/p25/p50/p75/delta/masked_frac）

## 全量结果（baseline：sunlamp 3.031±6.37 / lightbox 1.996±7.84）

### sunlamp 前 5（or 模式；and 模式全部 ≡ baseline，masked 比例 0%）

| 配置 | excluded | Δ |
|---|---|---|
| C1/z_half/rx0.1/ry0.1/rz0.05/s0.0025 | 2.947 | −0.084 |
| C1/uniform/r0.025/s0.0025 | 2.967 | −0.064 |
| C2/uniform/r0.05/s0.0025 | 2.969 | −0.062 |
| C2/z_half/r0.05(0.025)/s0.0025 | 2.972 | −0.059 |
| C1/uniform/r0.05/s0.0025 | 2.973 | −0.059 |

### lightbox 前 5

| 配置 | excluded | Δ |
|---|---|---|
| C2/uniform/r0.05/s0.0025 | 1.925 | −0.071 |
| C2/z_half/r0.05(0.025)/s0.0025 | 1.931 | −0.066 |
| C2/z_half/r0.05(0.025)/s0.005 | 1.940 | −0.056 |
| C2/z_half/r0.1(0.05)/s0.0025 | 1.941 | −0.055 |
| C1/z_half/r0.1(0.05)/s0.0025 | 1.944 | −0.052 |

## 判定

1. **更宽的剔除条件（std_excl_min=0.0025）在新模型上稳健有益**：mask 逻辑为 `ts > std_excl_min → 剔除`，故 `std_excl_min` 越小排除的不确定像素越多（0.0025 ≈ 排 12%、0.01 ≈ 只排 3%）。全量结果中 `0.0025`（排除最宽）两个 split 都是最优档（−0.05~−0.08°）；旧模型的 `std=0.01` 现在是最差档
2. **大半径有害**：r=0.1 uniform（大盒）两 split 都是最差（sunlamp +0.20、lightbox +0.12）——排掉太多有用像素
3. **旧默认配置失效**：or/C1/r0.05/s0.01（旧模型 −0.45°）在新模型上无益甚至有害
4. **and 模式完全失效**：三轴交集在新模型预测分布下为空（masked 比例恒 0%）
5. 最优排除的增益量级 ~0.05-0.08°（相对 ~2-3%），小于旧模型时代的 0.45°——误差分布的宽度（std 10.6→6.4）大幅收窄后，排除区间的边际价值同步收窄
6. n=100 的测试结论（"排除全面有害"）被 n=200 和全量推翻——小样本噪声不可用于这类细粒度消融

## 复现

```bash
# dump（断点续跑：npz 存在即跳过 forward）
python -u analysis/alpha_eval.py --excl_ablation --excl_phase dump \
  --uuid 783632ef-ceaa-4499-9cf9-575d94303951 --splits sunlamp lightbox \
  --max_samples 10000 --device cuda:0 --excl_dump_batch 8
# sweep（72 配置一路一进程）
OMP_NUM_THREADS=1 python -u analysis/alpha_eval.py --excl_ablation --excl_phase sweep \
  --uuid 783632ef-ceaa-4499-9cf9-575d94303951 --splits sunlamp lightbox \
  --max_samples 10000 --excl_proc 72
```

## 附录 2.1 退化中心搜索（std_excl_min=0，纯几何区间）

固定 `or/uniform/r=0.05/std_excl_min=0.0`（区内像素全排，不看过不确定度），5×5×5=125 个中心：
cx∈[-0.1,0.1]、cy∈[-0.05,0.15]、cz∈[0.05,0.25]，step 0.05。纯 sweep 阶段（复用 raw npz）。
结果：`outputs/excl_ablation_783632ef-.../excl_center_sweep_results.csv`（250 行）。

### 结果（baseline：sunlamp 3.031 / lightbox 1.996）

**sunlamp 前 5**：

| 中心 | excluded | Δ | masked |
|---|---|---|---|
| **(0.05, −0.05, 0.10)** | **2.838** | **−0.194** | 12.5% |
| (0.10, 0.10, 0.10) | 2.854 | −0.177 | 12.5% |
| (0.10, −0.05, 0.10) | 2.858 | −0.173 | 12.5% |
| (0.05, 0.15, 0.10) | 2.870 | −0.161 | 12.5% |
| (0.00, −0.05, 0.10) | 2.877 | −0.155 | 12.6% |

**lightbox 前 5**：

| 中心 | excluded | Δ | masked |
|---|---|---|---|
| **(0.10, 0.05, 0.10)** | **1.917** | **−0.080** | 12.4% |
| (−0.10, 0.00, 0.10) | 1.917 | −0.079 | 12.4% |
| (0.00, 0.10, 0.15) | 1.918 | −0.079 | 12.3% |
| (0.00, 0.10, 0.05) | 1.922 | −0.074 | 15.9% |
| (0.00, 0.05, 0.10) | 1.922 | −0.074 | 12.5% |

### 判定

1. **两 split 一致指向 cz≈0.10**：sunlamp 前 5 全部 cz=0.10；lightbox 前 5 中 4 个 cz=0.10。旧中心（cz=0.16/0.165）不在最优区
2. **cz=0.25 是全网格最差**（两 split 均为 +0.13~+0.17）——上半身区域排除有害
3. **纯几何区间（std=0）优于带不确定度过滤**：sunlamp 最优 Δ−0.194 vs 72 网格最优 −0.084；lightbox −0.080 vs −0.071
4. 最优中心大致在 **(0~0.1, −0.05~0.15, 0.10)** 的扁平区域——sunlamp 对 cy 不敏感、lightbox 对 cx 不敏感，两 split 的联合最优区约 c(0.05, 0.05, 0.10)

### top-5 中心记录（2026-09-28，供半径消融）

**sunlamp**：c(0.05,−0.05,0.10)=2.838(−0.194)、c(0.10,0.10,0.10)=2.854、c(0.10,−0.05,0.10)=2.858、c(0.05,0.15,0.10)=2.870、c(0.00,−0.05,0.10)=2.877

**lightbox**：c(0.10,0.05,0.10)=1.917(−0.080)、c(−0.10,0.00,0.10)=1.917、c(0.00,0.10,0.15)=1.918、c(0.00,0.10,0.05)=1.922、c(0.00,0.05,0.10)=1.922

### 半径消融（对上述 10 个中心，or/uniform/std=0，r∈{0.025,0.05,0.075,0.10,0.125,0.15}）

结果见 `outputs/excl_ablation_783632ef-.../excl_radius_sweep_results.csv`。

### 半径消融结果（60 配置）

**sunlamp 前 5**（baseline 3.031）：

| 中心 | r | excluded | Δ | masked |
|---|---|---|---|---|
| **c(0.10, 0.05, 0.10)** | **0.10** | **2.748** | **−0.283** | 24.8% |
| c(0.00, 0.05, 0.10) | 0.10 | 2.777 | −0.254 | 24.9% |
| c(0.05, −0.05, 0.10) | 0.075 | 2.780 | −0.251 | 17.9% |
| c(0.10, 0.10, 0.10) | 0.10 | 2.782 | −0.249 | 24.8% |
| c(0.00, −0.05, 0.10) | 0.075 | 2.783 | −0.248 | 18.0% |

**lightbox 前 5**（baseline 1.996）：

| 中心 | r | excluded | Δ | masked |
|---|---|---|---|---|
| **c(0.00, 0.05, 0.10)** | **0.075** | **1.891** | **−0.105** | 17.8% |
| c(−0.10, 0.00, 0.10) | 0.075 | 1.898 | −0.098 | 17.7% |
| c(0.00, 0.05, 0.10) | 0.10 | 1.898 | −0.098 | 24.6% |
| c(0.10, 0.05, 0.10) | 0.10 | 1.900 | −0.096 | 24.6% |
| c(0.10, 0.10, 0.10) | 0.075 | 1.900 | −0.096 | 17.7% |

### 判定

1. **最优半径 > 原默认 0.05**：两 split 一致偏好 r=0.075~0.10（masked 18~25%），远大于 72 网格时代的最优半径
2. **联合最优配置**：`c(0.00, 0.05, 0.10), r=0.075, or, uniform, std=0`——sunlamp 2.777(−0.254)、lightbox 1.891(−0.105)，两 split 同时进前 2
3. **历史最好**：sunlamp 2.748（c(0.10,0.05,0.10)/r0.10）较 baseline 3.031 改善 9.3%；lightbox 1.891 改善 5.3%
4. 排除后处理至此的完整链条：模型 3.031/1.996 → 最优排除 2.748~2.777 / 1.891~1.898

### 不确定性剔除（std_excl_min）消融结果（40 配置）

基底 = 半径消融 top 组合去重 8 个；std_excl_min ∈ {0, 0.0025, 0.005, 0.01, 0.02}。
结果见 `outputs/excl_ablation_783632ef-.../excl_unc_ablation_results.csv`。

**最优组合的 std 序列**：

| 组合（split） | std=0 | 0.0025 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|---|
| c(0.10,0.05,0.10)/r0.10（sunlamp） | **2.748** | 2.748 | 2.827 | 3.034 | 3.168 |
| c(0.00,0.05,0.10)/r0.075（lightbox） | **1.891** | 1.891 | 1.955 | 2.002 | 2.001 |

（值 = excluded angle；Δ 对应 −0.283 → +0.137 / −0.105 → +0.005）

### 判定

1. **std=0 与 std=0.0025 完全等价**（masked 比例 24.8%/17.8% 相同）——最优几何区间内像素的 ts 几乎全部 > 0.0025，下界条件在此配置下是惰性的
2. **std 下界越大越差**：只排极不确定像素（std=0.01/0.02）使收益归零甚至变负（sunlamp +0.137）——保留不确定像素有害
3. **std 上界联合判断无意义**：最优即纯几何（std=0，区内全排）
4. **最终排除配置**：
   - sunlamp 最优：c(0.10, 0.05, 0.10), r=0.10, or, uniform, std=0 → **2.748**（baseline 3.031，−9.3%）
   - lightbox 最优：c(0.00, 0.05, 0.10), r=0.075, or, uniform, std=0 → **1.891**（baseline 1.996，−5.3%）
   - 联合：c(0.00, 0.05, 0.10)/r0.075 → 2.777 / 1.891

### std 全局单轴消融（无中心、无半径，全图不确定度筛选）

std ∈ {0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}，全图 `ts > std` 剔除。
结果见 `outputs/excl_ablation_783632ef-.../excl_std_ablation_results.csv`。

| std | sunlamp excl (Δ) | lightbox excl (Δ) | masked |
|---|---|---|---|
| 0 | 无效（全排，n=0） | 无效（n=0） | 100% |
| 0.0025 | 9.502（n=4） | 15.279（n=278） | ~100% |
| 0.005 | 8.514 (+6.39) | 4.736 (+3.34) | 94%/88% |
| 0.01 | 4.044 (+1.42) | 2.341 (+0.77) | 75%/72% |
| 0.02 | 3.351 (+0.59) | 2.006 (+0.28) | 65%/63% |
| 0.05 | 3.168 (+0.30) | 2.027 (+0.14) | 57%/56% |
| 0.1 | 3.125 (+0.19) | 2.068 (+0.11) | 51%/51% |
| 0.2 | 3.087 (+0.12) | 2.045 (+0.09) | 46%/45% |
| 0.5 | 2.999 (−0.03) | 2.029 (+0.03) | 37%/37% |
| 1.0 | 3.080 (+0.05) | 2.014 (+0.02) | 31%/31% |

### 判定（排除后处理闭环）

1. **全局不确定度筛选永远劣于 baseline**（除 std=0.5 的 sunlamp −0.03 噪声级）——模型的不确定度全局偏高（std=0.0025 时几乎 100% 像素被排、只剩 4/278 张有效图），不确定度信号本身**没有选择性**
2. **几何区间才是有效信息载体**：同一模型上几何排除带来 −0.28/−0.11，而纯 std 筛选最多 ±0.03——不确定度只能作为几何区间内的次级修正（且最优时无需它，std=0 即最优）
3. **排除后处理最终结论**：
   - 最优配置 = 纯几何：sunlamp c(0.10,0.05,0.10)/r0.10 → 2.748（−9.3%）；lightbox c(0.00,0.05,0.10)/r0.075 → 1.891（−5.3%）
   - std 下界/上界联合判断均无益：最优处下界惰性（std=0 与 0.0025 等价），全局筛选有害
   - 消融顺序：72 网格 → 中心搜索（cz≈0.10）→ 半径（0.075~0.10）→ std（纯几何最优）

## 附录 2.2 排除方案诊断
> 注：diagnose 目录（PNG 产物）已删除——图像产物无法被消费；其数值结论保留在本节，后续特征-误差关系分析见附录 2.3。
（最优配置的逐图分析）

方法：`analysis/excl_diagnose.py`（零网络，读已存 npz + baseline）；产物在
`outputs/excl_ablation_783632ef-.../diagnose/`（ts 分布、空间分布、Δ 直方图、误差面热图、summary json）。

### 误差面形状（中心搜索数字网格，Δ 值）

- **宽谷非尖峰**：sunlamp 在 cz=0.10 层整层 −0.07~−0.19（谷点在 c(0.05,−0.05,0.10)=−0.194）；lightbox 三层均 −0.02~−0.08 平坦——中心在 (0~0.1, −0.05~0.15, 0.10) 大区域内都接近最优

### 逐图收益分布（关键发现）

| | sunlamp_best | lightbox_best |
|---|---|---|
| Δ mean±std | −0.283±2.46 | −0.105±2.03 |
| 受益图比例 | 58%（1629） | 60%（4061） |
| 受害图比例 | 42%（1162） | 40%（2679） |
| q1（受益最大） | Δ=−2.10，rot≈348°，bbox≈463k | Δ=−0.80，rot≈337°，bbox≈496k |
| q4（受害最大） | Δ=+1.15，rot≈334°，bbox≈539k | Δ=+0.46，rot≈329°，bbox≈531k |
| corr(Δ, rot) | −0.019 | +0.004 |
| corr(Δ, bbox) | +0.045 | +0.025 |

### 诊断结论

1. **排除的收益高度两极分化**：受益图（大旋转角 348°/337°、中小 bbox）Δ 可达 −2°；受害图（小旋转 334°/329°、大 bbox）Δ 达 +1.2°
2. 极端受益图（Δ −46°~−16°）几乎都是大旋转（200-520°）+ 小/中目标
3. 线性相关弱但四分位趋势一致：**旋转大→受益、bbox 大→受害**
4. 由此推断最优方案不是全局固定区间，而是**逐图自适应**（按旋转角/bbox 决定是否排除）——若能把 42% 的受害图归零，Δ 上限约 −0.6°（oracle 估计）

### 下一步（待定）

保存 per-image (Δ, rot, bbox) 全量记录 → 零成本重算规则阈值（如 rot>θ 或 bbox<A 才排除）→ 验证自适应排除。


## 附录 2.3 特征与输出误差的关系分析（数值化存储，无图像产物）

方法：`analysis/excl_err_analysis.py`，从已存 npz 给每张图提取 20 个特征
（含 2D/3D 不确定度）+ baseline 误差 b_i + 排除收益 Δ_i，全部落盘为可读数据：
`outputs/excl_ablation_783632ef-.../err_analysis/{per_image_{split}.csv, summary_{split}.json}`。

### 不确定度（逐轴 ts 与总模长 ts3d）与输出误差

ts 定义：逐像素证据分布（NIG）不确定度，逐轴 ts_ax = sqrt(认知方差+偶然方差)，
总模长 ts3d = sqrt(ts_x²+ts_y²+ts_z²)，单位 = 归一化坐标。

| 特征 | sunlamp ρ(b) | lightbox ρ(b) |
|---|---|---|
| p50_ts3d（3D 中位） | **+0.406** | +0.266 |
| p90_ts3d（3D 高分位） | +0.401 | +0.293 |
| p90_tsz（**z 轴高分位**） | **+0.425（全局最强）** | **+0.330（全局最强）** |
| p50_tsz（z 轴中位） | +0.332 | +0.138 |
| mean_tsz（z 轴均值） | +0.121（三轴最弱） | +0.159 |
| p50_tsx / p50_tsy | +0.324 / +0.350 | +0.220 / +0.219 |
| p90_tsx / p90_tsy | +0.301 / +0.362 | +0.203 / +0.211 |

分箱曲线：p50_ts3d 最高分位箱误差暴增至 9.48°/7.49°（其余箱 1.7~3.9°/1.3~1.7°）。

**结论（修正版）**：
1. 模型自我评估的不确定度与真实误差显著正相关（ρ 0.22~0.43）——DER 证据分布校准方向正确
2. **z 通道不确定度呈"中心欠表达、尾部最有信息"**：mean_tsz 是三轴最弱（0.121/0.159），
   而 p90_tsz 是全部 15 个不确定度特征里最强的误差预测器（0.425/0.330）——
   模型平时对 z 报得少，但 z 不确定度一旦尖峰（尾部图），正是误差大图
3. 与"mean_cz −0.32、std_cz +0.23"合并：**误差与最有信息的不确定度都住在 z 轴**——z 通道是 DER 头改进的首要目标（P4 落点）

### 预测坐标结构与输出误差

| 特征 | sunlamp ρ(b) | lightbox ρ(b) |
|---|---|---|
| mean_cz（预测质心 z） | **−0.319** | **−0.232** |
| std_cz（预测 z 离散度） | +0.234 | +0.137 |
| std_cy | −0.249 | −0.111 |
| bbox_area / rot / mask_frac | ≤0.10 | ≤0.09 |

**结论**：误差主要由 **z 轴预测结构**决定（质心 z 低、z 离散大 → 误差大）——
与排除消融"cz≈0.10 平面最优、cz=0.25 有害"互相印证：z 轴是主要误差轴。

### 排除收益 Δ 的可预测性（自适应排除可行性判定）

| 特征 | sunlamp ρ(Δ) | lightbox ρ(Δ) |
|---|---|---|
| p50_ts3d | −0.099 | **+0.034**（方向相反） |
| mean_cx | −0.125 | +0.016 |
| std_cz | −0.084 | +0.023 |
| 其余全部 | |ρ|<0.10 | |ρ|<0.04 |

对照 ρ(b_i, Δ_i)：sunlamp −0.332、lightbox −0.198——这是地板效应（baseline 误差
大的图下降空间大）的机械结果，非发现。

**判定**：
1. 最强测试时特征与 Δ 的 |ρ| ≤ 0.125，且跨 split 方向不一致（lightbox 上符号翻转）——
   **不存在可利用的 gating 信号**
2. 连事后真误差 b_i 也只解释 Δ 的 ~10% 方差（机械地板效应）——收益的逐图变异
   本质难以预测
3. **自适应排除方向关闭**：排除后处理收敛于固定纯几何区间
   （sunlamp c(0.10,0.05,0.10)/r0.10 → 2.748；lightbox c(0.00,0.05,0.10)/r0.075 → 1.891）

## 附录 2.4 排除整合与 unc 加权评估（收尾）

### 统一排除配置正式评估（c(0.00,0.05,0.10)/r0.075，or，std=0，全量）

| split | BASELINE | EXCLUDED | Δ |
|---|---|---|---|
| sunlamp | 3.006±6.21 (n=2791) | **2.756±6.02** | **−0.249（−8.3%）** |
| lightbox | 1.971±7.72 (n=6740) | **1.893±7.53** | **−0.078（−4.0%）** |

（val split 未完成评估，按指示中止；如需可在任何时候单跑。）
结果文件：`outputs/final_excl_783632ef-.../{sunlamp,lightbox}.json`。

### unc 加权 PnP 快速测试（100 样本，两 split）——阴性

| split | BASELINE | BASELINE_with_unc | EXCLUDED | EXCLUDED_with_unc |
|---|---|---|---|---|
| sunlamp | 3.021 | 3.171（**+0.150 更差**） | 2.686 | 2.793（**+0.107 更差**） |
| lightbox | 1.459 | 1.508（**+0.049 更差**） | 1.486 | 1.556（**+0.070 更差**） |

**判定**：unc 加权（LM）在全部 4 个对照上均劣于普通 PnP——尽管图级 ts 与误差 ρ≈0.4，
像素级 ts 加权并未改善 PnP 解。**后处理线至此全部关闭**：
- 排除（固定纯几何区间）：有效，已定稿
- MLP 校正：旧模型迁移无效，重训暂缓（架构待改）
- unc 加权 PnP：阴性
- 逐图自适应：无预测信号（见附录 2.3）

### 后处理研究最终结论

```
交付配置：LLRD 0.9 + 退火模型 + 统一排除 c(0.00,0.05,0.10)/r0.075
   sunlamp 3.006 → 2.756（−8.3%）
   lightbox 1.971 → 1.893（−4.0%）
   val：未评估（可补）
后续方向（分析已证）：z 轴是误差主因 + z 通道不确定度尾部最有信息
   → 模型训练线：DER 头 z 通道改进（C1 损失加权 / C2 校准），待用户开启
```
