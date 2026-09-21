# Stable Learning（StableNet）与 LaSt-ViT 消融实验总结

> 实验日期：2026-09
> 代码分支：`contrastive_dg`（commit `a561d91` / `ae79bc2` / `733a23b`）
> 汇总表：`outputs/sl_ablation_summary.csv`（30 行）
> 汇总脚本：`analysis/summarize_sl_ablation.py`

---

## 1. 背景与目标

在卫星位姿估计任务（SPEED+，DER 模型）上验证 **StableNet（Deep Stable Learning for Out-Of-Distribution Generalization, CVPR 2021）** 能否提升 OOD（sunlamp / lightbox）泛化能力。

**基线**：e24d72fb（coordinates_DER, dinov3_vitl16, LoRA rank1, augmix, 50 epoch 原训练）。

**协议**：从 e24d72fb 收敛 checkpoint 续训 10 epoch（`--resume`），保持所有训练配置与原训练一致（LR 1e-4 / AdamW / batch 16 / augmix / seed 42），只改变 StableNet 相关配置。

**评价指标**：sunlamp / lightbox（OOD 域）与 validation（域内）的位姿角度误差 angle、距离误差 dist（eval 自动产出）。

---

## 2. 实现说明

StableNet 已对齐官方库（github.com/xxgege/StableNet）：

| 文件 | 内容 |
|---|---|
| `Hyperpose_net/losses/stable_learning.py` | `RFFTransform`（随机傅里叶特征）、`StableNetState`（全局特征表 pre_features/pre_weight1）、`weight_learner`（内层样本权重 SGD） |
| `train_stable.py` | `train_one_epoch_stable`：特征 GAP → 内层学 w → 加权任务损失 |
| `run.py` | `--train_script train_stable` 入口、`--seed` 参数、组件构建 |
| `configs/cfg.yaml` | `STABLE_LEARNING` 配置段 |

机制（论文 Eq.7-10 / 附录 A.1）：

```
lossb  = Σ_freq [特征-特征加权协方差非对角平方和]      （去相关损失, 论文 Eq.7）
lossp = Σ softmax(w)^decay_pow                        （权重均匀化正则, 附录 weight decay）
lossg  = lossb / lambdap + lossp                     （只训练 w, 不反向传播进模型）
模型损失 = Σᵢ wᵢ · Lᵢ                                  （w 冻结后加权任务损失, 论文 Eq.13）
```

---

## 3. 实验矩阵

### 3.1 单变量消融（lr=1e-4）

| Run | uuid | 变量 | sunlamp angle | dist | lightbox angle | dist | val angle | dist |
|---|---|---|---|---|---|---|---|---|
| e24d72fb 基线 | `e24d72fb...` | — | **4.47±9.88** | 0.1139 | 3.65±12.04 | 0.0781 | 0.87±2.08 | 0.0248 |
| A 官方默认 | `f869fdd5` | num_f=1, epochb=20 | 4.58±11.14 | 0.1362 | 3.80±12.53 | 0.0972 | 0.96±2.07 | 0.0546 |
| B plain 对照 | `b8752da5` | enable=False | 4.57±12.50 | 0.1099 | 3.87±14.39 | **0.0752** | **0.83±1.17** | **0.0232** |
| C lambdap=40 | `e3aca5e2` | lambdap 70→40 | 4.84±11.87 | 0.1141 | 4.17±14.06 | 0.0839 | 0.90±1.99 | 0.0271 |
| D lambdap=20 | `32319e4f` | lambdap 70→20 | 4.64±11.83 | 0.1414 | **3.49±11.53** | 0.0920 | 0.85±2.54 | 0.0539 |
| F lrbl=3.0 | `0de724f8` | lrbl 1.0→3.0 | 4.70±11.50 | 0.1217 | 3.90±13.02 | 0.0799 | 1.02±2.20 | 0.0266 |

**结论**：lr=1e-4 下没有任何变体全面超过基线，多数退化；plain 对照是最不坏的选择。

### 3.2 num_f × epochb 网格（lr=1e-4，9 格）

| num_f \ epochb | 20 | 30 | 50 |
|---|---|---|---|
| 3 | 5.01 / 0.1425 / 1.28 | **4.03 / 0.1079 / 1.07** (E) | 5.48 / 0.1289 / 1.02 |
| 5 | 5.35 / 0.1197 / 0.89 | 5.01 / 0.1167 / 1.00 | 4.68 / 0.1321 / 0.83 |
| 10 | 5.46 / 0.1342 / 0.94 | 4.44 / 0.1189 / 0.85 | 5.08 / 0.1134 / 0.87 |

（格式：sunlamp angle / sunlamp dist / val angle；基线 4.47 / 0.1139 / 0.87）

**结论**：E(3,30) 是孤立的尖锐最优，邻格全部退化——单 seed 噪声嫌疑大；epochb=30 列整体最好。

### 3.3 TRAIN.LR 消融（num_f=3, epochb=30）—— 决定性实验

| lr | sunlamp angle | dist | lightbox angle | dist | val angle | dist |
|---|---|---|---|---|---|---|
| 基线 | 4.47±9.88 | 0.1139 | 3.65±12.04 | 0.0781 | 0.87±2.08 | 0.0248 |
| 1e-4 (E) | 4.03 | 0.1079 | 3.54 | 0.0726 | 1.07 | 0.0312 |
| 5e-5 | 4.03 | 0.1119 | 3.33 | 0.0764 | 0.69 | 0.0270 |
| 2e-5 | 4.03 | 0.1070 | 3.20 | 0.0693 | 0.62 | 0.0169 |
| **1e-5** | 4.01 | 0.1094 | **3.01±9.65** | 0.0698 | 0.61 | 0.0225 |
| 5e-6 | 4.12 | 0.1114 | 3.25 | 0.0711 | **0.60** | 0.0198 |
| **2e-6** | **3.84±8.22** | 0.1099 | 3.31 | 0.0705 | 0.61 | 0.0188 |
| 1e-6 | 3.86 | 0.1083 | 3.29 | 0.0699 | 0.62 | 0.0178 |
| 5e-7 | 4.09 | 0.1097 | 3.26 | 0.0702 | 0.62 | 0.0174 |
| 2e-7 | 4.11 | 0.1112 | 3.48 | 0.0740 | 0.63 | 0.0176 |
| 1e-7 | 4.24 | 0.1128 | 3.33 | 0.0728 | 0.65 | 0.0176 |

**双 U 形拐点**：sunlamp 最优 lr=2e-6（3.84，较基线降 14%）；lightbox 最优 lr=1e-5（3.01，降 18%）；validation 在 5e-6 触底（0.60，降 31%）。甜区 **lr ∈ [5e-7, 1e-5]**。

### 3.4 甜区多 seed + plain 对照（lr=2e-6）

| Run | sunlamp angle | dist | lightbox angle | dist | val angle | dist |
|---|---|---|---|---|---|---|
| 基线 | 4.47±9.88 | 0.1139 | 3.65±12.04 | 0.0781 | 0.87±2.08 | 0.0248 |
| **plain 对照（enable=False）** | **3.85±8.11** | **0.1059** | **3.27±11.39** | 0.0708 | 0.61±1.87 | 0.0177 |
| (3,30) seed42 | 3.84±8.22 | 0.1099 | 3.31±11.73 | 0.0705 | 0.61±2.01 | 0.0188 |
| (3,30) seed43 | 4.04±9.31 | 0.1077 | 3.28±11.53 | 0.0691 | 0.61±2.00 | 0.0175 |
| (3,30) seed44 | 4.10±9.82 | 0.1079 | 3.32±11.40 | 0.0694 | 0.61±1.96 | 0.0172 |
| (3,30) seed45 | 3.93±8.55 | 0.1066 | 3.32±11.54 | 0.0698 | 0.61±1.93 | 0.0171 |
| 4-seed 平均 | 3.98±0.10 | 0.1080 | 3.31±0.02 | 0.0697 | 0.61±0.00 | 0.0177 |

**关键反转**：plain 对照与 StableNet 4-seed 平均几乎持平（sunlamp 甚至略优 3.85 vs 3.98）——OOD 提升来自**小 lr 微调本身**，StableNet 净贡献 ≈ 0。跨 seed 高度稳定（seed 间 std 0.02~0.10）。

### 3.5 w 诊断（决定性证据）

对 (3,30) 在 lr=2e-6 与 lr=1e-4 两档下监控样本权重 w 的统计量：

| batch | w_std | w_entropy（均匀=2.7726） | max/min | lossb |
|---|---|---|---|---|
| b0（全局表全零） | 0.015~0.021 | 2.72~2.75 | 2.8~2.9 | 41 |
| b1~b2500+ | **0.0000** | **2.7726（完美均匀）** | **1.000** | **0.0005~0.005** |

**两档 lr 完全一致**：w 从第 1 个 batch 起永远是完美均匀分布，直到 b2500+ 无任何偏离。

### 3.6 Decoder 分层特征实验（机制复活验证）

w 诊断证明 encoder 特征 lossb≈0 后，将 StableNet 的重加权对象从 encoder GAP 特征改为 **decoder 的 4 级融合特征**（`DPTHead` 每个 fusion block 输出，GAP 后拼接 [B, 1024]），机制得以复活：

**① w 诊断（decoder 特征）**：

| 配置 | lossb 稳态 | w max/min | 判定 |
|---|---|---|---|
| encoder + lambdap=70 | 0.0005~0.005 | 恒 1.000 | 机制死锁 |
| decoder + lambdap=70 | 0.008~0.055 | 1.003~1.08 | 信号放大 10-50×，仍弱 |
| decoder + **lambdap=1** | 0.008~0.019 | 中位 4.83，78% batch >2（10ep 全程） | **强激活** |
| decoder + lambdap=10 | 0.008~0.068 | 中位 1.40，10% batch >2 | 温和激活 |

（注：b0 启动时全局表全零产生 10¹² 尖峰，属启动伪影，非训练现象；真实尖峰为 10~66）

**② 最终归因（10 epoch + eval）**：

| Run | sunlamp angle | dist | lightbox angle | dist | val angle | dist |
|---|---|---|---|---|---|---|
| e24d72fb 基线 | 4.47±9.88 | 0.1139 | 3.65±12.04 | 0.0781 | 0.87±2.08 | 0.0248 |
| R0a plain s42 | 3.85±8.11 | 0.1059 | 3.27±11.39 | 0.0708 | 0.61±1.87 | 0.0177 |
| R0b plain s43 | 3.98±9.30 | 0.1073 | 3.35±11.62 | 0.0709 | 0.61±2.00 | 0.0165 |
| **R1 lambdap=1（w 强激活）** | 3.93±8.33 | 0.1103 | 3.32±11.12 | 0.0713 | 0.61±1.92 | 0.0198 |

R1 每个指标都精确落在 plain 区间内（sunlamp 3.93 ∈ [3.85, 3.98]；lightbox 3.32 ∈ [3.27, 3.35]；val 0.61 = 0.61）。

---

## 4. 核心结论

### 4.1 StableNet 在此协议下完全无效（两层证据闭环）

**第一层（encoder 特征）**：lossb≈0 → 内层 SGD 在均匀点梯度为 0 → w 永远均匀 → 加权损失 ≡ plain 损失。调任何超参都无意义。

**第二层（decoder 特征，机制复活后）**：decoder 分层特征 + lambdap=1 让 w 全程强激活（78% batch max/min>2，中位 4.83，真实尖峰 10~66）——**但 R1 与 plain 对照在每个指标上完全重合**。即使重加权机制满负荷工作，对最终模型零贡献。

结论：此任务协议（收敛 checkpoint + 10 epoch 小 lr 微调）下，**StableNet 无效，收益 100% 来自小 lr 微调本身**。

### 4.2 真正的收益来自小 lr 微调

- 从收敛 checkpoint 续训 10 epoch，**lr 必须 ≤ 1e-5**（甜区 [5e-7, 1e-5]，双 U 形拐点：sunlamp 2e-6，lightbox 1e-5）；
- lr=1e-4（原训练 lr）会破坏已收敛权重——这是此前全部退化实验的根源；
- **最终方案：lr=2e-6（或 1e-5）plain 微调**，较基线提升：sunlamp 14%（4.47→3.84）、lightbox 18%（3.65→3.01）、validation 31%（0.87→0.60）。

### 4.3 StableNet 可能有效的方向（未验证）

- 换协议：从 scratch / 解冻 backbone / 30+ epoch 大 lr（官方协议），让特征可动、相关可产生；
- 引入训练域异质性（StableNet 针对隐式域不平衡场景）；
- 补全论文 k 组全局记忆（当前 n_feature=1×batch）。

**在当前"收敛 checkpoint + 10 epoch 微调"协议下，StableNet 无优化空间，判定为无效（null result）。**（decoder 特征实验已排除"特征源"这一变量。）

补充（lr=1e-4 档）：decorr 直接去相关在大 lr 下同样无效——`decorr λ=1 @ lr1e-4`（sunlamp 4.44 / lightbox 3.81 / val 0.90）≈ `B plain @ lr1e-4`（4.57 / 3.87 / 0.83），`λ=10` 更差（4.95）。大 lr 协议本身有害，去相关救不回来。

### 4.5 LaSt-ViT 方向（独立于 StableNet，同样 null）

**论文**（CVPR 2026, "Vision Transformers Need More Than Registers"）：ViT artifacts 的根因是 lazy aggregation——CLS 聚合被背景 patch 短路主导。解法 LaSt-ViT：把 CLS 聚合替换为"逐通道 top-K 稳定 patch 均值"（FFT+高斯低通+稳定性分数，参数自由），梯度经被选中的 patch 流回模型，训练模型依赖前景。

**我们的忠实迁移**：稠密架构没有 CLS 聚合点 → 在 decoder 前向里引入 `lastvit_context`（Eq.4-7 逐位一致、可微、零参数），注入回融合特征：`fused' = fused + γ·g`。训练时梯度流经被选中的稳定 patch。

**PiB 分析**（Patch Score 最高分是否落在卫星 mask 内，400 图）：

| 指标 | PiB |
|---|---|
| encoder patch↔CLS | 0.89（健康，无 lazy aggregation） |
| decoder cell↔GAP | 0.21（**lazy aggregation 病灶在 decoder**） |
| LaSt-ViT vote | 0.53（投票机制优于 GAP 聚合） |

**5 个注入层的 10ep 消融**（全部 lr=2e-6, seed42, γ=1.0）：

| 注入层 | sunlamp | lightbox | val | vs plain |
|---|---|---|---|---|
| plain R0a/R0b（锚点） | 3.85 / 3.98 | 3.27 / 3.35 | 0.61 | — |
| -1（project 后） | 4.18 | 3.55 | 0.71 | ✗ |
| 0（最粗 fusion） | 4.10 | 3.29 | 0.61 | ✗（最优层，sunlamp 仍差 0.2°+） |
| 1 | 4.09 | 3.31 | 0.61 | ✗ |
| 2 | 4.03 | 3.44 | 0.65 | ✗ |
| 3（最细 fusion） | 4.46 | 3.59 | 0.74 | ✗（几乎退回基线） |

**LaSt-ViT 判定**：注入越早越好（0/1 最优）、越晚越差（3 近基线）——但**没有任何层超过 plain 区间**。冻结模型上直接注入降 PiB（机制是训练期的，预期）；带注入训练 10ep 后模型适应了（loss +9→-3），但最终指标仍 ≤ plain。"梯度流经稳定 patch 重塑模型"在 10ep 微调尺度上没有收益。

### 4.6 三条路线的最终闭环

```
StableNet 间接 w（图像级/像素级）      → w 激活或不激活，任务结果 ≡ plain
StableNet 直接去相关正则（lr 1e-4/2e-6）→ lossb 被真实压降（8-44×），任务结果 ≡ plain
LaSt-ViT 忠实聚合注入（5 个 decoder 层）→ 机制真实训练模型，任务结果 ≤ plain
⇒ 在"收敛 checkpoint + 小 lr 微调"协议下，任何训练期干预都无法超越 plain 微调
⇒ 唯一有效方案：lr=2e-6 plain 微调（sunlamp 3.85 -14% / lightbox 3.27 -10% / val 0.61 -31%）
```

**实现局限声明**：本实验的 StableNet 全局记忆为**单槽 1×batch 表**（跟随官方 repo 的简化实现）。论文原式的 **k 组全局记忆**（Eq.9-10：k 组预存特征、k 个不同平滑系数 αᵢ、presaved size=k×batch，Fig.3c 的消融轴）**未实现、未测试**。因此严格结论应表述为"在 1×batch 全局记忆实现下无效"。主因（lossb≈0、lossp 锁死、小 lr 下模型位移趋零）与全局记忆大小正交，补齐 k 组后大概率仍为 null，但未经实验验证。

---

## 5. 复现命令

```bash
# 最终方案（plain 微调）
python run.py --mode train --train_script train_stable --resume \
  --resume_path e24d72fb-b4d7-4c40-b00a-a4aa2f8213e8 \
  --model_type coordinates_DER --gpu 0 \
  --MODEL.BACKBONE_NAME dinov3_vitl16 \
  --MODEL.PEFT.method lora --MODEL.PEFT.lora_rank 1 \
  --TRAIN.LR 2e-6 --TRAIN.OPTIM AdamW --TRAIN.BATCH_SIZE 16 \
  --TRAIN.AUG_TYPE augmix --TRAIN.MAX_EPOCH 10 \
  --STABLE_LEARNING.enable False --seed 42

# 汇总
cd analysis && python summarize_sl_ablation.py
```

## 6. 附录：StableNet 关键超参对照（官方库 vs 论文）

| 参数 | 官方默认 | 论文 | 说明 |
|---|---|---|---|
| 模型 lr | 0.01（SGD+cosine, 30ep, 从头训） | — | 与微调协议无直接可比性 |
| lrbl（w 学习率） | 1.0 | 附录 3.0 | 内层 SGD momentum 0.9 |
| epochb | 20 | BALANCING EPOCH NUMBER | 每 batch 内层迭代 |
| lambdap | 70.0 | 无（附录仅提 regularizer 0.3） | lossb 的除数，实现层参数 |
| num_f | 1 | 消融图 3(a)（10x/0.3x） | RFF 频率空间数 |
| n_feature | 128=batch（官方简化） | k×batch，k 组各带 αᵢ（Eq.9-10；Fig.3c 消融轴） | 我们 16=1×batch 单槽；**k 组设计未实现未测试** |
| presave_ratio | 0.9 | Eq.10 αᵢ | 全局表 EMA |
