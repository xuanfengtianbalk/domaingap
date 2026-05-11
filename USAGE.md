# Usage Guide

## 激活环境
```bash
conda activate dinov3
cd /opt/dl_workspace/algorithm/04-myself/domaingap
```

## 训练 (train)

```bash
python run.py --mode train --gpu 0 \
  --TRAIN.LR 2e-4 --TRAIN.MAX_EPOCH 50 --TRAIN.BATCH_SIZE 16 \
  --TRAIN.P_AUG_SUN 0 --MODEL.ACTIVATE None \
  --model_type coordinates
```

### 训练参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode train` | 训练模式 | |
| `--gpu 0` / `0 1` | GPU ID | |
| `--model_type` | 模型类型 | `coordinates`, `keypoints_gs`, `coordinates_gs`, `coordinates keypoints_gs` |
| `--TRAIN.LR` | 学习率 | `2e-4` |
| `--TRAIN.MAX_EPOCH` | 总 epoch 数 | `50` |
| `--TRAIN.BATCH_SIZE` | 批大小 | `16` / `32` |
| `--TRAIN.P_AUG_SUN` | Sun flare 增强概率 | `0` / `1` |
| `--MODEL.ACTIVATE` | 激活函数 | `None`, `sigmoid`, `softmax` |
| `--MODEL.BACKBONE_NAME` | 骨干网络 | `dinov3_vits16`, `dinov3_vitb16` |
| `--train_backbone` | 解冻 backbone | 加 flag 解冻 |
| `--MODEL.BIN_CONVERTER.LOSS_REDUCTION` | bin loss 模式 | `mean` / `sum` |
| `--MODEL.BIN_CONVERTER.USE_MASK` | 是否用 mask head | `True` / `False` |
| `--MODEL.BIN_CONVERTER.SAMPLE_RANGE` | bin 范围 | `[-0.7,0.7]` |

### coordinates_gs 示例
```bash
# USE_MASK=False, LOSS_REDUCTION=sum
python run.py --mode train --gpu 0 --TRAIN.LR 1e-4 --TRAIN.MAX_EPOCH 50 \
  --TRAIN.BATCH_SIZE 16 --TRAIN.P_AUG_SUN 0 --MODEL.ACTIVATE None \
  --MODEL.BIN_CONVERTER.LOSS_REDUCTION sum \
  --MODEL.BIN_CONVERTER.USE_MASK False \
  --model_type coordinates_gs

# 解冻 backbone
python run.py --mode train --gpu 0 --TRAIN.LR 2e-4 --TRAIN.MAX_EPOCH 50 \
  --TRAIN.BATCH_SIZE 16 --TRAIN.P_AUG_SUN 0 --MODEL.ACTIVATE None \
  --model_type coordinates_gs --train_backbone
```

## 评估 (evaluate)

对已有权重跑 sunlamp + lightbox eval，结果存回权重文件夹：

```bash
python run.py --mode evaluate --gpu 0 --MODEL.ACTIVATE None \
  --model_type coordinates_gs --resume_path c04d815b
```

| 参数 | 说明 |
|------|------|
| `--mode evaluate` | 评估模式 |
| `--resume_path` | `workingdir/` 下的 UUID 文件夹名 |

生成文件：
- `workingdir/{UUID}/lightbox_result_coordinates_gs.json`
- `workingdir/{UUID}/sunlamp_result_coordinates_gs.json`

## 可视化 (vis)

对已有权重生成可视化图像，保存到 `visuals/{UUID}/`：

```bash
# sunlamp 可视化
python run.py --mode sunlamp --gpu 0 --MODEL.ACTIVATE None \
  --model_type coordinates --resume --resume_path c04d815b

# lightbox 可视化
python run.py --mode lightbox --gpu 0 --MODEL.ACTIVATE None \
  --model_type coordinates --resume --resume_path c04d815b

# coordinates_gs 可视化 (USE_MASK=False)
python run.py --mode sunlamp --gpu 0 --MODEL.ACTIVATE None \
  --MODEL.BIN_CONVERTER.USE_MASK False \
  --model_type coordinates_gs --resume --resume_path c04d815b
```

## 本地 screen 操作

```bash
screen -ls                    # 查看所有 screen
screen -r <name>              # 连接
screen -dmS <name> bash ...   # 创建后台 screen
Ctrl+A D                      # 断开当前 screen
screen -S <name> -X quit      # 关闭
```

## 远程服务器

```bash
ssh -p 301 sat001@192.10.84.217   # 密码 Admin@9000
cd ~/lk/domaingap
conda activate dinov3
```

## 配置文件

`configs/cfg.yaml` 中的关键字段：

```yaml
MODEL:
  TYPE: []                    # 模型类型列表
  BACKBONE_NAME: "dinov3_vits16"
  ACTIVATE: 'None'
  BIN_CONVERTER:              # coordinates_gs 专用
    SAMPLE_RANGE: [-0.7, 0.7]
    N_PER_UNIT: 30
    SIGMA_FACTOR: 1.5
    LOSS_TYPE: 'js'
    LOSS_REDUCTION: 'mean'    # mean / sum
    USE_MASK: true

TRAIN:
  LR: 5e-4
  MAX_EPOCH: 1
  BATCH_SIZE: 16
  P_AUG_SUN: 0.0
```

命令行 `--TRAIN.LR 1e-4` 会覆盖配置文件中的值。
