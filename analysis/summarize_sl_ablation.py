"""汇总所有 Stable Learning 消融实验结果到 outputs/sl_ablation_summary.csv。

每行：分组 / run 名 / uuid / 超参 / 三个 split 的 angle(mean±std) + dist。
运行: python summarize_sl_ablation.py
"""

import json
import os
import csv
import numpy as np

WORKINGDIR = os.path.join(os.path.dirname(__file__), "..", "workingdir")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "sl_ablation_summary.csv")

# 共同协议（所有 stable 微调 run 一致）
COMMON = ("e24d72fb续训;coordinates_DER;vitl16;lora_r1;AdamW;batch16;augmix;10ep;seed42;"
          "presave_ratio=0.9;epochp=0;n_feature=16;decay_pow=2;first_step_cons=1")

# (group, run, uuid, enable, num_f, epochb, lrbl, lambdap, train_lr, note)
RUNS = [
    # ── 基线 ──
    ("baseline", "e24d72fb_base", "e24d72fb-b4d7-4c40-b00a-a4aa2f8213e8",
     "-", "-", "-", "-", "-", "1e-4", "原始模型,50ep,无微调"),
    # ── 单变量消融（早期，lr=1e-4）──
    ("single_var", "A_stable_default", "[Stable Learning]f869fdd5-8846-443a-b75e-e15a63738e93",
     True, 1, 20, 1.0, 70.0, "1e-4", "官方默认"),
    ("single_var", "B_plain", "b8752da5-ed2e-40e8-94fd-5a30c08be394",
     False, 1, 20, 1.0, 70.0, "1e-4", "plain对照(enable=False)"),
    ("single_var", "C_lambdap40", "e3aca5e2-b7c1-451d-bc73-96a1ae8a90a2",
     True, 1, 20, 1.0, 40.0, "1e-4", "lambdap 70→40"),
    ("single_var", "D_lambdap20", "32319e4f-b39c-4b23-8ff1-540e460ec9d4",
     True, 1, 20, 1.0, 20.0, "1e-4", "lambdap 70→20"),
    ("single_var", "F_lrbl3", "0de724f8-d3fd-42b7-a144-176ff752b29f",
     True, 1, 20, 3.0, 70.0, "1e-4", "lrbl 1.0→3.0"),
    # ── num_f × epochb 网格（lr=1e-4）──
    ("grid", "(3,20)", "939e0f35-2070-4638-85d2-e9e539e51fa7",
     True, 3, 20, 1.0, 70.0, "1e-4", ""),
    ("grid", "(3,30)=E", "90e03184-e647-4932-a979-31b822f3bd85",
     True, 3, 30, 1.0, 70.0, "1e-4", ""),
    ("grid", "(3,50)", "dc101746-0f0d-40f3-b4dc-6d5ad00bfb97",
     True, 3, 50, 1.0, 70.0, "1e-4", ""),
    ("grid", "(5,20)", "f651cec9-9492-47c1-9d5b-798e33e438ae",
     True, 5, 20, 1.0, 70.0, "1e-4", ""),
    ("grid", "(5,30)", "792f8018-9124-4be8-a15c-83fc0231725f",
     True, 5, 30, 1.0, 70.0, "1e-4", ""),
    ("grid", "(5,50)", "d0f7d77d-d154-4c74-a57c-0fc97920863f",
     True, 5, 50, 1.0, 70.0, "1e-4", ""),
    ("grid", "(10,20)", "0198b01e-0763-4248-b89e-cac01f9061d1",
     True, 10, 20, 1.0, 70.0, "1e-4", ""),
    ("grid", "(10,30)", "ec0ad17a-c96e-4771-b547-cc3651eeb855",
     True, 10, 30, 1.0, 70.0, "1e-4", ""),
    ("grid", "(10,50)", "c8bae526-5c38-48c5-b1b1-341947486388",
     True, 10, 50, 1.0, 70.0, "1e-4", ""),
    # ── TRAIN.LR 消融（num_f=3, epochb=30）──
    ("lr_sweep", "E_lr1e-4", "90e03184-e647-4932-a979-31b822f3bd85",
     True, 3, 30, 1.0, 70.0, "1e-4", ""),
    ("lr_sweep", "L1_lr5e-5", "589510f6-cc5e-4e35-a6e3-4f6b0b3d6031",
     True, 3, 30, 1.0, 70.0, "5e-5", ""),
    ("lr_sweep", "L2_lr2e-5", "61d0eb63-2931-4735-9e3e-56d5b63f9c91",
     True, 3, 30, 1.0, 70.0, "2e-5", ""),
    ("lr_sweep", "L3_lr1e-5", "d58d5405-b8f4-4933-9f88-a0c0b7c18ce6",
     True, 3, 30, 1.0, 70.0, "1e-5", ""),
    ("lr_sweep", "L4_lr5e-6", "2f7e66b1-e0d3-4735-a515-71c8e7c39393",
     True, 3, 30, 1.0, 70.0, "5e-6", ""),
    ("lr_sweep", "L5_lr2e-6", "443c397c-0f9e-49bb-a34a-f87a66a29382",
     True, 3, 30, 1.0, 70.0, "2e-6", ""),
    ("lr_sweep", "L6_lr1e-6", "ea82fb54-153d-4c37-a774-175e3114423a",
     True, 3, 30, 1.0, 70.0, "1e-6", ""),
    ("lr_sweep", "L7_lr5e-7", "8f55f1dd-4103-432b-85c7-7b3431bfb89a",
     True, 3, 30, 1.0, 70.0, "5e-7", "训练中"),
    ("lr_sweep", "L8_lr2e-7", "6a45dd00-d669-47b3-b71c-738fdc73d9c3",
     True, 3, 30, 1.0, 70.0, "2e-7", "训练中"),
    ("lr_sweep", "L9_lr1e-7", "",
     True, 3, 30, 1.0, 70.0, "1e-7", "排队"),
]


def load_metrics(uuid, split):
    """返回 (angle_mean, angle_std, dist_mean) 或 (None, None, None)"""
    if not uuid:
        return None, None, None
    p = os.path.join(WORKINGDIR, uuid, f"{split}_result_coordinates_DER.json")
    if not os.path.exists(p):
        return None, None, None
    d = json.load(open(p))
    err = np.array([r["err_ori"] for r in d])
    los = np.array([r["los_r"] for r in d])
    return float(err.mean()), float(err.std()), float(los.mean())


def main():
    header = [
        "group", "run", "uuid", "enable", "num_f", "epochb", "lrbl", "lambdap",
        "train_lr", "note",
        "sunlamp_angle_mean", "sunlamp_angle_std", "sunlamp_dist_mean",
        "lightbox_angle_mean", "lightbox_angle_std", "lightbox_dist_mean",
        "val_angle_mean", "val_angle_std", "val_dist_mean",
        "common_config",
    ]
    rows = []
    for group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note in RUNS:
        row = [group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note]
        for split in ["sunlamp", "lightbox", "validation"]:
            am, asd, dm = load_metrics(uuid, split)
            row += [
                f"{am:.4f}" if am is not None else "",
                f"{asd:.4f}" if asd is not None else "",
                f"{dm:.4f}" if dm is not None else "",
            ]
        row.append(COMMON)
        rows.append(row)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"written {len(rows)} rows -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
