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
          "presave_ratio=0.9;epochp=0;n_feature=16(1xbatch,k组全局记忆未实现);decay_pow=2;first_step_cons=1")

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
     True, 3, 30, 1.0, 70.0, "5e-7", ""),
    ("lr_sweep", "L8_lr2e-7", "6a45dd00-d669-47b3-b71c-738fdc73d9c3",
     True, 3, 30, 1.0, 70.0, "2e-7", ""),
    ("lr_sweep", "L9_lr1e-7", "bc436b9f-19e0-4417-aa3f-ad345c02d884",
     True, 3, 30, 1.0, 70.0, "1e-7", ""),
    # ── 甜区多 seed + plain 对照（lr=2e-6）──
    ("seed_verify", "plain_lr2e-6_seed42", "55748478-38a0-43da-89c3-bfe480dbe46e",
     False, 3, 30, 1.0, 70.0, "2e-6", "plain对照 enable=False"),
    ("seed_verify", "S42_lr2e-6", "443c397c-0f9e-49bb-a34a-f87a66a29382",
     True, 3, 30, 1.0, 70.0, "2e-6", "=L5"),
    ("seed_verify", "S43_lr2e-6", "b60e1c63-c599-40e0-817a-fa771f4cda25",
     True, 3, 30, 1.0, 70.0, "2e-6", ""),
    ("seed_verify", "S44_lr2e-6", "02eff09a-c6d3-4456-9483-543c37b2628d",
     True, 3, 30, 1.0, 70.0, "2e-6", ""),
    ("seed_verify", "S45_lr2e-6", "ca30883e-54be-4d77-ad51-130150b129b8",
     True, 3, 30, 1.0, 70.0, "2e-6", ""),
    # ── decorr @ lr=1e-4（stable learning 大 lr 收尾）──
    ("lr1e-4_decorr", "E1_decorr_lam1", "98ec02c1-ad7f-43aa-9c6d-39b8bb93289b",
     True, 3, 30, 1.0, 70.0, "1e-4", "decorr_reg,lambda=1"),
    ("lr1e-4_decorr", "E2_decorr_lam10", "6ddf2440-7bf1-4f8c-a28d-571ff040cd07",
     True, 3, 30, 1.0, 70.0, "1e-4", "decorr_reg,lambda=10"),
    # ── LaSt-ViT 忠实注入（decoder 各层, γ=1.0, lr=2e-6）──
    ("lastvit", "LV_layer-1_proj", "fcc07591-27d1-4c35-92d5-f5e40c84760f",
     False, 3, 30, 1.0, 70.0, "2e-6", "LASTVIT_CONTEXT,layer=-1"),
    ("lastvit", "LV_layer0", "ccd044f6-193e-4047-9863-e8631c1a5bf4",
     False, 3, 30, 1.0, 70.0, "2e-6", "LASTVIT_CONTEXT,layer=0"),
    ("lastvit", "LV_layer1", "26d2d5ee-2a61-4a53-81fa-38043f506f8e",
     False, 3, 30, 1.0, 70.0, "2e-6", "LASTVIT_CONTEXT,layer=1"),
    ("lastvit", "LV_layer2", "9c09911b-2d57-48a2-aa7f-b4c6bc2ea58a",
     False, 3, 30, 1.0, 70.0, "2e-6", "LASTVIT_CONTEXT,layer=2"),
    ("lastvit", "LV_layer3", "c5a74357-c18d-4ff0-b09c-311f7ba603ef",
     False, 3, 30, 1.0, 70.0, "2e-6", "LASTVIT_CONTEXT,layer=3"),
    # ── decoder 分层特征实验（lr=2e-6）──
    ("decoder", "D_R1_lam1_s42_10ep", "b7e907b2-fcf8-43af-9840-7b8483f04836",
     True, 3, 30, 1.0, 1.0, "2e-6", "decoder特征,lambdap=1,w强激活"),
    ("decoder", "D_R0b_plain_s43_10ep", "c329bf95-53ac-4176-8e08-f2acdd389d2e",
     False, 3, 30, 1.0, 70.0, "2e-6", "decoder协议plain对照,seed43"),
    ("decoder", "D_P1_lam1_s42_1ep", "f26ebd0f-a81b-434a-aa0a-67dd433096c3",
     True, 3, 30, 1.0, 1.0, "2e-6", "1ep诊断,lambdap=1"),
    ("decoder", "D_P3_lam10_s42_1ep", "31b997fc-b379-452b-9acd-1ec511a23fd6",
     True, 3, 30, 1.0, 10.0, "2e-6", "1ep诊断,lambdap=10"),
]

# ── 微调方法对比轮（DINOv3预训练+新头, 50ep 退火 1e-4→2e-6）──
COMMON_FT = ("DINOv3预训练+新头;coordinates_DER;vitl16;batch16;augmix;seed42;"
             "50ep;cosine退火1e-4->2e-6;warmup1000")
FT_RUNS = [
    ("ft_methods", "FT_LoRA_anchor_50ep", "25e06910-052f-4a3c-93b9-3612999915b7",
     "-", "-", "-", "-", "-", "anneal", "忠实LoRA(官方MergedLinear,qv,r1,α/r,kaiming)"),
    ("ft_methods", "FT_LPFT_s1_head25ep", "99fa6859-8ff0-4701-bb92-0279bd1e29b0",
     "-", "-", "-", "-", "-", "anneal", "LP-FT阶段1:冻结encoder仅训头25ep"),
    ("ft_methods", "FT_LPFT_final_25+25", "240f0e28-f3b8-4ef1-a4a8-095040e764cd",
     "-", "-", "-", "-", "-", "anneal", "LP-FT两阶段(25头+25联合)"),
    ("ft_methods", "FT_L2SP_alpha0.01", "46bf9c64-9e6c-40d0-a67b-3802b6c2ec9e",
     "-", "-", "-", "-", "-", "anneal", "L2-SP α=0.01 β=0.01 SGD mom0.9 全量"),
    ("ft_methods", "FT_L2SP_alpha0.1", "803126e6-2b60-425c-b719-688be43c3f66",
     "-", "-", "-", "-", "-", "anneal", "L2-SP α=0.1 β=0.01 SGD mom0.9 全量"),
    ("ft_methods", "FT_wiseft_a0.3", "25e06910-052f-4a3c-93b9-3612999915b7_wise0.3",
     "-", "-", "-", "-", "-", "-", "WiSE-FT α=0.3(LoRA delta缩放,零训练)"),
    ("ft_methods", "FT_wiseft_a0.5", "25e06910-052f-4a3c-93b9-3612999915b7_wise0.5",
     "-", "-", "-", "-", "-", "-", "WiSE-FT α=0.5"),
    ("ft_methods", "FT_wiseft_a0.7", "25e06910-052f-4a3c-93b9-3612999915b7_wise0.7",
     "-", "-", "-", "-", "-", "-", "WiSE-FT α=0.7"),
    ("ft_methods", "FT_soup_3models", "soup_uniform_3models_8031_46bf_240f",
     "-", "-", "-", "-", "-", "-", "均匀soup{L2SP0.1,L2SP0.01,LPFT}(2成员已崩)"),
]
# ── 批2：固定 lr 1e-4 协议（无退火）──
COMMON_FT2 = ("DINOv3预训练+新头;coordinates_DER;vitl16;batch16;augmix;seed42;"
              "50ep;固定lr=1e-4;warmup1000")
FT_RUNS2 = [
    ("ft_methods", "FT_LPFT_fixed1e-4_s1", "dc4453b9-e619-4294-bf62-7d3bc3a93495",
     "-", "-", "-", "-", "-", "1e-4", "LP-FT阶段1固定lr(冻结训头25ep)"),
    ("ft_methods", "FT_LPFT_fixed1e-4_s2", "3d0bd107-cc17-4018-8c0d-b96a7835fbaf",
     "-", "-", "-", "-", "-", "1e-4", "LP-FT固定lr最终(25+25,无退火)"),
    ("ft_methods", "FT_LLRD_decay0.9", "8be5bc36-b5fb-495d-bc46-14681a3fc74b",
     "-", "-", "-", "-", "-", "1e-4", "逐层lr衰减0.9,26组,全量微调"),
    ("ft_methods", "FT_LNtune_50ep", "d249a850-f503-446c-8903-e5ae6ab45698",
     "-", "-", "-", "-", "-", "1e-4", "LN tuning(backbone仅LN+头训练)"),
    ("ft_methods", "FT_DoRA_r1", "43194aa6-adcc-429e-a2d6-a7a3df09d299",
     "-", "-", "-", "-", "-", "1e-4", "DoRA(HF PEFT语义,qv块,r1)"),
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

    for group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note in FT_RUNS:
        row = [group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note]
        for split in ["sunlamp", "lightbox", "validation"]:
            am, asd, dm = load_metrics(uuid, split)
            row += [
                f"{am:.4f}" if am is not None else "",
                f"{asd:.4f}" if asd is not None else "",
                f"{dm:.4f}" if dm is not None else "",
            ]
        row.append(COMMON_FT)
        rows.append(row)

    for group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note in FT_RUNS2:
        row = [group, name, uuid, enable, num_f, epochb, lrbl, lambdap, lr, note]
        for split in ["sunlamp", "lightbox", "validation"]:
            am, asd, dm = load_metrics(uuid, split)
            row += [
                f"{am:.4f}" if am is not None else "",
                f"{asd:.4f}" if asd is not None else "",
                f"{dm:.4f}" if dm is not None else "",
            ]
        row.append(COMMON_FT2)
        rows.append(row)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"written {len(rows)} rows -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
