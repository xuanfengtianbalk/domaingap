"""Phase A: LaSt-ViT faithful probe (paper metrics only).

Paper metrics (arXiv 2602.22394):
  - Patch Score (Eq. 3): cosine similarity between patch features and the
    global representation (CLS token for ViT encoder; GAP for decoder).
  - Stability Score (Eq. 4-5) + channel-wise top-K vote count (Eq. 8).
  - Point-in-Box (PiB): does the highest-scoring patch fall inside the
    foreground region? Here the "foreground box" is the satellite mask.

Run: python lastvit_probe.py [--splits sunlamp lightbox] [--max_images 200]
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))

from analysis_utils import load_model, get_model_type_from_traininfo, build_dataloader
from Hyperpose_net.losses.lastvit import StabilityScore

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lastvit_probe")
GRID = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default="e24d72fb-b4d7-4c40-b00a-a4aa2f8213e8")
    ap.add_argument("--splits", nargs="*", default=["sunlamp", "lightbox"])
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lastvit", action="store_true",
                    help="enable LaSt-ViT context injection in the decoder")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="injection strength when --lastvit")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    mt, bb = get_model_type_from_traininfo(args.uuid)
    model_type = mt[0]
    print(f"Model: {model_type} on {bb}  lastvit_context={args.lastvit} gamma={args.gamma}")
    model, _ = load_model(args.uuid, model_type, bb, args.device,
                          lastvit_context=args.lastvit)
    if args.lastvit:
        model.decoder[model_type].lastvit_gamma = args.gamma
    model.eval()

    stab = StabilityScore(256).to(args.device)

    pib = defaultdict(list)

    def top1_in_mask(scores, mask):
        """PiB for one image: highest-score patch inside foreground mask?"""
        s = scores.reshape(-1)
        m = mask.reshape(-1).cpu()
        return float(m[int(s.argmax().cpu())])

    def frac_top_in_mask(scores, mask, q=0.1):
        s = scores.reshape(-1)
        m = mask.reshape(-1).cpu()
        k = max(1, int(len(s) * q))
        idx = torch.topk(s, k).indices.cpu()
        return float(m[idx].float().mean())

    for split in args.splits:
        print(f"\n=== {split} ===")
        dl = build_dataloader(args.uuid, split, max_samples=args.max_images, batch_size=1)
        for samples, targets in tqdm(dl, desc=split, ncols=80):
            image = samples.to(args.device)
            mask_gt = targets["mask_gt"][0] > 0.5          # [256, 256]
            if mask_gt.sum() < 100:
                continue

            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs, features, dec_hier = model(image, return_features=True)

            # ── 1. Patch Score (encoder): patch feats vs CLS token ──
            patch_feats, cls_token = features[-1]          # [1, C, h, w], [1, C]
            C, h, w = patch_feats.shape[1], patch_feats.shape[2], patch_feats.shape[3]
            pf = patch_feats.flatten(2).permute(0, 2, 1)   # [1, h*w, C]
            ct = cls_token.unsqueeze(1).expand(-1, h * w, -1)
            ps_enc = F.cosine_similarity(pf, ct, dim=-1).reshape(h, w)  # [h, w]
            mask_enc = F.adaptive_avg_pool2d(
                mask_gt.float().unsqueeze(0).unsqueeze(0), (h, w)).squeeze() > 0.5

            # ── 2. Patch Score (decoder): cell feats vs GAP (implicit CLS) ──
            hier = next(iter(dec_hier.values()))
            last_stage = hier[-1].float()                  # [1, 256, 64, 64]
            cell_feat = F.adaptive_avg_pool2d(last_stage, (GRID, GRID))
            feat = cell_feat.permute(0, 2, 3, 1).reshape(-1, 256)   # [g*g, 256]
            gap = feat.mean(dim=0, keepdim=True)
            ps_dec = F.cosine_similarity(feat, gap.expand(feat.shape[0], -1), dim=-1)  # [g*g]
            mask_cell = F.adaptive_avg_pool2d(
                mask_gt.float().unsqueeze(0).unsqueeze(0), (GRID, GRID)).squeeze() > 0.5

            # ── 3. LaSt-ViT: stability + vote ──
            S = stab(feat)                                  # [g*g, 256]
            stab_cell = S.mean(dim=-1)                      # [g*g]
            vote = stab.vote_count(feat)                    # [g*g]

            pib["ps_enc_top1_in"].append(top1_in_mask(ps_enc, mask_enc))
            pib["ps_dec_top1_in"].append(top1_in_mask(ps_dec, mask_cell))
            pib["stab_top1_in"].append(top1_in_mask(stab_cell, mask_cell))
            pib["vote_top1_in"].append(top1_in_mask(vote, mask_cell))
            pib["ps_enc_top10_in"].append(frac_top_in_mask(ps_enc, mask_enc))
            pib["ps_dec_top10_in"].append(frac_top_in_mask(ps_dec, mask_cell))
            pib["stab_top10_in"].append(frac_top_in_mask(stab_cell, mask_cell))
            pib["vote_top10_in"].append(frac_top_in_mask(vote, mask_cell))

    results = {
        "split": list(args.splits),
        "max_images": args.max_images,
        "lastvit_context": args.lastvit,
        "n_images": len(pib["ps_enc_top1_in"]),
        "PiB": {k: float(np.mean(v)) for k, v in pib.items()},
    }
    out_json = os.path.join(OUT_DIR, "pib_results.json" if not args.lastvit
                            else f"pib_results_lastvit_g{args.gamma}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== PiB 结果 (n={results['n_images']} images, 卫星mask=前景) ===")
    for k, v in results["PiB"].items():
        print(f"  {k:16s}: {v:.3f}")
    print(f"\nsaved → {out_json}")


if __name__ == "__main__":
    main()
