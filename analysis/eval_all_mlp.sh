#!/bin/bash
# eval_all_mlp.sh — Evaluate all MLP .pt files across eval param combinations
# Run from analysis/ directory

set -uo pipefail

cd "$(dirname "$0")"
source /home/liukun/miniconda3/etc/profile.d/conda.sh
conda activate dinov3

# ── Model params (determine .pt file path) ──
LOSSES=("l1" "l2" "smooth_l1")
LRS=("1e-3" "5e-4" "2e-4")
RAWS=(0 1)
KEEPS=(0 1)

# ── Eval params (passed to alpha_eval.py) ──
EVAL_SPLITS=("sunlamp" "lightbox")
STD_EXCL_MIN_VALS=(0.01)
CORR_EXCL_VALS=(0 1)
STD_MIN_VALS=(0.0)
STD_MAX_VALS=(1000.0 )
MAX_SAMPLES=10000   # reduce for faster testing, set to 10000 for full eval

# ── Output ──
OUT_DIR="../outputs/alpha_cali/mlp_eval"
mkdir -p "$OUT_DIR"
CSV_PATH="$OUT_DIR/mlp_eval_results.csv"

echo "pt_file,split,raw,keep,loss,lr,std_min,std_max,std_excl_min,corr_excl,baseline_angle_mean,baseline_angle_std,baseline_dist_mean,baseline_n,excluded_angle_mean,excluded_angle_std,excluded_dist_mean,excluded_n,corrected_angle_mean,corrected_angle_std,corrected_dist_mean,corrected_n,unc_angle_mean,unc_angle_std,unc_dist_mean,unc_n,unc_corr_angle_mean,unc_corr_angle_std,unc_corr_dist_mean,unc_corr_n" > "$CSV_PATH"

TOTAL=0
SKIPPED=0

for RAW in "${RAWS[@]}"; do
for KEEP in "${KEEPS[@]}"; do
for LOSS in "${LOSSES[@]}"; do
for LR in "${LRS[@]}"; do
    # map shell KEEP (0=excl, 1=noexcl) → filename keep (1=excl, 0=noexcl)
    KEEP_FILE=$((1 - KEEP))

    # convert lr 1e-3 → 0.001, 5e-4 → 0.0005, 2e-4 → 0.0002
    LR_FMT=$(python -c "print($LR)")

    FNAME="alpha_mlp_fgsm_augmix_${LOSS}_lr${LR_FMT}_raw${RAW}_keep${KEEP_FILE}_excl_cx0.0_cy0.04_cz0.165_rx0.05_ry0.05_rz0.05"
    PT="../outputs/alpha_cali/${FNAME}.pt"

    if [ ! -f "$PT" ]; then
        echo "SKIP $FNAME (missing .pt)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "====================================================================="
    echo ">>> $FNAME"

    for SPLIT in "${EVAL_SPLITS[@]}"; do
    for STD_EXCL in "${STD_EXCL_MIN_VALS[@]}"; do
    for CORR in "${CORR_EXCL_VALS[@]}"; do
    for SMIN in "${STD_MIN_VALS[@]}"; do
    for SMAX in "${STD_MAX_VALS[@]}"; do
        TOTAL=$((TOTAL + 1))

        CORR_FLAG=""
        [ "$CORR" = "1" ] && CORR_FLAG="--corr_excl"

        echo "  [$TOTAL] split=$SPLIT corr_excl=$CORR"

        python -u alpha_eval.py \
            --alpha_mlp "$PT" \
            --splits "$SPLIT" \
            --max_samples "$MAX_SAMPLES" \
            --std_excl_min "$STD_EXCL" \
            --std_min "$SMIN" \
            --std_max "$SMAX" \
            --no_sweep \
            $CORR_FLAG \
            2>&1 | grep -E "^(===|  BASELINE|  EXCLUDED|  CORRECTED|  UNC|  →|Loaded|Model:|Error|Traceback)" || true

        # Parse output JSON
        JSON_PATH="../outputs/alpha_eval/${SPLIT}.json"
        if [ -f "$JSON_PATH" ]; then
            ROW=$(python -c "
import json
with open('$JSON_PATH') as f:
    d = json.load(f)
b = d['BASELINE']
c = d['CORRECTED']
e = d.get('EXCLUDED', {})
u = d.get('UNC', {})
uc = d.get('UNC_CORR', {})
row = [
    '${FNAME}', '${SPLIT}', '${RAW}', '${KEEP}', '${LOSS}', '${LR}',
    '${SMIN}', '${SMAX}', '${STD_EXCL}', '${CORR}',
    b['angle']['mean'], b['angle']['std'], b['dist']['mean'], b['angle']['n'],
    e.get('angle',{}).get('mean',''), e.get('angle',{}).get('std',''), e.get('dist',{}).get('mean',''), e.get('angle',{}).get('n',''),
    c['angle']['mean'], c['angle']['std'], c['dist']['mean'], c['angle']['n'],
    u.get('angle',{}).get('mean',''), u.get('angle',{}).get('std',''), u.get('dist',{}).get('mean',''), u.get('angle',{}).get('n',''),
    uc.get('angle',{}).get('mean',''), uc.get('angle',{}).get('std',''), uc.get('dist',{}).get('mean',''), uc.get('angle',{}).get('n',''),
]
print(','.join(str(x) for x in row))
" 2>/dev/null)
            if [ -n "$ROW" ]; then
                echo "$ROW" >> "$CSV_PATH"
            fi
            # backup per-run JSON
            mkdir -p "$OUT_DIR/${FNAME}"
            cp "$JSON_PATH" "$OUT_DIR/${FNAME}/${SPLIT}.json"
        fi
    done; done; done; done; done
done; done; done; done

echo ""
echo "Done: $TOTAL eval(s), $SKIPPED skipped"
echo "CSV: $CSV_PATH"
