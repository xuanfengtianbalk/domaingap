#!/bin/bash
# mlp_ablation.sh — Sweep MLP hyperparameters, record epoch losses
# Run: bash mlp_ablation.sh from the analysis/ directory

set -uo pipefail

cd "$(dirname "$0")"
source /home/liukun/miniconda3/etc/profile.d/conda.sh
conda activate dinov3

OUT_DIR="../outputs/alpha_cali/mlp_ablation"
mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parse logs into CSV
CSV_PATH="$OUT_DIR/ablation_results.csv"
echo "run,mlp_raw,mlp_keep_center,mlp_loss,lr,epoch,train,raw_train,val,raw_val" > "$CSV_PATH"

RUN=0
for RAW in 0; do
for KEEP in 0 1; do
for LOSS in l1; do
for LR in 1e-3 1e-2; do
    RUN=$((RUN + 1))
    RAW_FLAG=""
    KEEP_FLAG=""
    [ "$RAW" = "1" ] && RAW_FLAG="--mlp_raw"
    [ "$KEEP" = "1" ] && KEEP_FLAG="--mlp_keep_center"

    LOG_FILE="$LOG_DIR/run_${RUN}_raw${RAW}_keep${KEEP}_${LOSS}_lr${LR}.log"
    echo "=== RUN $RUN: raw=$RAW keep=$KEEP loss=$LOSS lr=$LR ==="

    python -u alpha_calibration.py \
        --train_mlp \
        --mlp_epochs 50 \
        --mlp_batch 16 \
        --mlp_lr "$LR" \
        --mlp_loss "$LOSS" \
        --max_samples 10000 \
        $RAW_FLAG $KEEP_FLAG \
        > "$LOG_FILE" 2>&1

    # Parse epoch lines into CSV
    grep "epoch" "$LOG_FILE" | while read -r line; do
        # epoch   0: train=0.0264(raw=0.0021) val=0.0162(raw=0.0006)
        epoch=$(echo "$line" | grep -oP 'epoch\s+\K\d+')
        train=$(echo "$line" | grep -oP 'train=\K[0-9.]+')
        raw_tr=$(echo "$line" | grep -oP 'train=[^(]+\(raw=\K[0-9.]+')
        val=$(echo "$line" | grep -oP 'val=\K[0-9.]+')
        raw_val=$(echo "$line" | grep -oP 'val=[^(]+\(raw=\K[0-9.]+')
        echo "$RUN,$RAW,$KEEP,$LOSS,$LR,$epoch,$train,$raw_tr,$val,$raw_val" >> "$CSV_PATH"
    done

    echo "  → log: $LOG_FILE"
done
done
done
done

echo ""
echo "Done! $RUN runs completed."
echo "CSV: $CSV_PATH"
echo "Models in: $OUT_DIR/"
