#!/bin/bash
cd /opt/dl_workspace/algorithm/04-myself/domaingap
source /home/liukun/miniconda3/etc/profile.d/conda.sh
conda activate dinov3

UUIDS=$(ls -d workingdir/*/model_final.pth 2>/dev/null | while read f; do basename $(dirname "$f"); done)

for uuid in $UUIDS; do
    for aug in aug4 aug4s; do
        echo "=== $uuid | $aug ==="
        python imageprocess/pca_stages.py \
          --model_uuid $uuid \
          --m 10 --n 100 --aug_type $aug \
          --out_name "imageprocess/pca_${uuid}_${aug}.png" \
          2>&1 | tail -1
    done
done
echo "ALL DONE"
