#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/alignment_zj_mix0.8synthetic_TANet/ckpt/ckpt-epoch=918-val_total_loss=61.6593.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_alignment_5x}"
INDEX_PATH="${INDEX_PATH:-files/alignment_zj}"
run_gendental test configs/arrangement.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
