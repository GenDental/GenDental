#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/Gendental/tmp}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/Gendental/TMDM1x_2/ckpt/ckpt-epoch=15-val_total_loss=0.562748.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_process_nonsmoothed10x_newckpt}"
INDEX_PATH="${INDEX_PATH:-files/ours_process}"
BATCH_SIZE="${BATCH_SIZE:-64}"

run_gendental test configs/TMDM.yaml \
  --output_dir "$OUTPUT_DIR" \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
