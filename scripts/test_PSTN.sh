#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/Gendental/tmp}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/Gendental/PSTN0_to_5/5.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_alignment_5x}"
INDEX_PATH="${INDEX_PATH:-files/alignment5x}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run_gendental test configs/PSTN.yaml \
  --output_dir "$OUTPUT_DIR" \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
