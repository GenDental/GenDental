#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth3}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth2/ckpt/last.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
INDEX_PATH="${INDEX_PATH:-files/motion}"
run_gendental train configs/stage_two.yaml \
  --output_dir "$OUTPUT_DIR" --ckpt_path "$CKPT_PATH" \
  --epochs 2000 --base_lr 1e-5 --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
