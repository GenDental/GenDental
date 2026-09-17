#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/Gendental/tooth_motion3x}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/Gendental/tooth_motion0x/ckpt/ckpt-epoch=274-val_total_loss=49.991379.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_process_nonsmoothed10x_newckpt}"
INDEX_PATH="${INDEX_PATH:-files/ours_process}"
run_gendental train configs/stage_prediction.yaml \
  --output_dir "$OUTPUT_DIR" --epochs 500 --base_lr 1e-4 --fast \
  --ckpt_path "$CKPT_PATH" \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
