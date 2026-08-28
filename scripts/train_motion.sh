#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/tooth_motion_nonsmoothed_20steps_3xpure_synthetic_newckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_process_nonsmoothed10x_newckpt}"
INDEX_PATH="${INDEX_PATH:-files/ours_process}"
run_gendental train configs/stage_prediction.yaml \
  --output_dir "$OUTPUT_DIR" --epochs 1000 --base_lr 1e-4 --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
