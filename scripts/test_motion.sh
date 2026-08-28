#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/tooth_motion_nonsmoothed_20steps_5xpure_synthetic_newckpt/ckpt/ckpt-epoch=968-val_total_loss=59.3741.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_process_nonsmoothed10x_newckpt}"
INDEX_PATH="${INDEX_PATH:-files/ours_process}"
run_gendental test configs/stage_prediction.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
