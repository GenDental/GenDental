#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/stage_one_zj_direct}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/stage_one_zj/ckpt/last-v1.ckpt}"
INDEX_PATH="${INDEX_PATH:-files/zj}"
EPOCHS="${EPOCHS:-1500}"
BASE_LR="${BASE_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run_gendental train configs/stage_one.yaml \
  --output_dir "$OUTPUT_DIR" --epochs "$EPOCHS" \
  --monitor val_prior_rec \
  --base_lr "$BASE_LR" --fast \
  --gradient_clip_val 1.0 --gradient_clip_algorithm norm \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
