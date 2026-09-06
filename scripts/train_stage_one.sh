#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/stage_one_zj_old_stochastic}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
INDEX_PATH="${INDEX_PATH:-files/zj}"
EPOCHS="${EPOCHS:-1000}"
BASE_LR="${BASE_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run_gendental train configs/stage_one.yaml \
  --output_dir "$OUTPUT_DIR" --epochs "$EPOCHS" \
  --monitor val_total_loss \
  --base_lr "$BASE_LR" --fast \
  --gradient_clip_val 1.0 --gradient_clip_algorithm norm \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
