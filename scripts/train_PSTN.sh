#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/tmp}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/PSTN_zj/ckpt/last-v1.ckpt}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
INDEX_PATH="${INDEX_PATH:-files/zj}"
EPOCHS="${EPOCHS:-500}"
BASE_LR="${BASE_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run_gendental train configs/PSTN.yaml \
  --output_dir "$OUTPUT_DIR" --epochs "$EPOCHS" \
  --base_lr "$BASE_LR" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
