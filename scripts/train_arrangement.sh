#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/ToothWise/alignment_zj_mix0.6synthetic_TANet}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_alignment_5x}"
INDEX_PATH="${INDEX_PATH:-files/alignment_zj}"
run_gendental train configs/arrangement.yaml \
  --output_dir "$OUTPUT_DIR" --epochs 1000 --base_lr 1e-4 --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH"
