#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Frequently changed generation settings live here, not in the YAML file.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/stage_one_zj/ckpt/last-v1.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/GenDental/zj_synthetic_samples}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_MERGED="${SAVE_MERGED:-true}"

run_gendental generate configs/stage_one.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "generation.output_dir=$OUTPUT_DIR" \
  --set "generation.num_samples=$NUM_SAMPLES" \
  --set "generation.batch_size=$BATCH_SIZE" \
  --set "generation.save_merged=$SAVE_MERGED"
