#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/ToothWise/motion_transfer3/ckpt/last.ckpt}"
STYLE_DIR="${STYLE_DIR:-reference_data}"
DATA_DIR="${DATA_DIR:-gpt_samples}"
OUTPUT_DIR="${OUTPUT_DIR:-stage_two_samples}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-3407}"

run_gendental generate configs/stage_two_sample.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "generation.style_dir=$STYLE_DIR" \
  --set "generation.data_dir=$DATA_DIR" \
  --set "generation.output_dir=$OUTPUT_DIR" \
  --set "generation.batch_size=$BATCH_SIZE" \
  --set "generation.seed=$SEED"
