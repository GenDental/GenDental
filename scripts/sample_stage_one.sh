#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# Frequently changed generation settings live here, not in the YAML file.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
CKPT_PATH="${CKPT_PATH:-/data3/leics/dataset/checkpoints/Gendental/stage_one_zj_old_stochastic/ckpt/last-v1.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/GenDental/old_gpt_stochastic_samples2}"
NUM_SAMPLES="${NUM_SAMPLES:-800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_MERGED="${SAVE_MERGED:-true}"
REFERENCE_DATA_PATH="${REFERENCE_DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
REFERENCE_INDEX_PATH="${REFERENCE_INDEX_PATH:-files/zj}"
CENTER_NOISE_SCALE="${CENTER_NOISE_SCALE:-0.0001}"
# reference or autoregressive
CENTER_MODE="${CENTER_MODE:-no}"

run_gendental generate configs/stage_one.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "generation.output_dir=$OUTPUT_DIR" \
  --set "generation.num_samples=$NUM_SAMPLES" \
  --set "generation.batch_size=$BATCH_SIZE" \
  --set "generation.save_merged=$SAVE_MERGED" \
  --set "generation.reference_data_path=$REFERENCE_DATA_PATH" \
  --set "generation.reference_index_path=$REFERENCE_INDEX_PATH" \
  --set "generation.center_noise_scale=$CENTER_NOISE_SCALE" \
  --set "generation.center_mode=$CENTER_MODE"
