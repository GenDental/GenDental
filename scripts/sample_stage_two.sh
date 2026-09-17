#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
TASK_MODE="${TASK_MODE:-target}"
case "$TASK_MODE" in
  target|motion) ;;
  *) echo "TASK_MODE must be target or motion" >&2; exit 2 ;;
esac
if [[ "$TASK_MODE" == "target" ]]; then
  DEFAULT_CKPT_PATH="/data3/leics/dataset/checkpoints/GenDental/stage_two_target/ckpt/last.ckpt"
else
  DEFAULT_CKPT_PATH="/data3/leics/dataset/checkpoints/GenDental/stage_two_motion/ckpt/last.ckpt"
fi
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
if [[ -z "$CKPT_PATH" ]]; then
  echo "CKPT_PATH is required when TASK_MODE=target" >&2
  exit 2
fi
STYLE_DIR="${STYLE_DIR:-./reference_data}"
DATA_DIR="${DATA_DIR:-./stage_one_samples}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage_two_samples}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-3407}"

run_gendental generate configs/stage_two_sample.yaml \
  --ckpt_path "$CKPT_PATH" --fast \
  --set "model.params.task_mode=$TASK_MODE" \
  --set "generation.style_dir=$STYLE_DIR" \
  --set "generation.data_dir=$DATA_DIR" \
  --set "generation.output_dir=$OUTPUT_DIR" \
  --set "generation.batch_size=$BATCH_SIZE" \
  --set "generation.seed=$SEED"
